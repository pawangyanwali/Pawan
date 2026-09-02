# RFI/RFP Response System: Architecture

Status: design, pre-build
Last updated: 2026-09-02

## Decisions

| Question | Answer | Consequence |
|---|---|---|
| Data classification | Commercial, no CUI | AWS commercial regions. No FedRAMP boundary. |
| Platform | Serverless-native. No Kubernetes. | Lambda container images, Step Functions, SageMaker Async. Nothing to patch, no cluster to hand off. |
| Bulk inference | Amazon Bedrock, Qwen3-32B | ~$5/month. Zero ops, zero cold start, multi-AZ. |
| Long-context and vision inference | Self-hosted Qwen3.8-27B on SageMaker Async, scale to zero | 262K context for RFP shred, vision encoder for scanned pages. ~$25/month. |
| Vector store | Amazon S3 Vectors | GA December 2025. 500K chunks is a rounding error on a 2B-per-index limit. Removes Aurora. |
| State store | DynamoDB, on-demand | Multi-AZ by default. Nothing to fail over. |
| Document store | SharePoint Online (M365) | Microsoft Graph, Entra ID, on-behalf-of flow for per-user permission trimming. |
| Corpus | 500 to 2,000 bid packages | Enough for a LoRA adapter and a real held-out evaluation set. |
| Pricing data | Labor rate cards in Excel | Deterministic calculator. Price-to-win waits for competitor award data. |
| Writers | 2 to 3, ~10 pursuits/month | Peak concurrency of 3. Sizes everything. |
| Latency budget for a drafted section | 2 to 5 minutes | Queue-and-return. Nothing here is a chat. |
| Writer surface | Word, round-trip required | Office.js add-in plus a template-driven docx renderer. |
| Environments | Production only | Canary deploys with automatic rollback. See RELIABILITY.md. |
| Ops model | Build now, hand off later | Every choice favors fewer moving parts over peak performance. |

Cost: roughly **$90/month** at this volume. See [SCALING.md](SCALING.md).
Failure modes and recovery: see [RELIABILITY.md](RELIABILITY.md).

## Inference routing

One gateway, two backends, routed by what the call actually needs.

| Workload | Backend | Why |
|---|---|---|
| Section drafting | Bedrock, Qwen3-32B | 95% of calls. No cold start, no capacity risk during crunch week. |
| Q&A over past bids | Bedrock, Qwen3-32B | Facts come from retrieval. The model composes and cites. |
| Query rewriting, classification, metadata extraction | Bedrock, Qwen3-32B | Small, frequent, latency-sensitive. |
| RFP shred | SageMaker Async, Qwen3.8-27B | Needs the 262K window. A 500-page solicitation reads in one pass. |
| Vision parsing of scanned pages | SageMaker Async, Qwen3.8-27B | Built-in vision encoder. Replaces an OCR pipeline. |
| Embeddings | Lambda, Qwen3-Embedding-0.6B (ONNX) | Runs on CPU. Never touches a GPU. |
| Reranking | Lambda, Qwen3-Reranker-0.6B (ONNX) | 2 to 4 seconds for 50 chunks on CPU. Fine at this volume. |

Every caller talks to `rfp_common.llm.InferenceGateway` and names a **capability**, not a model. The gateway maps capability to backend, handles retries, and fails over to the other backend when one is unavailable. Switching a workload from Bedrock to self-hosted is a table entry, not a code change.

### Why Qwen3.8-27B is not on Bedrock

Bedrock Custom Model Import supports Qwen2, Qwen2_VL, and Qwen2_5_VL architectures. Qwen3.8 is not importable. Bedrock's managed Qwen lineup covers Qwen3-32B, Qwen3-235B-A22B, Qwen3 Next 80B A3B, and the Coder variants, none of which is Qwen3.8.

So the two properties that made Qwen3.8-27B the right pick, 262K context and a native vision encoder, are only available if you host it. Hosting it for every call costs $150 to $250/month. Hosting it for the two jobs that need it costs $25. That's the split.

Re-check this at each Bedrock release. If Qwen3.8-27B lands as a managed model, the self-hosted endpoint deletes itself and the gateway table changes by one line.

## Service decomposition

Nine units. Each is a container image, each has one job, each talks to the others through a versioned contract in `rfp_common.contracts`. That package is the modularity. Not the count of services.

| Unit | Runtime | Trigger | Responsibility |
|---|---|---|---|
| `api` | Lambda (container) | API Gateway HTTP API | BFF for the web app and Word add-in. Entra token validation, on-behalf-of exchange. |
| `worker-ingest` | Lambda (container) | SQS `ingest-jobs` | Pull a file from Graph, write to S3 raw, emit a parse job. |
| `worker-parse` | Lambda (container) | SQS `parse-jobs` | PyMuPDF for text-layer PDFs, python-docx for Word, hand scanned pages to the vision endpoint. |
| `worker-index` | Lambda (container) | SQS `index-jobs` | Structure-aware chunking, embed, write to S3 Vectors and DynamoDB. |
| `retrieval` | Lambda (container) | Invoked by `api` and `worker-draft` | Vector query, metadata filter, rerank, ACL check. |
| `worker-draft` | Lambda (container) | SQS `draft-jobs` | One section per message. Retrieve, generate, verify citations, persist. |
| `shred` | Step Functions + SageMaker Async | API or SQS | Full-RFP read, requirements matrix, checkpointed per section. |
| `render` | Lambda (container) | Invoked by `api` | Section JSON to .docx against the template. Also reads .docx back in. |
| `compute` | Lambda (container) | Invoked by `api` | Deterministic engines: page counts, format checks, cost calculator. No model involved. |

`shred` is Step Functions rather than Lambda because a 262K-token read plus a per-section matrix build runs past Lambda's 15-minute ceiling. Step Functions submits to the async endpoint, waits on the SNS callback, and checkpoints each completed section to DynamoDB, so a failure resumes rather than restarting the solicitation.

Nothing runs continuously. Idle cost across all nine is zero.

## Data layer

**S3 Vectors** holds the vector index. One index, 500K to 1M chunks, filterable metadata: agency, NAICS, contract vehicle, submit date, outcome, document type. Queries return chunk IDs and metadata.

**DynamoDB**, on-demand billing, single table with a composite key. Holds chunk text keyed by chunk ID, document metadata, the ACL index, requirements matrices, rate card versions, generated section versions, and job state. Point-in-time recovery on. On-demand means idle cost is storage only.

**S3** holds three buckets: `raw` (SharePoint mirror, versioned), `parsed` (extracted text and page images), `output` (generated .docx). All SSE-KMS, all versioned.

**Athena** over scheduled DynamoDB exports to S3 for the analytics you'll want later: win rate by agency, which past-performance write-ups get reused most, cost per drafted section. Serverless, pennies, and it keeps analytical queries off the operational store.

No relational database. The three things that wanted SQL, requirements matrices, rate cards, and section versions, are all accessed by a single partition key (RFP ID, card version, section ID) and computed in Python. DynamoDB is the better fit and it removes an instance that can fail.

### Losing BM25

S3 Vectors is vector-only. Hybrid search with a BM25 leg is not available, and that costs exact-phrase recall.

The mitigation is that almost everything you want exact matching on is ID-shaped: solicitation numbers, CLINs, contract numbers, NAICS codes, agency names. Those become indexed metadata fields and exact-match lookups in DynamoDB, not full-text queries. Free-text phrase search over prose is the remaining gap. At 300 queries a month it has not been worth a $350/month OpenSearch Serverless collection. Revisit if writers start complaining they can't find a phrase they remember.

## SharePoint integration

**Two identities.**

App-only for the crawler. Entra app registration with `Sites.Selected` granted per bid library, not tenant-wide read. Certificate auth, not a client secret.

Delegated for user queries. The Word add-in calls `Office.auth.getAccessToken()`, the `api` Lambda validates that token and runs an on-behalf-of exchange for a Graph token as that user.

**Incremental sync.** Graph delta queries per document library, deltaLink persisted in DynamoDB. A nightly EventBridge rule sweeps for anything a webhook missed. Graph change notifications drive near-real-time updates on active pursuit folders, renewed on the expiry Graph reports.

**Permission trimming, two layers.** Index the readable Entra object IDs alongside every chunk and filter on that in the S3 Vectors query. Then re-verify only the final top-k documents against Graph as the calling user before anything renders. The filter keeps the query fast; the top-k check catches ACL drift between crawls. Eight Graph calls per query instead of thousands.

## Retrieval

Query rewrite through Bedrock, embed on the retrieval Lambda, S3 Vectors query with metadata filters for top 50, rerank to top 8 on CPU, ACL re-verify, return with source anchors.

**Chunking is structure-aware.** Split on the heading hierarchy python-docx reads out of the style names. A subsection stays whole up to a token limit. Every chunk carries its full heading path prepended to the embedded text, plus document ID, page anchor, and bid metadata. Fixed 512-token windows cut a past-performance write-up in half and hand you two chunks that each look complete and are both wrong.

**Wins get boosted, losses stay reachable.** Default ranking favors wins and recency. A writer who explicitly asks what didn't work on a past pursuit gets the losses.

## Generation

Section at a time. Never a whole volume in one call.

Each `worker-draft` message carries one requirement. The call gets the verbatim requirement text, the Section M criteria it maps to, the top 8 reranked chunks with source IDs, the outline slot, and the page limit.

**Citations are mandatory.** Every factual claim renders with a source ID pointing at a real chunk. The UI marks uncited sentences. A writer can accept one, and they have to look at it first.

**Facts come from retrieval. Always.** Contract numbers, period of performance, CPARS ratings, staff names, dollar values, and place of performance are retrieved and cited, never generated. A fine-tuned adapter would learn your voice, not your contract history.

## Compliance

**Shred** reads the full RFP in one 262K-token pass on the self-hosted endpoint and writes a structured matrix to DynamoDB: requirement ID, source reference, verbatim text, type, volume, page limit, owner, response location, status.

**Cross-check** verifies after drafting that every requirement has a response location and that the response addresses it. Gaps go on a report the capture manager reads before pink team.

**Format checks are Python.** Page counts, font size, margins, line spacing, file naming, and size limits are arithmetic and string matching. Do not ask a model to count pages. It will be confidently wrong and you will submit a 31-page volume against a 30-page limit.

## Pricing

Rate cards load into DynamoDB, versioned. A proposal priced in March references the card that was live in March.

**The model proposes, Python computes.** Qwen3-32B reads the SOW and proposes labor categories and hours by period. The `compute` Lambda does every multiplication, escalation, and rollup. The model writes the BOE narrative explaining the hours.

No arithmetic that lands in a Volume III comes out of a language model. A transposed digit in a cost volume is not a quality problem, it's a protest.

Every rate the narrative cites gets validated against the card version. Mismatches block export.

## Word round-trip

**Generation emits structure, not markdown.** The model returns JSON: heading level, paragraph runs, tables, lists, citation anchors. The `render` Lambda maps that to python-docx calls against your `.dotx`, so headings, numbering, headers, footers, and section breaks come from the template you already use. Markdown-to-Word loses your numbering scheme on the first heading.

**Inbound, styles survive.** An edited section comes back, python-docx reads the style names and heading hierarchy out, and it re-indexes and diffs against what the machine wrote.

**Insert as tracked changes.** Generated content lands as tracked insertions. The writer sees exactly what came from the machine. This one feature does more for adoption than any accuracy improvement.

**The add-in** is an Office.js task pane running in Word desktop and Word on the web, authenticating through Entra SSO into the `api` on-behalf-of flow.

**Admin consent is a gate.** Deploying an Office add-in to your tenant needs an M365 admin to approve it in Integrated Apps. Start that conversation in week one. It's a two-day task that becomes a three-week task if you raise it in month four.

## Phasing

Drafting is Phase 1, and drafting cannot ship without retrieval and shred underneath it. What follows puts drafting in Phase 1 by building what it stands on.

**Phase 1, weeks 1 to 12. Draft generation with citations.**
Entra app registration and Graph connector. Parsing including the vision pass. Metadata and outcome labeling as a parallel human track starting week 1. Chunking, S3 Vectors index, retrieval, reranking. RFP shred producing the matrix. Section drafting with mandatory citations. The docx renderer against your template. Word add-in v1. All Terraform.

**Phase 2, weeks 13 to 19. Compliance and round-trip.**
Requirements-to-response cross-check with a gap report. Deterministic format and page-limit validation. Tracked-changes insertion. Inbound docx re-ingestion and diffing. Pink team scoring against Section M.

**Phase 3, weeks 20 to 26. Pricing.**
Rate card ingestion and versioning. Staffing plan proposal from the SOW. Deterministic cost calculator with escalation and wrap. BOE narrative generation. Rate validation blocking export on mismatch.

**Phase 4, weeks 27 to 33. Training and handoff.**
Training pair extraction from the labeled corpus. LoRA run against Qwen3.8-27B and the evaluation harness. Blind preference test, ship-or-drop decision. Operations runbook and handoff.

Three weeks shorter than the EKS plan, entirely because there is no cluster to build.

## Evaluation

Four measures, all of them running before the fine-tuning question comes up.

1. **Compliance recall.** Percentage of Section L requirements the shred catches, against a human-built matrix for the same solicitation. Target above 98%. A missed submission requirement is a non-responsive bid.
2. **Citation accuracy.** Percentage of generated factual claims that trace to a chunk containing that fact. Sampled and human-checked. Under 95% is not shippable.
3. **Blind preference.** A capture manager reads two drafts of the same section without knowing which came from where, and picks.
4. **Numeric accuracy.** Every figure in a generated cost narrative matched against the deterministic calculator. Tolerance is zero.

Hold out 50 bid packages the system never indexes, losses included. A system evaluated only against wins tells you nothing about whether it's learning anything.

## Open items

| Item | Gates | Owner |
|---|---|---|
| Outcome labeling: who, starting when | Retrieval quality, everything downstream | Needs a named person |
| M365 admin consent for the Office add-in | Phase 1 delivery | Start week 1 |
| SharePoint site inventory, which libraries the crawler gets | Ingestion | M365 admin |
| Your `.dotx` proposal templates | Phase 1 renderer | Proposal ops |
| Rate card owner and update cadence | Phase 3 | Contracts or finance |
| Bedrock enablement for Qwen3-32B in your account and region | Phase 1 | AWS account admin |
| Who receives the handoff | Phase 4 runbook scope | Leadership |
