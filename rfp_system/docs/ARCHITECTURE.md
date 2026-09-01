# RFI/RFP Response System: Architecture

Status: design, pre-build
Last updated: 2026-09-01

## Decisions already made

| Question | Answer | What it forces |
|---|---|---|
| Data classification | Commercial, no CUI | AWS commercial regions. No FedRAMP boundary, no GovCloud procurement wait. |
| Model hosting | Self-hosted vLLM on EKS | You own GPU capacity, node lifecycle, and on-call. Flat cost at volume. |
| Document store | SharePoint Online (M365) | Microsoft Graph, Entra ID, on-behalf-of token flow for per-user permission trimming. |
| Corpus size | 500 to 2,000 bid packages | Enough for a LoRA adapter and a real held-out evaluation set. |
| Pricing data | Labor rate cards in Excel | Deterministic cost calculator. Price-to-win is out of scope until competitor award data exists. |
| Phase 1 | First-draft generation | Retrieval and RFP shred have to ship underneath it. See "Phasing." |
| Ops model | Build now, hand off later | Every design choice favors fewer moving parts over peak performance. |
| Writer surface | Word, round-trip required | Office.js add-in plus a template-driven docx renderer. The hardest piece in the build. |

## Model stack

| Role | Model | License | Notes |
|---|---|---|---|
| Generation, shred, extraction | Qwen3.8-27B | Apache 2.0 | 28B dense, hybrid attention (3 gated DeltaNet linear blocks per 1 full-attention block, 64 layers), 262,144 native context, built-in vision encoder |
| Embeddings | Qwen3-Embedding-8B | Apache 2.0 | 32K input length. Drop to the 4B if GPU memory gets tight. |
| Reranking | Qwen3-Reranker-4B | Apache 2.0 | Reranks top 50 retrieved chunks down to top 8 |
| Page-image retrieval (Phase 2) | Qwen3-VL-Embedding | Apache 2.0 | For retrieving graphics, org charts, and scanned tables by visual content |

One text model, not two. Qwen3.8-27B handles generation, requirement extraction, and classification. A separate small extraction model would save a little GPU time and cost the handoff team another component to understand. Not worth the trade.

Verify every license with counsel before the first training run. Model licenses change between releases.

### Why not the 2.4T flagship

Qwen3.8-2.4T-A95B ships a 2.4TB FP8 checkpoint, requires all experts resident in HBM, and needs two B300-class nodes for inference. It also carries a custom Qwen3.8-Max license rather than Apache 2.0, with display requirements and a separate-agreement clause for model-as-a-service businesses above roughly $50M revenue. Neither the cost nor the license review is justified for an internal proposal tool.

### What the 262K context changes

A large DoD solicitation with amendments and attachments runs 500 to 1,500 pages, roughly 150K to 400K tokens. Most of them fit in one window.

This removes an entire class of bug. Chunked RFP processing loses the cross-reference between an L.3.2 submission instruction and the M.2.1 evaluation factor it maps to, because the two live in different chunks and neither chunk knows about the other. Reading the whole document in one pass keeps those links intact.

Retrieval is still required. The 500 to 2,000 past bid packages run into hundreds of millions of tokens and will never fit in any context window.

## System layout

```
SharePoint Online ──Graph delta──> Ingestion (EKS) ──> S3 raw (versioned, SSE-KMS)
                                          │
                                          ├──> Parse: PyMuPDF for text-layer PDFs,
                                          │           Qwen3.8-27B VL for scanned pages
                                          │           and graphics, python-docx for Word
                                          │
                                          ├──> Metadata + outcome extraction
                                          │           (agency, solicitation no., NAICS,
                                          │            vehicle, win/loss/no-bid, value)
                                          │
                                          └──> Structure-aware chunking
                                                      │
                                                      v
                                      Aurora PostgreSQL Serverless v2
                                      (pgvector HNSW + tsvector full-text
                                       + relational: rate cards, requirements
                                       matrices, ACL index, job state)
                                                      │
        Word add-in (Office.js) ──Entra SSO──> FastAPI on EKS ──> vLLM (Qwen3.8-27B)
        Web app                                     │              on g6e.12xlarge
                                                    └──> Deterministic engines:
                                                         page-count checker,
                                                         cost calculator,
                                                         format validator
```

## Ingestion and SharePoint integration

### Two identities, not one

**App-only identity** for the background crawler. Entra app registration with `Sites.Selected` granted per-site, not tenant-wide `Sites.Read.All`. Ask your M365 admin to grant only the bid libraries. A crawler with tenant-wide read is a finding waiting to happen in your next security review.

**Delegated identity** for user queries. The Word add-in calls `Office.auth.getAccessToken()`, sends that token to your API, and your API runs an on-behalf-of exchange to get a Graph token as that user. Every retrieval result gets checked against what that specific person can open.

### Incremental sync

Graph delta queries against each document library (`/drives/{driveId}/root/delta`). Persist the deltaLink per drive in Aurora. A full recrawl of 2,000 bid packages is an hour of work you only want to do once.

Add Graph change notifications (webhooks) for near-real-time updates on active bid folders. Subscriptions expire, so renew on the expiry Graph reports rather than a hardcoded interval. Fall back to a nightly delta sweep so a missed webhook never means a missed document.

### Permission trimming

Index the ACL alongside every chunk: the set of Entra group and user object IDs that can read the source file. Filter on that at query time. Then re-verify only the final top-k documents against Graph as the calling user before anything renders.

The index filter keeps queries fast. The top-k re-check catches ACL drift between crawls. You pay 8 Graph calls per query instead of thousands, and a permission revoked an hour ago still takes effect.

### The task nobody budgets for

Your 500 to 2,000 bid packages need outcome labels: win, loss, or no-bid. Without them the corpus is a pile of text and retrieval will happily surface a losing proposal's technical approach as your best example.

With them, retrieval can weight wins, the evaluation set is real, and a win/loss signal becomes trainable later.

Where SharePoint columns already carry the outcome, take them. Where they don't, run Qwen3.8-27B over the debrief letters and award notices and route anything under a confidence threshold to a human queue. Budget one person for two to three weeks on this. It is the highest-value data work in the project and it will not do itself.

## Retrieval

Aurora PostgreSQL Serverless v2 with pgvector, not OpenSearch. At 200K to 500K chunks, HNSW in pgvector is fast enough, and it puts your vectors, your bid metadata, your rate cards, your requirements matrices, and your job state in one database. The team taking this over learns one system instead of three.

**Hybrid search.** Dense vector similarity (pgvector HNSW) and PostgreSQL full-text (tsvector, GIN index) run in parallel, fused with reciprocal rank fusion. Vector search alone misses exact matches on solicitation numbers, CLINs, and contract numbers. Keyword search alone misses paraphrase. Proposal work needs both.

**Chunking is structure-aware, not fixed-size.** Split on the heading hierarchy that python-docx reads out of the style names. A subsection stays whole up to a token limit. Every chunk carries its full heading path ("Volume II > 3.2 Technical Approach > 3.2.4 Transition Plan") prepended to the embedded text, plus source document ID, page anchor, and bid metadata.

Fixed 512-token windows cut a past-performance write-up in half and hand you two chunks that each look complete and are both wrong.

**Filters and boosts.** Agency, NAICS, contract vehicle, date range, and outcome are all queryable. Default retrieval boosts wins and recency. A 2019 loss is still worth retrieving when a writer explicitly asks what didn't work.

**Rerank.** Top 50 from hybrid search, reranked by Qwen3-Reranker-4B, top 8 into the generation context.

## Generation

Section at a time. Never a whole volume in one call.

Every generation call gets:
- the specific requirement text from the shred, verbatim
- the Section M evaluation criteria that map to it
- the top 8 reranked past-content chunks with source IDs
- the outline slot and its page limit
- the style adapter (once Phase 4 ships)

**Citations are mandatory.** Every factual claim renders with a source ID pointing at a real chunk. The UI marks uncited sentences in yellow. A writer can accept an uncited sentence, and they have to look at it first.

This is the guardrail that makes the system usable in a real bid. A model that produces fluent past-performance narrative with an invented contract number will pass casual review and fail at the customer.

**Facts come from retrieval. Always.** Contract numbers, period of performance, CPARS ratings, staff names, dollar values, and place of performance get retrieved and cited, never generated. The fine-tuned adapter learns your voice and structure. It does not learn your contract history.

## Compliance

**Shred.** The full RFP goes into one 262K context call. Output is structured JSON written to Aurora:

```
requirement_id, source_ref (L.3.2.1), verbatim_text, requirement_type
(format | content | submission | evaluation), volume, page_limit,
owner, response_location, status
```

**Cross-check.** After drafting, a second pass verifies that every requirement has a response location and that the response text actually addresses it. Gaps go on a report the capture manager reads before pink team.

**Format checks are Python, not the model.** Page counts, font size, margins, line spacing, file naming, and file size limits are arithmetic and string matching. Do not ask an LLM to count pages. It will be confidently wrong and you will submit a 31-page volume against a 30-page limit.

**Pink team scoring (Phase 2).** A separate call scores each drafted section against its Section M criteria and lists what's missing. Cheap to run, and it catches the gap between "we answered the question" and "we answered it the way the evaluator scores it."

## Pricing

You have labor rate cards in Excel and nothing else structured. That sets the scope precisely.

**Ingest the rate cards into Aurora tables:** labor category, direct rate, escalation by option year, wrap rate, fringe, overhead, G&A, fee. Version them. When a rate card changes, old proposals keep referencing the card that was live when they were priced.

**The model proposes, Python computes.** Qwen3.8-27B reads the SOW and proposes a staffing plan: labor categories, hours by category by period. Python does every multiplication, escalation, and rollup. The model writes the BOE narrative explaining why those hours.

No arithmetic that lands in a Volume III comes out of a language model. Ever. A transposed digit in a cost volume is not a quality problem, it's a protest.

**Cross-check against the card.** Every rate the narrative cites gets validated against the current card version. Mismatches block export.

**Price-to-win is Phase 3+ and needs data you don't have.** It requires FPDS or USAspending award history for your competitors. Cheap to add later, impossible to fake now.

## Word round-trip

You picked the hardest option and the right one. Writers who have to leave Word stop using the tool by week three.

**Generation emits structure, not markdown.** The model returns JSON: heading level, paragraph runs, tables, lists, citation anchors. A renderer maps that to python-docx calls against your `.dotx` template, so headings, numbering, headers, footers, and section breaks all come from the template you already use. Markdown-to-Word conversion loses your numbering scheme and your compliance-mandated formatting on the first heading.

**Inbound, styles survive.** When a writer edits a section in Word and it comes back, python-docx reads the style names and heading hierarchy back out. The section re-indexes and diffs against what the machine wrote.

**Insert as tracked changes.** Generated content lands in the document as tracked insertions. The writer sees exactly what came from the machine and what they wrote themselves. This single feature does more for adoption than any accuracy improvement.

**The add-in.** Office.js task pane, runs in Word desktop and Word on the web. Entra SSO through `getAccessToken()` into your API's on-behalf-of flow. Task pane surfaces: search past bids, view the compliance matrix, generate a section, check citations.

**Admin consent is a gate.** Deploying an Office add-in to your tenant needs an M365 admin to approve it in Integrated Apps. Start that conversation in week one. It is a two-day task that becomes a three-week task if you raise it in month four.

## AWS infrastructure

Everything in Terraform. The handoff team gets state files, not a screenshot of a console.

**Network.** One VPC, private subnets for compute and data. No public ingress to the model. ALB in public subnets terminating TLS for the app tier only. VPC endpoints for S3, ECR, Secrets Manager, and CloudWatch so GPU nodes never route through a NAT gateway. NAT egress at GPU-node volume is a line item you will notice.

**EKS.** Karpenter for GPU node provisioning. Two node pools:

- `gpu-inference`: g6e.12xlarge (4x L40S, 48GB each, 192GB total). Minimum 2 nodes so a rolling update doesn't take the service down. FP8 quantized weights at roughly 30GB per replica, tensor parallel 1, four replicas per node, one per GPU. Higher throughput than TP=4 on a single replica and simpler to reason about.
- `app`: managed node group or Fargate for FastAPI, the ingestion workers, and the renderer.

Dev cluster scales GPU nodes to zero outside working hours. That is roughly 60% of the dev GPU bill.

**bf16 option.** If evaluation shows FP8 quantization costs measurable output quality, bf16 needs about 74GB with KV cache, so tensor parallel 2 across two L40S, two replicas per node instead of four. Measure before you decide. Run the same 50 held-out RFP sections through both and have a capture manager read them blind.

**Serving.** vLLM as a plain Kubernetes Deployment behind a Service, with a Gateway API route. Not KServe. KServe buys autoscaling sophistication the handoff team has to learn, and you have a fixed two-node floor anyway.

Weights pull from S3 via an init container onto a node-local volume, cached across pod restarts. Baking 30GB of weights into a container image makes every deploy a 30GB pull.

**Data.**
- Aurora PostgreSQL Serverless v2, pgvector extension, private subnets, encrypted with a customer-managed KMS key
- S3 buckets: `raw` (SharePoint mirror, versioned), `parsed` (extracted text and page images), `output` (generated docx). All SSE-KMS, all versioned, lifecycle to Infrequent Access at 90 days.
- Secrets Manager for the Entra client certificate and database credentials. Certificate auth to Entra, not a client secret. Secrets expire and rotate badly.

**Observability.** DCGM exporter for GPU utilization, memory, and temperature into Prometheus. Grafana dashboards for tokens per second, queue depth, time to first token, and cache hit rate. CloudWatch for application logs. Alert on GPU memory above 90%, queue depth sustained above 10, and any pod restart.

**GPU quota. Start this week.** g6e capacity in your target region needs a service quota increase and AWS has been slow to approve them. This request blocks the entire build and it costs nothing to file now.

## Fine-tuning and evaluation

Phase 4, not phase 1. Ship retrieval-based generation first and measure it. Then find out whether the adapter beats it.

**Training data.** Pairs of (RFP requirement, winning response section), extracted by running the shred pipeline over your old solicitations and mapping the requirements to the sections that answered them. This is a byproduct of the ingestion work, which is why fine-tuning comes late rather than early.

**Configuration.** LoRA rank 32, alpha 64, targeting attention and MLP projections on Qwen3.8-27B. On a 28B dense model this fits on two H100s, or one 80GB card with QLoRA. Hours, not days. Considerably cheaper than the 70B path.

**Held out: 50 bid packages the adapter never sees, including losses.** Losses in the evaluation set matter. A model trained only on wins learns your house style. A model evaluated against losses tells you whether the style is doing any work.

**The evaluation harness, all four measures:**

1. *Compliance recall.* Percentage of Section L requirements the shred catches, measured against a human-built matrix for the same solicitation. Target above 98%. A missed submission requirement is a non-responsive bid.
2. *Citation accuracy.* Percentage of generated factual claims that trace to a real source chunk containing that fact. Sampled and human-checked. Anything under 95% is not shippable.
3. *Blind preference.* A capture manager reads base output and adapter output on held-out RFPs without knowing which is which, and picks.
4. *Numeric accuracy.* Every figure in a generated cost narrative matched exactly against the deterministic calculator. Tolerance is zero.

**Ship the adapter only if it wins the blind test.** If plain retrieval beats it, that is a real result and it saves you the training pipeline, the adapter versioning, and the retraining cadence. Take the win.

## Phasing

Phase 1 is first-draft generation, and drafting cannot ship without retrieval and shred underneath it. There's no version of this where drafting comes first. What follows puts drafting in Phase 1 by building the substrate it stands on.

**Phase 1, weeks 1 to 14. Draft generation with citations.**
- Entra app registration, Graph connector, delta sync
- Document parsing including vision pass on scanned pages
- Metadata and outcome labeling (parallel human track, starts week 1)
- Structure-aware chunking, hybrid retrieval, reranking
- RFP shred producing the requirements matrix
- Section drafting with mandatory citations
- docx renderer against your template
- Word add-in v1: search, generate, cite
- EKS, vLLM, Aurora, Terraform, all of it

**Phase 2, weeks 15 to 22. Compliance and round-trip.**
- Requirements-to-response cross-check with a gap report
- Deterministic format and page-limit validation
- Tracked-changes insertion
- Inbound docx re-ingestion and diffing
- Pink team scoring against Section M
- Page-image retrieval for graphics and tables

**Phase 3, weeks 23 to 30. Pricing.**
- Rate card ingestion and versioning
- Staffing plan proposal from the SOW
- Deterministic cost calculator with escalation and wrap
- BOE narrative generation
- Rate validation blocking export on mismatch

**Phase 4, weeks 31 to 38. Training and handoff.**
- Training pair extraction from the labeled corpus
- LoRA training run and evaluation harness
- Blind preference test, ship-or-drop decision
- Operations runbook, on-call playbook, handoff sessions

## Cost

Monthly, steady state, us-east-1. Verify current rates before you build a budget on these.

| Item | Estimate |
|---|---|
| 2x g6e.12xlarge, on-demand, 24/7 | $15,000 to $15,500 |
| Same, 1-year Compute Savings Plan | $9,000 to $9,500 |
| Dev GPU (scaled to zero off-hours) | $1,200 to $1,800 |
| Aurora Serverless v2 | $300 to $800 |
| S3, KMS, data transfer | $150 to $400 |
| App tier, ALB, observability | $300 to $600 |
| **Total with Savings Plan** | **$11,000 to $13,000** |

Training runs add a few hundred dollars each, occasional rather than recurring.

Buy the Savings Plan after Phase 1, once real utilization is known. Committing before you have a load profile locks in the wrong instance family.

## Open items

These need answers or owners before the phase they gate.

| Item | Gates | Owner |
|---|---|---|
| g6e service quota increase | All of Phase 1 | File this week |
| Outcome labeling: who, and starting when | Retrieval quality, everything downstream | Needs a named person |
| M365 admin consent for the Office add-in | Phase 1 add-in delivery | Start week 1 |
| SharePoint site inventory and which libraries the crawler gets | Ingestion | M365 admin |
| Rate card owner and update cadence | Phase 3 | Contracts or finance |
| Your `.dotx` proposal templates | Phase 1 renderer | Proposal ops |
| Model provenance policy (deferred) | Nothing yet. Revisit if a customer questionnaire asks. | Contracts |
| Who receives the handoff | Phase 4 runbook scope | Leadership |

## Risks worth naming now

**Outcome labeling slips and retrieval quality never recovers.** This is the most likely failure and the least visible. Unlabeled corpus means the system cheerfully retrieves losing content. Assign a person, not a team.

**Word round-trip eats more schedule than planned.** Office.js has real constraints, tracked-changes manipulation through OOXML is finicky, and your templates almost certainly have quirks nobody has documented. Build the renderer against your actual `.dotx` in week 2, not week 10.

**Citation discipline erodes under deadline.** The first time a proposal manager is 12 hours from submission, somebody will want to turn off the uncited-sentence warning. Make it visible in the export rather than optional in the editor.

**Handoff to a team that wasn't in the room.** Every architecture decision here has a reason, and the reasons are what the operators need. The runbook is a Phase 4 deliverable, and the decision log starts now.
