# RFI/RFP Response System

Proprietary proposal response system on open-weight LLMs, hosted in AWS.

Reads the SharePoint bid repository, answers questions about past wins, losses,
and no-bids, shreds incoming solicitations into a requirements matrix, drafts
proposal sections with citations back to source documents, and prices against
labor rate cards.

## Status

Design and scaffold. Nothing deployed.

## Documents

| Document | Covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Service decomposition, data layer, SharePoint integration, generation, compliance, pricing, Word round-trip, phasing |
| [docs/SCALING.md](docs/SCALING.md) | Cost model, scale-to-zero behavior, growth thresholds |
| [docs/RELIABILITY.md](docs/RELIABILITY.md) | Failure modes, canary deploys without a staging environment, degraded modes |
| [docs/INFERENCE.md](docs/INFERENCE.md) | The GPU machine: instance, memory math, container, cold start, backfill |

## Shape

Serverless. No Kubernetes, no relational database, no VPC.

```
SharePoint Online ──Graph delta──> Lambda workers ──> S3 (raw, parsed, output)
                                          │
                                          ├──> DynamoDB   chunks, matrices,
                                          │               rate cards, versions
                                          └──> S3 Vectors embeddings

Word add-in ──Entra SSO──> API Gateway ──> Lambda ──> Inference gateway
                                                          │
                                             ┌────────────┴────────────┐
                                        Bedrock                  SageMaker Async
                                       Qwen3-32B                  Qwen3.8-27B
                                    drafting, Q&A,              262K shred,
                                     classification            vision parsing
                                    no cold start              MinCapacity 0
```

Two inference backends that share no capacity and no scaling behavior. Losing
one makes the system slower, not stopped.

## Models

| Role | Model | Where | License |
|---|---|---|---|
| Drafting, Q&A, classification | Qwen3-32B | Bedrock, managed | Apache 2.0 |
| RFP shred (262K), vision parsing | Qwen3.8-27B | SageMaker Async, `ml.g6e.xlarge` | Apache 2.0 |
| Embeddings | Qwen3-Embedding-0.6B (ONNX) | Lambda, CPU | Apache 2.0 |
| Reranking | Qwen3-Reranker-0.6B (ONNX) | Lambda, CPU | Apache 2.0 |

Qwen3.8-27B is self-hosted because Bedrock does not offer it and Custom Model
Import does not support the architecture. The two properties it is hosted for
are the 262K context window and the built-in vision encoder. See
[docs/INFERENCE.md](docs/INFERENCE.md).

## Cost

Roughly **$90/month** at 2 to 3 writers and 10 pursuits. One-time corpus
backfill is $100 to $200. Breakdown in [docs/SCALING.md](docs/SCALING.md).

## Layout

```
services/
  common/       rfp_common: contracts, config, inference gateway, stores
  api/          BFF for the web app and Word add-in
  retrieval/    vector query, rerank, ACL trim
  worker_*/     ingest, parse, index, draft
  render/       section JSON to .docx against the customer template
  compute/      deterministic engines: page counts, cost calculator
  inference/    vLLM container for the SageMaker endpoint
infra/terraform/
docker/         shared Lambda base image
local/          LocalStack plus a mocked inference gateway
```

`services/common/rfp_common/contracts.py` is the module boundary. A service may
change anything about how it works as long as it still accepts and produces
those shapes.

## Running it

```
make build            # every service image
make test             # contract tests
make local            # LocalStack plus mocked inference
make plan             # terraform plan against production
make deploy           # push, apply, canary at 10% for 5 min, smoke test
make eval             # held-out evaluation, publishes to CloudWatch
```

There is no staging environment. `make deploy` canaries and rolls itself back
on an error-rate alarm. Quality regressions do not throw errors, which is why
`make eval` runs nightly. See [docs/RELIABILITY.md](docs/RELIABILITY.md).

## Blocking items

1. **Assign an owner for win/loss/no-bid labeling of the existing corpus.**
   Two to three weeks of one person's time, and it gates retrieval quality for
   everything downstream. Unlabeled, the system will confidently surface content
   from proposals you lost.
2. **Open the M365 admin consent conversation for the Office add-in.** Two days
   if you start now, three weeks if you raise it in month four.
3. **Enable Bedrock model access for Qwen3-32B** in the target account and
   region.
4. **Get the `.dotx` proposal templates** to the renderer team in week two, not
   week ten.
