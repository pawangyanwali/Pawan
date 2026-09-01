# RFI/RFP Response System

Proprietary proposal response system built on open-weight LLMs, hosted in AWS.

Reads our SharePoint bid repository, answers questions about past wins, losses,
and no-bids, shreds incoming solicitations into a requirements matrix, drafts
proposal sections with citations back to source documents, and supports pricing
against our labor rate cards.

## Documents

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - system design, model stack, AWS
  layout, phasing, cost, and open items

## Status

Design, pre-build. Nothing implemented yet.

## Model stack

| Role | Model | License |
|---|---|---|
| Generation, shred, extraction | Qwen3.8-27B | Apache 2.0 |
| Embeddings | Qwen3-Embedding-8B | Apache 2.0 |
| Reranking | Qwen3-Reranker-4B | Apache 2.0 |

Served with vLLM on EKS, g6e.12xlarge nodes.

## Blocking items

1. File the g6e service quota increase with AWS. This blocks every other task.
2. Assign an owner for win/loss/no-bid labeling of the existing bid corpus.
3. Open the M365 admin consent conversation for the Office add-in.
