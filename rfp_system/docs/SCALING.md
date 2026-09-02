# Scaling and Cost

Ceiling: $1,000/month. Design lands at roughly $90.

## What we are sizing for

| Input | Value |
|---|---|
| Writers | 2 to 3 |
| Peak concurrent users | 3 |
| Pursuits per month | ~10 |
| Drafted sections per pursuit | ~60 |
| Acceptable wait for a drafted section | 2 to 5 minutes |
| Q&A pattern | Bursty, during active pursuits only |

Derived monthly volume:

| Workload | Tokens |
|---|---|
| Section drafting output | ~1.8M |
| Section drafting input (requirement + 8 chunks) | ~7M |
| RFP shred input (10 solicitations at 262K) | ~3M |
| Q&A (300 queries) | ~3M in, 0.2M out |
| **Total** | **~13M in, ~2M out** |

## Cost model

us-east-1, monthly, steady state. Verify current rates before budgeting.

| Item | Basis | Cost |
|---|---|---|
| Bedrock Qwen3-32B | ~10M in, ~2M out (drafting, Q&A, classification) | $5 |
| SageMaker Async, ml.g6e.xlarge, MinCapacity 0 | ~10 hrs (10 shreds + vision parsing) | $25 |
| Lambda | ~50K invocations, mostly short | $5 |
| Lambda provisioned concurrency (retrieval, business hours) | Kills the Q&A cold start | $12 |
| Step Functions | A few thousand state transitions | $2 |
| API Gateway HTTP API | Add-in and web traffic | $1 |
| SQS | Five queues, low volume | $1 |
| DynamoDB on-demand | 500K items written once, light reads after | $8 |
| S3 Vectors | 500K vectors, ~300 queries | $5 |
| S3 | 250GB versioned, SSE-KMS | $8 |
| CloudWatch logs, metrics, alarms | | $15 |
| Secrets Manager, KMS, ECR | | $10 |
| **Total** | | **~$97** |

One-time backfill of 2,000 bid packages: $150 to $350, mostly vision parsing of scanned pages on the async endpoint plus the initial DynamoDB writes.

### Where the money is not going

**No EKS control plane.** $73/month for a cluster that would have run eight pods.

**No Aurora.** The 0.5 ACU floor was $88/month before storage. S3 Vectors plus DynamoDB costs $13 and neither has an instance that can fail.

**No NAT gateway.** $32/month plus $0.045/GB. Nothing in this design sits in a VPC. Lambda, DynamoDB, S3, SQS, Bedrock, and SageMaker are all reached over IAM-authenticated public endpoints. That also removes subnets, route tables, security groups, and VPC endpoints from the handoff.

**No idle GPU.** Real demand is 5 to 6 GPU-hours a month. The first version of this design provisioned 5,840.

## Scale-to-zero behavior

| Component | Idle cost | Cold start | Scales to |
|---|---|---|---|
| `api`, workers, `render`, `compute` | $0 | 1 to 3s (container image) | 1000 concurrent |
| `retrieval` | $12/mo with provisioned concurrency, else $0 | 12s cold, 1 to 2s warm | 1000 concurrent |
| Bedrock | $0 | None | AWS account quota |
| SageMaker Async endpoint | $0 (MinCapacity 0) | 2 to 5 min | MaxCapacity 2 |
| S3 Vectors, DynamoDB, S3, SQS | Storage only | None | Far past this workload |

The only cold start a human waits on is the SageMaker endpoint, and it only fires on an RFP shred, which happens once at the start of a pursuit. Nobody is waiting on it at 2 a.m. before a submission.

Drafting runs on Bedrock precisely because it has no cold start and no capacity to provision. That is the path that has to work during crunch week.

### The embedding Lambda cold start

Qwen3-Embedding-0.6B as ONNX in a Lambda container image at 3,008MB takes 10 to 15 seconds to load cold, 1 to 2 seconds warm. At 300 queries a month that's a cold start on most queries.

$12/month of provisioned concurrency during business hours removes it. Cheap enough to just do.

## Growth thresholds

The design holds until one of these trips. Instrument all four from day one.

| Signal | Threshold | What changes |
|---|---|---|
| Bedrock spend | Above $300/month | Self-hosting starts to compete. Model both before switching. |
| SageMaker async hours | Above 150/month | Consider a warm instance or a Savings Plan on the endpoint. |
| Chunks in S3 Vectors | Above 50M | Well within the 2B index limit, but query cost becomes the line item to watch. |
| Writers | Above 10 | Retrieval provisioned concurrency needs raising. Nothing structural. |
| "I know the phrase and can't find it" complaints | More than occasional | The missing BM25 leg is hurting. Price OpenSearch Serverless against the pain. |

## What growth does not require

Adding writers, pursuits, or corpus does not require re-architecting. Lambda concurrency, DynamoDB on-demand, S3 Vectors, and Bedrock all scale on request without a capacity decision. The only component with a ceiling anyone has to set is the SageMaker endpoint's MaxCapacity, and it is one Terraform value.

That is what the serverless choice bought. The cost of getting the sizing wrong is a slightly larger bill, not an outage and not a migration.
