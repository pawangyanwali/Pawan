# The Self-Hosted Endpoint

Everything except two jobs runs on Bedrock and needs no hardware. This document covers the two that do: the 262K-token RFP shred and vision parsing of scanned pages.

## The machine

**`ml.g6e.xlarge`**: one NVIDIA L40S at 48GB, 4 vCPU, 32GB host RAM.

Roughly $2.20 to $2.50/hour on SageMaker, which is about 20% above the equivalent EC2 rate. Verify the current figure. The premium buys away the entire operational layer: no cluster, no Karpenter, no KEDA, no AMI to bake, no node to patch, no Kubernetes upgrade cycle for whoever inherits this.

The endpoint runs at `MinCapacity: 0`. It costs nothing when nothing is queued.

## Does Qwen3.8-27B fit in 48GB?

Weights at FP8: about 30GB. That leaves roughly 15GB for KV cache and activations.

The shred runs batch size 1 at 262,144 tokens, which is the memory-hungriest thing this system does. Estimating the cache: 64 layers, of which only 16 use full attention (the other 48 are gated DeltaNet linear attention with fixed-size state). With grouped-query attention at 8 KV heads and 128 head dimension, FP8 KV cache costs about 2KB per token per full-attention layer, so roughly 32KB per token across the 16 full-attention layers. At 262K tokens that's **about 8.5GB**.

Fits, with room. But two things about that number:

**Validate it against the real config before you commit to the instance.** The head counts above are inferred, not read from the model card. Load the model, run one 262K-token request, and read `nvidia-smi`. This takes twenty minutes and it decides your instance type.

**FP8 KV cache is required, not optional.** At bf16 the same cache is roughly 17GB and it will not fit. Set `--kv-cache-dtype fp8` explicitly.

If it turns out not to fit, the options in order of preference:

1. Drop `max_model_len` to 180K. Still holds a 450-page solicitation, and RFPs that large are rare. Costs nothing.
2. Move to `ml.g6e.12xlarge` (4x L40S) with tensor parallel 4. Roughly four times the hourly rate, and at 10 hours a month that's $90 instead of $22. Still inside budget.

There is no single-GPU SageMaker instance between 48GB and the 4-GPU jump, so those really are the two choices.

## The container

SageMaker bring-your-own-container with vLLM. Two paths:

**AWS LMI container.** The Deep Learning Container for large model inference bundles vLLM and implements the `/invocations` and `/ping` contract SageMaker expects. Less to build, and you're on AWS's upgrade cadence for the vLLM version.

**Your own image from `vllm/vllm-openai`.** Roughly 60 lines of FastAPI wrapping vLLM's OpenAI-compatible server to satisfy the SageMaker contract. More control over the vLLM version, which matters here because Qwen3.8's hybrid attention needs a recent build.

Take the second. Qwen3.8 shipped in August 2026 and gated DeltaNet support is new enough that pinning your own vLLM version is worth the 60 lines. `services/inference/` holds this image.

## Weights and cold start

SageMaker pulls model artifacts from S3 into the container at startup. 30GB of FP8 weights is the bulk of the cold start.

Honest numbers for scale-from-zero on a 30GB model:

| Stage | Time |
|---|---|
| Instance provision | 60 to 120s |
| Container image pull | 30 to 60s |
| Weight download from S3 | 90 to 180s |
| vLLM engine init and warmup | 60 to 120s |
| **Total** | **4 to 8 minutes** |

That is above the 2 to 5 minute figure I gave earlier, and it does not matter, because of what the endpoint is used for.

**The shred is a background job.** A writer uploads a solicitation and the shred kicks off immediately, before anyone needs the output. By the time a human opens the compliance matrix, it's been ready for a while. Nobody watches a shred.

**Drafting never touches this endpoint.** Drafting runs on Bedrock, which has no cold start and no capacity to provision. That is the path that has to work at 2 a.m. before a submission, and it's the reason the split exists.

The one case where the cold start is visible is the Bedrock failover path: if Bedrock is throttling and drafting reroutes here, the first section takes 8 minutes. That is a bad day, not an outage, and it beats the alternative of having no second path at all.

## Autoscaling

```
MinCapacity: 0
MaxCapacity: 2
Metric:      ApproximateBacklogSizePerInstance
Target:      1
ScaleInCooldown:  600s
ScaleOutCooldown: 60s
```

Ten minutes of scale-in cooldown is deliberate. Parsing a backlog of scanned pages arrives in bursts, and paying for ten idle minutes beats paying an 8-minute cold start twice.

`MaxCapacity: 2` covers two pursuits shredding at once. Raise it if that ever becomes three.

## Monthly usage

| Job | Frequency | Instance-hours |
|---|---|---|
| RFP shred | ~10/month, ~20 min each including cold start | 3.5 |
| Vision parsing, new scanned documents | Nightly delta | 2 |
| Bedrock failover drafting | Rare | 1 |
| Headroom | | 3.5 |
| **Total** | | **~10 hrs, ~$22/month** |

## The one-time backfill

2,000 bid packages, of which maybe 30% are scanned rather than text-layer PDFs. Call it 600 documents at 40 pages, so 24,000 pages of vision parsing at roughly 3 seconds a page on an L40S. That's about 20 instance-hours.

Run it with `MaxCapacity: 4` over a weekend: 5 wall-clock hours, 20 instance-hours, roughly **$45**.

Embedding the resulting 500K chunks runs on Lambda with the ONNX model on CPU, not on this endpoint. Parallel Lambda invocations do it in under an hour for a few dollars.

Total one-time cost to load the corpus: **$100 to $200**, dominated by the vision pass and the initial DynamoDB writes.

Make it idempotent per document. You will run it more than once.

## What would change the design

| Signal | Threshold | Action |
|---|---|---|
| Qwen3.8-27B appears as a managed Bedrock model | Any release | Delete the endpoint. The gateway table changes by one line. |
| Bedrock Custom Model Import adds Qwen3 architectures | Any release | Import Qwen3.8-27B, delete the endpoint, keep the 262K window. |
| Shred jobs per month | Above 60 | Endpoint hours pass 60/month. Evaluate a warm instance against the cold-start pain. |
| KV cache measurement | Above 15GB at 262K | Drop to 180K context, or move to `ml.g6e.12xlarge`. |

The first two are the ones to watch. Both delete hardware from this system, and both are plausible within a year.
