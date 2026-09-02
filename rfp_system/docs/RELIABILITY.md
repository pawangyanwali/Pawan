# Reliability

Requirement: the system cannot fail while writers are working a proposal.
Constraint: production only. No staging environment.

## The property that matters most

**A system failure must never block a submission.**

Every generated section lives in the writer's own `.docx` on their machine and in SharePoint, not on a server. If the entire stack is dark, the writer keeps working in Word and submits on time. They lose the assistant, not the proposal.

This is worth more than any amount of multi-AZ redundancy, and it costs nothing. Two rules protect it:

1. The Word add-in degrades to plain Word. Every add-in call has a timeout and a failure path that leaves the document intact and editable. No add-in feature ever holds a document open, locks it, or requires a server round-trip to save.
2. Nothing the writer has produced exists only in the system. Sections persist to their `.docx` as tracked changes at the moment they're accepted.

Test this deliberately. Point the add-in at a dead endpoint and confirm a writer can still work.

## Failure modes

| Failure | Blast radius | Mitigation | Recovery |
|---|---|---|---|
| Bedrock throttling or outage | Drafting and Q&A stop | Gateway fails over to the SageMaker endpoint | 2 to 5 min (endpoint cold start), automatic |
| SageMaker endpoint fails to scale | RFP shred stops | Gateway falls back to a chunked shred on Bedrock at reduced fidelity | Immediate, degraded |
| Lambda function error | One job type stops | SQS redrive to DLQ after 3 attempts, CloudWatch alarm | Fix and redrive the DLQ |
| Bad deploy | Whole surface | Canary at 10%, automatic rollback on error-rate alarm | Under 5 min, automatic |
| Microsoft Graph outage | No fresh sync, no ACL re-verify | Serve from the cached index with a staleness banner; block only content whose cached ACL is older than 24h | Automatic when Graph returns |
| DynamoDB or S3 unavailable in an AZ | None | Both are multi-AZ by design | None needed |
| Region outage | Everything | S3 cross-region replication of `raw`, DynamoDB PITR | Hours. Documented, not automated. |
| A writer's laptop dies | Their in-flight edits | Content is in SharePoint and in DynamoDB section versions | Reopen from SharePoint |

Two independent inference paths is the design point. Bedrock and the SageMaker endpoint share no capacity, no AZ constraint, and no scaling behavior. One being unavailable is a slower system, not a stopped one.

## Deploying to production without a staging environment

Prod-only is the choice. These four things carry the weight a staging environment would have.

**Canary with automatic rollback.** Every Lambda deploys behind a weighted alias. CodeDeploy shifts 10% of traffic to the new version, watches a CloudWatch alarm on error rate and p99 duration for 5 minutes, then shifts the rest or rolls back. A bad deploy affects one request in ten for five minutes and then undoes itself. No human has to be awake.

**A smoke suite that runs against production after every deploy.** Six checks, under 30 seconds: retrieve a known chunk, draft a known section and assert its citations resolve, render a `.docx` and assert the heading styles survive, shred a stored 3-page RFP and assert the requirement count, run a rate lookup and assert the arithmetic, verify Graph auth. Failure triggers the rollback.

**A local stack for logic changes.** docker-compose with LocalStack for DynamoDB, SQS, and S3, plus a mocked inference gateway that returns fixtures. Catches the most common class of bug (contract mismatch between services) on a laptop, for free. This is what the docker images are for beyond deployment.

**Contract tests in CI.** Every service validates its inputs and outputs against the pydantic models in `rfp_common.contracts`. A change that breaks a contract fails the build, not production.

### What prod-only still costs you

Prompt and model regressions do not throw errors. A change that makes drafts subtly worse passes every smoke check and every alarm. The only thing that catches it is the evaluation harness, which is why it runs on a held-out set on a schedule and not just before a fine-tuning decision.

Run the eval suite nightly against production. A drop in citation accuracy or compliance recall is the signal a staging environment would otherwise have given you.

## Alarms worth having

Alert on these. Ignore everything else.

| Alarm | Threshold | Why |
|---|---|---|
| DLQ depth | Above 0 | A job died. Someone's section never appeared. |
| Draft job age | Oldest message above 10 min | Writers are waiting past the SLA. |
| Citation accuracy (nightly eval) | Below 95% | The output is no longer trustworthy. |
| Compliance recall (nightly eval) | Below 98% | The shred is missing requirements. |
| Bedrock throttle rate | Above 1% | Approaching an account quota. |
| SageMaker endpoint scale failure | Any | Shred is degraded to the Bedrock fallback. |
| Graph auth failure | Any sustained | Certificate expiry, almost always. |

That last one deserves a calendar reminder as well as an alarm. Certificate expiry is the single most common cause of a Graph integration going dark, and it fails at the worst possible moment because nobody was watching a date.

## Deliberate degradation

When something is unavailable, the system should do less rather than nothing.

- Graph down: retrieval works from the cached index, sync pauses, banner shows staleness.
- SageMaker down: shred runs chunked on Bedrock, output flags reduced cross-reference confidence.
- Bedrock down: drafting queues to the SageMaker endpoint, writers see a longer wait, not an error.
- Retrieval down: drafting is blocked, and the add-in still opens, still edits, still saves.
- Everything down: Word works. That's the floor, and it's high enough.
