# Legacy Runtime Retirement Plan

## Goal

The redesigned platform must not leave old containers, systemd application
processes, duplicated schedulers, or legacy execution code running beside the
new stack. Retirement is a controlled cutover with proof, not an immediate
deletion.

## Current Runtime Inventory

| Current component | Release 1 state | Final disposition |
|---|---|---|
| `web-api` container | Keep | Replace legacy dashboard/API handlers after UI parity |
| `token-service` container | Keep | Keep as sole Schwab token owner |
| `market-data` container | Keep | Evolve into quote and bar source for new engine |
| `scanner` container | Keep legacy execution | Replace with `scalp-engine` |
| `learner` container | Keep | Replace mixed model training with scalp outcome learning |
| `scheduler` container | Keep | Keep only time-owned jobs still required |
| `context-intel` container | Keep as context | Must never create brackets or bypass scalp gates |
| `watchdog` container | Keep temporarily | Re-evaluate after orchestration health policy is proven |
| legacy systemd app service | Audit only | Disable and remove after compose ownership is verified |
| legacy prediction/paper path | Production path | Read-only shadow, then remove after execution cutover |

## Clean Deployment Contract

Every production deployment must:

1. Build one immutable image tag for the commit.
2. Validate the compose file before stopping anything.
3. Run database migrations exactly once.
4. Start with `docker compose up -d --remove-orphans`.
5. Wait for all required health checks.
6. Verify the running image digest and commit for every service.
7. Verify exactly one owner for token refresh, market data, execution, learning,
   EOD scheduling, and WebSocket publication.
8. Disable the old systemd application only after the compose stack is healthy.
9. Fail deployment if an application container from the old manifest remains.
10. Preserve durable models, tokens, logs, and database facts during cleanup.

## Ownership Invariants

The deployment validator must prove:

- One Schwab token refresh owner: `token-service`.
- One market data publisher: `market-data`.
- One scalp plan publisher: `scalp-engine` after cutover.
- One execution owner per account/mode.
- One EOD close owner.
- One public web listener on port 8000.
- No monolith service running scanner, learner, or market data flags in-process.

## Cutover Sequence

### Stage A: Shadow

- New engine publishes plans to separate versioned keys/tables.
- Legacy scanner remains the only execution publisher.
- Compare coverage, freshness, direction, brackets, and blockers.

### Stage B: Paper cutover

- Pause legacy paper entry creation.
- Enable paper orders from valid scalp plans.
- Keep legacy output read-only for comparison.
- Roll back by switching one execution-owner feature flag.

### Stage C: Learning and UI cutover

- New outcome learner becomes the only adaptive gate writer.
- Command center becomes default after browser/data parity validation.
- Legacy learner and dashboard remain read-only for one release.

### Stage D: Retirement

- Remove legacy scanner and learner compose services.
- Remove obsolete health checks, pub/sub channels, endpoints, tables only after
  retention/export requirements are met.
- Remove the systemd monolith unit and legacy deployment branch.
- Run `docker compose up -d --remove-orphans` and assert the final manifest.

## No-Delete Safety Rules

- Never delete token files, model artifacts, logs, or database tables as part of
  an ordinary deployment.
- Rename/deprecate persistent schemas before a later archival migration.
- Stop and observe an old service for one release before deleting its code.
- Every removal requires a rollback command and a verified backup.

## Retirement Acceptance Criteria

- No duplicate owner publishes the same domain event.
- No old application container or systemd worker remains active.
- No orphan container is present after deploy.
- All required health checks remain green for a complete market session.
- One-second dashboard updates and execution SLAs are unchanged or improved.
- Rollback restores the previous image without restoring duplicate services.

