#!/usr/bin/env python3
"""
BiLSTM training diagnostic — verifies the full persistence chain.

Run INSIDE the learner container (or any container with PG + Valkey + disk access):
    docker compose exec learner python scripts/diagnose_bilstm.py

Or on the host if env vars are set:
    python scripts/diagnose_bilstm.py

Checks, in order:
  1. Disk:        cluster .pt model files exist + their mtimes (the trained artifact)
  2. PostgreSQL:  deep:history durable row + learner:status
  3. Valkey:      deep:history live mirror + learner:status + pending train request
  4. In-process:  deep_model._cluster_trained + _training_history
"""
import os
import sys
import time
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; DIM = "\033[2m"; RST = "\033[0m"
def ok(m):   print(f"{GREEN}✓{RST} {m}")
def bad(m):  print(f"{RED}✗{RST} {m}")
def warn(m): print(f"{YEL}!{RST} {m}")
def hdr(m):  print(f"\n{'='*70}\n{m}\n{'='*70}")
def age(ts):
    if not ts: return "never"
    s = time.time() - float(ts)
    return f"{s:.0f}s ago" if s < 3600 else f"{s/3600:.1f}h ago"


def check_disk():
    hdr("1. DISK — cluster model files (the durable trained artifact)")
    try:
        from agent.deep_model import _CLUSTER_CONFIGS
    except Exception as e:
        bad(f"cannot import deep_model: {e}"); return
    any_trained = False
    for cluster, cfg in _CLUSTER_CONFIGS.items():
        p = str(cfg.get("path", ""))
        if p and os.path.exists(p):
            mt = os.path.getmtime(p)
            sz = os.path.getsize(p) / 1024
            ok(f"Cluster {cluster}: {p} ({sz:.0f} KB, trained {age(mt)} — "
               f"{datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M:%S')})")
            any_trained = True
        else:
            bad(f"Cluster {cluster}: {p or '(no path)'} — MISSING")
    if not any_trained:
        bad("No cluster model files on disk — training has never completed successfully")


def check_postgres():
    hdr("2. POSTGRESQL — durable source of truth")
    try:
        from agent.service_state import get_state
    except Exception as e:
        bad(f"cannot import service_state: {e}"); return

    row = get_state("deep:history", ignore_expiry=True)
    if row and row.get("history"):
        h = row["history"]
        ok(f"deep:history present — {len(h)} epoch points, updated {age(row.get('updated_at'))}")
        last = h[-1]
        print(f"   {DIM}last epoch: cluster={last.get('cluster')} "
              f"epoch={last.get('epoch')}/{last.get('total_epochs')} "
              f"loss={last.get('loss')} mode={last.get('mode')}{RST}")
    else:
        bad("deep:history MISSING in PostgreSQL — loss curve has no durable record")

    ls = get_state("learner:status", ignore_expiry=True)
    if ls:
        deep = ls.get("deep", {})
        ok(f"learner:status present — published {age(ls.get('ts'))}")
        print(f"   {DIM}deep.running={deep.get('running')} "
              f"trained={deep.get('trained')} "
              f"trained_clusters={deep.get('trained_clusters')} "
              f"last_error={deep.get('last_error')!r} "
              f"last_tickers={deep.get('last_tickers')}{RST}")
    else:
        bad("learner:status MISSING in PostgreSQL")


def check_valkey():
    hdr("3. VALKEY — live cache + pub/sub mirror")
    try:
        from agent.valkey_client import _get_client
        c = _get_client()
    except Exception as e:
        bad(f"cannot get Valkey client: {e}"); return
    if not c:
        bad("Valkey client is None — connection unavailable"); return
    ok("Valkey connection OK")

    raw = c.get("deep:history")
    if raw:
        h = json.loads(raw).get("history", [])
        ok(f"deep:history mirror present — {len(h)} points")
    else:
        warn("deep:history not in Valkey (PG is still source of truth)")

    raw = c.get("learner:status")
    if raw:
        deep = json.loads(raw).get("deep", {})
        ok(f"learner:status mirror present — deep.running={deep.get('running')} "
           f"history={len(deep.get('history', []))} pts")
    else:
        warn("learner:status not in Valkey")

    pending = c.get("deep:train:requested")
    if pending:
        warn(f"deep:train:requested is SET — a manual train request is queued/unconsumed "
             f"(TTL={c.ttl('deep:train:requested')}s)")
    else:
        ok("deep:train:requested clear (no pending request)")


def check_inprocess():
    hdr("4. IN-PROCESS — deep_model working copy (this process only)")
    try:
        from agent.deep_model import _cluster_trained, get_training_history
    except Exception as e:
        bad(f"cannot import deep_model: {e}"); return
    print(f"   _cluster_trained = {_cluster_trained}")
    h = get_training_history()
    print(f"   _training_history = {len(h)} points")
    note = ("(empty is EXPECTED in web-api — it never trains; "
            "should be populated in the learner after a run or restart)")
    if not h:
        warn(f"in-process history empty {note}")
    else:
        ok(f"in-process history has {len(h)} points")


if __name__ == "__main__":
    print(f"BiLSTM diagnostic — {datetime.now(timezone.utc).isoformat()}")
    print(f"Running in PID {os.getpid()} — hostname {os.uname().nodename}")
    check_disk()
    check_postgres()
    check_valkey()
    check_inprocess()
    print(f"\n{DIM}Tip: run this in BOTH the learner and web-api containers to compare:"
          f"\n  docker compose exec learner  python scripts/diagnose_bilstm.py"
          f"\n  docker compose exec web-api  python scripts/diagnose_bilstm.py{RST}")
