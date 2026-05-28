#!/usr/bin/env python3
"""
NASDAQ Agent — JSON file state → PostgreSQL migration
======================================================

Migrates accumulated ML learning state from the old flat-file persistence
(algo_params.json, loss_patterns.json, win_patterns.json, algo_selector.json,
model_registry.json, adaptive_filter.json) into PostgreSQL.

Run this ONCE on the server before restarting services with the new code.
It is safe to re-run: each step checks whether the destination already has
data and skips if so (--force overrides).

USAGE
-----
  # Dry run — shows what would be migrated, touches nothing:
  python3 migrate_json_to_postgres.py --dry-run

  # Real run (reads from /app/data by default):
  python3 migrate_json_to_postgres.py

  # Override data directory:
  python3 migrate_json_to_postgres.py --data-dir /opt/nasdaq-agent/nasdaq_agent/data

  # Force overwrite even if destination already has data:
  python3 migrate_json_to_postgres.py --force
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate_json")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        log.warning("  [SKIP] %s — file not found", path.name)
        return None
    try:
        data = json.loads(path.read_text())
        log.info("  [READ] %s  (%d bytes)", path.name, path.stat().st_size)
        return data
    except Exception as exc:
        log.error("  [ERR]  %s — %s", path.name, exc)
        return None


def _pg_connect():
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ.get("PGDATABASE", "nasdaq_agent"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ["PGPASSWORD"],
    )


def _ensure_tables(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS algo_params (
            family              TEXT NOT NULL,
            param               TEXT NOT NULL,
            current_val         DOUBLE PRECISION NOT NULL,
            previous_val        DOUBLE PRECISION NOT NULL,
            rollback_val        DOUBLE PRECISION NOT NULL,
            last_updated_cycle  INTEGER NOT NULL DEFAULT 0,
            last_reason         TEXT NOT NULL DEFAULT '',
            updated_at          TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (family, param)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS param_tune_log (
            id       SERIAL PRIMARY KEY,
            family   TEXT NOT NULL,
            param    TEXT NOT NULL,
            old_val  DOUBLE PRECISION NOT NULL,
            new_val  DOUBLE PRECISION NOT NULL,
            reason   TEXT NOT NULL,
            source   TEXT NOT NULL DEFAULT 'auto',
            tuned_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS system_kv (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def _kv_has_data(cur, key: str) -> bool:
    cur.execute("SELECT 1 FROM system_kv WHERE key = %s", (key,))
    return cur.fetchone() is not None


def _kv_upsert(cur, key: str, data, dry_run: bool) -> None:
    payload = json.dumps(data)
    log.info("    → upsert system_kv key=%s  (%d bytes)", key, len(payload))
    if not dry_run:
        cur.execute("""
            INSERT INTO system_kv (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
        """, (key, payload))


# ── Migration steps ────────────────────────────────────────────────────────────

def migrate_algo_params(data_dir: Path, cur, dry_run: bool, force: bool) -> int:
    """algo_params.json → algo_params table"""
    log.info("─── algo_params.json → algo_params table")
    raw = _load_json(data_dir / "algo_params.json")
    if raw is None:
        return 0

    # Check destination
    cur.execute("SELECT COUNT(*) FROM algo_params")
    existing = cur.fetchone()[0]
    if existing > 0 and not force:
        log.info("  [SKIP] algo_params table already has %d rows (use --force to overwrite)", existing)
        return 0

    rows_written = 0
    for family, params in raw.items():
        for param, vals in params.items():
            row = (
                family, param,
                float(vals.get("current",  vals.get("current_val",  0))),
                float(vals.get("previous", vals.get("previous_val", 0))),
                float(vals.get("rollback", vals.get("rollback_val", 0))),
                int(vals.get("last_updated_cycle", 0)),
                str(vals.get("last_reason", "")),
            )
            log.info(
                "    %s.%s: current=%.4f previous=%.4f %s",
                family, param, row[2], row[3],
                "(tuned)" if abs(row[2] - row[3]) > 1e-4 else "(default)",
            )
            if not dry_run:
                cur.execute("""
                    INSERT INTO algo_params
                        (family, param, current_val, previous_val, rollback_val,
                         last_updated_cycle, last_reason, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (family, param) DO UPDATE SET
                        current_val        = EXCLUDED.current_val,
                        previous_val       = EXCLUDED.previous_val,
                        rollback_val       = EXCLUDED.rollback_val,
                        last_updated_cycle = EXCLUDED.last_updated_cycle,
                        last_reason        = EXCLUDED.last_reason,
                        updated_at         = NOW()
                """, row)
            rows_written += 1

    log.info("  [OK] %d param rows migrated", rows_written)
    return rows_written


def migrate_kv_file(data_dir: Path, filename: str, kv_key: str,
                    cur, dry_run: bool, force: bool) -> bool:
    """Generic JSON file → system_kv migration."""
    log.info("─── %s → system_kv[%s]", filename, kv_key)
    raw = _load_json(data_dir / filename)
    if raw is None:
        return False

    if _kv_has_data(cur, kv_key) and not force:
        log.info("  [SKIP] system_kv key '%s' already exists (use --force to overwrite)", kv_key)
        return False

    _kv_upsert(cur, kv_key, raw, dry_run)
    log.info("  [OK] migrated")
    return True


def migrate_adaptive_filter(data_dir: Path, cur, dry_run: bool, force: bool) -> bool:
    """adaptive_filter.json → system_kv[adaptive_filter_state]"""
    log.info("─── adaptive_filter.json → system_kv[adaptive_filter_state]")
    raw = _load_json(data_dir / "adaptive_filter.json")
    if raw is None:
        return False

    kv_key = "adaptive_filter_state"
    if _kv_has_data(cur, kv_key) and not force:
        log.info("  [SKIP] system_kv key '%s' already exists (use --force to overwrite)", kv_key)
        log.info("  NOTE: adaptive_filter was already dual-writing to system_kv; DB copy may be newer than file")
        return False

    # Summarise what's being migrated
    blocked = len(raw.get("blocked_contexts", {}))
    wr = raw.get("current_win_rate", 0)
    thr = raw.get("dynamic_threshold", 0)
    log.info("  Filter state: threshold=%.1f%%  WR=%.1f%%  blocked_contexts=%d", thr, wr*100, blocked)
    _kv_upsert(cur, kv_key, raw, dry_run)
    log.info("  [OK] migrated")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate JSON learning state → PostgreSQL")
    parser.add_argument("--data-dir", default="/app/data",
                        help="Path to data directory on the host (default: /app/data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be migrated without writing anything")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite destination even if it already has data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dry_run  = args.dry_run
    force    = args.force

    log.info("=== JSON → PostgreSQL state migration  %s===",
             "[DRY RUN] " if dry_run else "")
    log.info("Data directory: %s", data_dir)

    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # List all JSON files present
    json_files = sorted(data_dir.glob("*.json"))
    log.info("JSON files found: %s", [f.name for f in json_files] or "none")

    try:
        pg = _pg_connect()
        log.info("Connected to PostgreSQL at %s", os.environ["PGHOST"])
    except Exception as exc:
        log.error("Cannot connect to PostgreSQL: %s", exc)
        log.error("Ensure PGHOST, PGPASSWORD (and optionally PGPORT, PGDATABASE, PGUSER) are set")
        sys.exit(1)

    results = {}
    try:
        with pg:
            with pg.cursor() as cur:
                _ensure_tables(cur)

                results["algo_params"]    = migrate_algo_params(data_dir, cur, dry_run, force)
                results["loss_analyzer"]  = migrate_kv_file(
                    data_dir, "loss_patterns.json", "ale_loss_analyzer", cur, dry_run, force)
                results["win_reinforcer"] = migrate_kv_file(
                    data_dir, "win_patterns.json", "ale_win_reinforcer", cur, dry_run, force)
                results["algo_selector"]  = migrate_kv_file(
                    data_dir, "algo_selector.json", "ale_algo_selector", cur, dry_run, force)
                results["model_registry"] = migrate_kv_file(
                    data_dir, "model_registry.json", "ale_model_registry", cur, dry_run, force)
                results["adaptive_filter"]= migrate_adaptive_filter(data_dir, cur, dry_run, force)

                if not dry_run:
                    pg.commit()

    except Exception as exc:
        log.error("Migration failed: %s", exc)
        pg.rollback()
        sys.exit(1)
    finally:
        pg.close()

    log.info("")
    log.info("=== Summary %s===", "[DRY RUN — nothing written] " if dry_run else "")
    for name, result in results.items():
        status = "SKIPPED" if not result else ("OK" if result else "OK")
        log.info("  %-20s %s", name, status)

    if dry_run:
        log.info("")
        log.info("Re-run without --dry-run to apply.")
    else:
        log.info("")
        log.info("Migration complete. You can now safely delete the JSON files in %s", data_dir)
        log.info("  rm %s/*.json", data_dir)
        log.info("")
        log.info("If the SQLite .db files were not already migrated via migrate_sqlite_to_postgres.py,")
        log.info("run that script before deleting them too.")


if __name__ == "__main__":
    main()
