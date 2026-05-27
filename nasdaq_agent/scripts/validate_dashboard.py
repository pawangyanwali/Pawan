#!/usr/bin/env python3
"""
Dashboard data validator — run inside any container:
  docker compose exec scanner python3 /app/scripts/validate_dashboard.py
"""
import os, sys, json
sys.path.insert(0, '/app')

# Env vars already injected by docker-compose — no load_dotenv needed
from agent.db import get_conn

SEP = "-" * 60

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")

# ── Q3 + Q4: Today's trades ───────────────────────────────────
section("Q3+Q4  Today's closed trades (PostgreSQL, CURRENT_DATE)")
with get_conn() as c:
    r = c.execute("""
        SELECT
            COUNT(*)                                                AS total,
            SUM(CASE WHEN pnl_dollar > 0  THEN 1 ELSE 0 END)      AS wins,
            SUM(CASE WHEN pnl_dollar <= 0 THEN 1 ELSE 0 END)       AS losses,
            ROUND(
                SUM(CASE WHEN pnl_dollar > 0 THEN 1.0 ELSE 0 END)
                / NULLIF(COUNT(*), 0) * 100, 1
            )                                                       AS win_rate_pct,
            ROUND(COALESCE(SUM(pnl_dollar), 0)::numeric, 2)        AS today_pnl
        FROM paper_trades
        WHERE status = 'CLOSED'
          AND closed_at::timestamptz::date = CURRENT_DATE
    """).fetchone()
    print(dict(r))

# ── Last 5 calendar days ──────────────────────────────────────
section("Trade dates — last 5 calendar days")
with get_conn() as c:
    rows = c.execute("""
        SELECT
            closed_at::timestamptz::date                     AS d,
            COUNT(*)                                         AS total,
            SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) AS wins,
            ROUND(COALESCE(SUM(pnl_dollar),0)::numeric, 2)  AS pnl
        FROM paper_trades
        WHERE status = 'CLOSED'
        GROUP BY d
        ORDER BY d DESC
        LIMIT 5
    """).fetchall()
    for row in rows:
        print(dict(row))

# ── Last 5 individual trades ──────────────────────────────────
section("Last 5 closed trades (raw)")
with get_conn() as c:
    rows = c.execute("""
        SELECT ticker, direction, pnl_dollar, closed_at, exit_reason
        FROM paper_trades
        WHERE status = 'CLOSED'
        ORDER BY closed_at DESC
        LIMIT 5
    """).fetchall()
    for row in rows:
        print(dict(row))

# ── Q5: Adaptive filter ───────────────────────────────────────
section("Q5  Adaptive filter status")
try:
    from agent.adaptive_filter import get_status
    s = get_status()
    for k in ("current_win_rate", "target_win_rate", "is_learning",
              "dynamic_threshold", "suppressed_count"):
        print(f"  {k:22s}: {s.get(k)}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Q7: Live backtest ─────────────────────────────────────────
section("Q7  Live backtest win rate")
try:
    with get_conn() as c:
        r = c.execute("""
            SELECT
                COUNT(*)                                           AS total,
                SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END)   AS wins,
                ROUND(
                    SUM(CASE WHEN status = 'WIN' THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0) * 100, 1
                )                                                  AS win_rate_pct
            FROM bt_signals
            WHERE status IN ('WIN', 'LOSS', 'TIMEOUT')
        """).fetchone()
        print(dict(r))
except Exception as e:
    print(f"  ERROR: {e}")

# ── Context-intel: is it producing data? ─────────────────────
section("Context-intel: sample Valkey snapshot (NVDA)")
try:
    from agent.context_snapshot import get_context_snapshot
    snap = get_context_snapshot("NVDA")
    for k in ("sentiment_30m", "news_shock", "earnings_phase",
              "news_count_30m", "asof_ts", "stale_age_s"):
        print(f"  {k:22s}: {snap.get(k)}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Earnings calendar ────────────────────────────────────────
section("Earnings calendar (PostgreSQL)")
with get_conn() as c:
    r = c.execute("SELECT COUNT(*) total FROM earnings_calendar").fetchone()
    print(f"  total rows: {r['total']}")
    rows = c.execute(
        "SELECT ticker, report_ts, hour FROM earnings_calendar "
        "WHERE report_ts >= NOW() ORDER BY report_ts ASC LIMIT 10"
    ).fetchall()
    if rows:
        for row in rows:
            print(" ", dict(row))
    else:
        print("  No upcoming earnings rows found")

print(f"\n{SEP}\nDone.\n{SEP}\n")
