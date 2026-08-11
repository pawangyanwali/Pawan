from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def _evidence(*, slow_last_day: bool = False):
    start = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0) - timedelta(days=7)
    cycles = []
    day = start
    market_days = 0
    while market_days < 5:
        if day.weekday() < 5:
            for index in range(300):
                cycles.append({
                    "bucket_ts": day + timedelta(minutes=index),
                    "session": "REGULAR",
                    "universe_total": 424,
                    "data_gap_count": 4,
                    "execution_universe_total": 200,
                    "execution_data_gap_count": 2,
                    "cycle_ms": 20_000 if slow_last_day and market_days == 4 else 4_000,
                    "live_count": 410,
                    "rest_count": 10,
                    "stale_count": 4,
                })
            market_days += 1
        day += timedelta(days=1)
    trials = [{"pnl_r": 1.0 if index % 2 == 0 else -0.5} for index in range(100)]
    return cycles, trials


def _run(monkeypatch, *, slow_last_day: bool = False):
    import agent.config_manager as manager
    import agent.db as db
    import agent.scalp.activation as activation
    import agent.scalp.store as store

    cycles, trials = _evidence(slow_last_day=slow_last_day)

    class Conn:
        def execute(self, sql, _params=()):
            return SimpleNamespace(fetchall=lambda: trials if "shadow_trades" in sql else cycles)

    @contextmanager
    def get_conn(read_only=False):
        yield Conn()

    monkeypatch.setattr(db, "get_conn", get_conn)
    monkeypatch.setattr(store, "init_scalp_tables", lambda: None)
    monkeypatch.setattr(manager, "config", SimpleNamespace(get=lambda _key, default=None: default))
    latest = max(row["bucket_ts"] for row in cycles).date()
    monkeypatch.setattr(activation, "_latest_completed_market_date", lambda: latest)
    return activation._build_report()


def test_five_day_activation_gate_accepts_operational_and_positive_evidence(monkeypatch):
    report = _run(monkeypatch)
    assert report["ready"] is True
    assert report["operational_ready"] is True
    assert report["statistical_ready"] is True
    assert len(report["days"]) == 5
    assert report["expectancy_r"] > 0
    assert report["profit_factor"] >= 1.1
    assert report["days"][0]["data_gap_pct"] == 1.0
    assert report["days"][0]["quote_coverage_pct"] == 99.06


def test_five_day_activation_gate_rejects_one_slow_market_day(monkeypatch):
    report = _run(monkeypatch, slow_last_day=True)
    assert report["ready"] is False
    assert report["operational_ready"] is False
    assert report["statistical_ready"] is True
    assert "FIVE_CONSECUTIVE_MARKET_DAYS_NOT_SLA_COMPLIANT" in report["reasons"]


def test_activation_uses_only_policy_approved_shadow_episode_v3_evidence(monkeypatch):
    import agent.scalp.activation as activation

    captured = []
    cycles, trials = _evidence()

    class Conn:
        def execute(self, sql, _params=()):
            captured.append(sql)
            return SimpleNamespace(
                fetchall=lambda: trials if "shadow_trades" in sql else cycles
            )

    @contextmanager
    def get_conn(read_only=False):
        yield Conn()

    monkeypatch.setattr("agent.db.get_conn", get_conn)
    monkeypatch.setattr("agent.scalp.store.init_scalp_tables", lambda: None)
    monkeypatch.setattr(
        "agent.config_manager.config",
        SimpleNamespace(get=lambda _key, default=None: default),
    )
    latest = max(row["bucket_ts"] for row in cycles).date()
    monkeypatch.setattr(activation, "_latest_completed_market_date", lambda: latest)

    report = activation._build_report()

    trial_sql = next(sql for sql in captured if "shadow_trades" in sql)
    assert "evidence_version>=3" in trial_sql
    assert report["canonical_evidence_contract"] == "POLICY_APPROVED_SHADOW_EPISODE_V3"


def test_five_day_activation_gate_excludes_sparse_extended_hours(monkeypatch):
    import agent.scalp.activation as activation

    cycles, trials = _evidence()
    for row in list(cycles):
        if row["bucket_ts"].minute == 0:
            cycles.append({
                **row,
                "session": "AFTER_HOURS",
                "data_gap_count": row["universe_total"],
                "execution_data_gap_count": row["execution_universe_total"],
                "cycle_ms": 90_000,
                "live_count": 0,
                "rest_count": 0,
                "stale_count": row["universe_total"],
            })

    class Conn:
        def execute(self, sql, _params=()):
            return SimpleNamespace(
                fetchall=lambda: trials if "shadow_trades" in sql else cycles
            )

    @contextmanager
    def get_conn(read_only=False):
        yield Conn()

    import agent.config_manager as manager
    import agent.db as db
    import agent.scalp.store as store

    monkeypatch.setattr(db, "get_conn", get_conn)
    monkeypatch.setattr(store, "init_scalp_tables", lambda: None)
    monkeypatch.setattr(manager, "config", SimpleNamespace(get=lambda _key, default=None: default))
    latest = max(row["bucket_ts"] for row in cycles).date()
    monkeypatch.setattr(activation, "_latest_completed_market_date", lambda: latest)

    report = activation._build_report()

    assert report["operational_ready"] is True
