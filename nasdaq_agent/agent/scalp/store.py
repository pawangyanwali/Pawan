"""Durable storage for scalp plans and execution decisions."""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from agent.db import get_conn, using_postgres

from .models import ScalpSignalPlan

logger = logging.getLogger(__name__)
_init_lock = threading.Lock()
_initialized = False


def init_scalp_tables() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        id_type = "SERIAL PRIMARY KEY" if using_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"
        timestamp_type = "TIMESTAMPTZ" if using_postgres() else "TEXT"
        statements = [
            f"""
            CREATE TABLE IF NOT EXISTS scalp_signal_plans (
                plan_id             TEXT PRIMARY KEY,
                schema_version      INTEGER NOT NULL DEFAULT 1,
                created_at          {timestamp_type} NOT NULL,
                ticker              TEXT NOT NULL,
                side                TEXT NOT NULL,
                valid               INTEGER NOT NULL,
                invalid_reason      TEXT DEFAULT '',
                entry_price         DOUBLE PRECISION DEFAULT 0,
                stop_loss           DOUBLE PRECISION DEFAULT 0,
                tp1                 DOUBLE PRECISION DEFAULT 0,
                tp2                 DOUBLE PRECISION DEFAULT 0,
                risk_per_share      DOUBLE PRECISION DEFAULT 0,
                reward_r            DOUBLE PRECISION DEFAULT 0,
                rr_ratio            DOUBLE PRECISION DEFAULT 0,
                setup_type          TEXT DEFAULT '',
                confidence          DOUBLE PRECISION DEFAULT 0,
                data_source         TEXT DEFAULT '',
                data_age_ms         INTEGER DEFAULT 0,
                bar_age_ms          INTEGER DEFAULT 0,
                plan_json           TEXT NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_execution_decisions (
                id                  {id_type},
                decided_at          {timestamp_type} NOT NULL,
                plan_id             TEXT,
                decision            TEXT NOT NULL,
                reason              TEXT DEFAULT '',
                trade_id            INTEGER,
                detail_json         TEXT DEFAULT '{{}}'
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_trade_outcomes (
                id                  {id_type},
                plan_id             TEXT NOT NULL,
                trade_id            INTEGER NOT NULL UNIQUE,
                closed_at           {timestamp_type} NOT NULL,
                ticker              TEXT NOT NULL,
                side                TEXT NOT NULL,
                context_key         TEXT NOT NULL,
                setup_type          TEXT DEFAULT '',
                session             TEXT DEFAULT '',
                rsi_zone            TEXT DEFAULT '',
                macd_state          TEXT DEFAULT '',
                vwap_event          TEXT DEFAULT '',
                spread_bucket       TEXT DEFAULT '',
                atr_bucket          TEXT DEFAULT '',
                entry_fill          DOUBLE PRECISION DEFAULT 0,
                exit_fill           DOUBLE PRECISION DEFAULT 0,
                tp1_hit             INTEGER DEFAULT 0,
                tp2_hit             INTEGER DEFAULT 0,
                stop_hit            INTEGER DEFAULT 0,
                time_stop           INTEGER DEFAULT 0,
                pnl_r               DOUBLE PRECISION DEFAULT 0,
                pnl_dollar          DOUBLE PRECISION DEFAULT 0,
                mfe_r               DOUBLE PRECISION DEFAULT 0,
                mae_r               DOUBLE PRECISION DEFAULT 0,
                exit_reason         TEXT DEFAULT ''
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_context_stats (
                context_key         TEXT PRIMARY KEY,
                updated_at          {timestamp_type} NOT NULL,
                sample_count        INTEGER NOT NULL DEFAULT 0,
                wins                INTEGER NOT NULL DEFAULT 0,
                losses              INTEGER NOT NULL DEFAULT 0,
                posterior_win_rate  DOUBLE PRECISION DEFAULT 0,
                ewma_expectancy_r   DOUBLE PRECISION DEFAULT 0,
                mean_expectancy_r   DOUBLE PRECISION DEFAULT 0,
                gate_state          TEXT DEFAULT 'ALLOW',
                confidence_floor    DOUBLE PRECISION DEFAULT 0,
                size_mult           DOUBLE PRECISION DEFAULT 1,
                expires_at          {timestamp_type}
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_learning_actions (
                id                  {id_type},
                action_ts           {timestamp_type} NOT NULL,
                context_key         TEXT NOT NULL,
                action_type         TEXT NOT NULL,
                old_state           TEXT DEFAULT 'ALLOW',
                new_state           TEXT NOT NULL,
                old_value           DOUBLE PRECISION DEFAULT 0,
                new_value           DOUBLE PRECISION DEFAULT 0,
                reason              TEXT DEFAULT '',
                expires_at          {timestamp_type}
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_ml_models (
                version_id              TEXT PRIMARY KEY,
                created_at              {timestamp_type} NOT NULL,
                status                  TEXT NOT NULL,
                feature_schema_version  INTEGER NOT NULL,
                sample_count            INTEGER NOT NULL DEFAULT 0,
                train_count             INTEGER NOT NULL DEFAULT 0,
                holdout_count           INTEGER NOT NULL DEFAULT 0,
                selected_count          INTEGER NOT NULL DEFAULT 0,
                trained_through         {timestamp_type},
                evaluated_from          {timestamp_type},
                evaluated_through       {timestamp_type},
                metrics_json            TEXT NOT NULL DEFAULT '{{}}',
                rejection_reason        TEXT DEFAULT '',
                artifact_path           TEXT DEFAULT '',
                artifact_sha256         TEXT DEFAULT ''
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_ml_predictions (
                plan_id                 TEXT PRIMARY KEY,
                predicted_at            {timestamp_type} NOT NULL,
                model_version           TEXT NOT NULL,
                tp1_probability         DOUBLE PRECISION NOT NULL,
                tp2_probability         DOUBLE PRECISION NOT NULL,
                expected_r              DOUBLE PRECISION NOT NULL,
                confidence_adjustment   DOUBLE PRECISION NOT NULL,
                applied                 INTEGER NOT NULL DEFAULT 0
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS scalp_shadow_trades (
                id                  {id_type},
                plan_id             TEXT NOT NULL,
                entry_bar_id        BIGINT NOT NULL,
                opened_at           {timestamp_type} NOT NULL,
                closed_at           {timestamp_type},
                ticker              TEXT NOT NULL,
                side                TEXT NOT NULL,
                setup_type          TEXT DEFAULT '',
                session             TEXT DEFAULT '',
                status              TEXT NOT NULL DEFAULT 'OPEN',
                entry_fill          DOUBLE PRECISION NOT NULL,
                current_price       DOUBLE PRECISION NOT NULL,
                stop_loss           DOUBLE PRECISION NOT NULL,
                original_stop       DOUBLE PRECISION NOT NULL,
                tp1                 DOUBLE PRECISION NOT NULL,
                tp2                 DOUBLE PRECISION NOT NULL,
                risk_per_share      DOUBLE PRECISION NOT NULL,
                shares              INTEGER NOT NULL,
                shares_remaining    INTEGER NOT NULL,
                t1_hit              INTEGER NOT NULL DEFAULT 0,
                t2_hit              INTEGER NOT NULL DEFAULT 0,
                realized_partial    DOUBLE PRECISION NOT NULL DEFAULT 0,
                pnl_r               DOUBLE PRECISION NOT NULL DEFAULT 0,
                pnl_dollar          DOUBLE PRECISION NOT NULL DEFAULT 0,
                mfe_r               DOUBLE PRECISION NOT NULL DEFAULT 0,
                mae_r               DOUBLE PRECISION NOT NULL DEFAULT 0,
                high_watermark      DOUBLE PRECISION NOT NULL,
                low_watermark       DOUBLE PRECISION NOT NULL,
                exit_fill           DOUBLE PRECISION,
                exit_reason         TEXT DEFAULT '',
                plan_json           TEXT NOT NULL,
                UNIQUE (ticker, entry_bar_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_scalp_plans_created ON scalp_signal_plans(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_plans_ticker ON scalp_signal_plans(ticker, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_decisions_plan ON scalp_execution_decisions(plan_id)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_outcomes_context ON scalp_trade_outcomes(context_key, closed_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_outcomes_closed ON scalp_trade_outcomes(closed_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_actions_context ON scalp_learning_actions(context_key, action_ts)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_ml_models_status ON scalp_ml_models(status, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_ml_predictions_model ON scalp_ml_predictions(model_version, predicted_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_shadow_status ON scalp_shadow_trades(status, opened_at)",
            "CREATE INDEX IF NOT EXISTS idx_scalp_shadow_closed ON scalp_shadow_trades(closed_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_scalp_shadow_open_ticker ON scalp_shadow_trades(ticker) WHERE status='OPEN'",
        ]
        try:
            with get_conn() as conn:
                for statement in statements:
                    conn.execute(statement)
            _initialized = True
        except Exception as exc:
            logger.error("[ScalpStore] schema initialization failed: %s", exc)
            raise


def save_plan(plan: ScalpSignalPlan) -> str:
    init_scalp_tables()
    payload = plan.to_dict()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scalp_signal_plans
              (plan_id, schema_version, created_at, ticker, side, valid,
               invalid_reason, entry_price, stop_loss, tp1, tp2, risk_per_share,
               reward_r, rr_ratio, setup_type, confidence, data_source,
               data_age_ms, bar_age_ms, plan_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (plan_id) DO UPDATE SET
              valid=excluded.valid,
              invalid_reason=excluded.invalid_reason,
              confidence=excluded.confidence,
              plan_json=excluded.plan_json
            """,
            (
                plan.plan_id,
                plan.schema_version,
                plan.created_at,
                plan.ticker,
                plan.side.value,
                int(plan.valid),
                plan.invalid_reason,
                plan.entry,
                plan.stop_loss,
                plan.tp1,
                plan.tp2,
                plan.risk_per_share,
                plan.reward_r,
                plan.rr_ratio,
                plan.setup_type,
                plan.confidence,
                plan.source.value,
                plan.data_age_ms,
                plan.bar_age_ms,
                json.dumps(payload, separators=(",", ":")),
            ),
        )
    return plan.plan_id


def record_execution_decision(
    plan_id: str | None,
    decision: str,
    *,
    reason: str = "",
    trade_id: int | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    from datetime import datetime, timezone

    init_scalp_tables()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scalp_execution_decisions
              (decided_at, plan_id, decision, reason, trade_id, detail_json)
            VALUES (?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                plan_id,
                decision,
                reason,
                trade_id,
                json.dumps(detail or {}, separators=(",", ":")),
            ),
        )


def latest_plans(limit: int = 250) -> list[dict[str, Any]]:
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT plan_json FROM scalp_signal_plans
            ORDER BY created_at DESC LIMIT ?
            """,
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row["plan_json"]))
        except Exception:
            continue
    return result


def latest_context_stats(limit: int = 100) -> list[dict[str, Any]]:
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM scalp_context_stats ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def latest_learning_actions(limit: int = 100) -> list[dict[str, Any]]:
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM scalp_learning_actions ORDER BY action_ts DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def latest_outcomes(limit: int = 100) -> list[dict[str, Any]]:
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM scalp_trade_outcomes ORDER BY closed_at DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def learning_dashboard_data(
    *, context_limit: int = 100, action_limit: int = 30, outcome_limit: int = 30
) -> dict[str, Any]:
    """Read command-center learning data through one pooled connection."""
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        contexts = conn.execute(
            "SELECT * FROM scalp_context_stats ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(int(context_limit), 500)),),
        ).fetchall()
        actions = conn.execute(
            "SELECT * FROM scalp_learning_actions ORDER BY action_ts DESC LIMIT ?",
            (max(1, min(int(action_limit), 500)),),
        ).fetchall()
        outcomes = conn.execute(
            "SELECT * FROM scalp_trade_outcomes ORDER BY closed_at DESC LIMIT ?",
            (max(1, min(int(outcome_limit), 500)),),
        ).fetchall()
        champion = conn.execute(
            "SELECT * FROM scalp_ml_models WHERE status='CHAMPION' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        evaluations = conn.execute(
            "SELECT * FROM scalp_ml_models ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        counts = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM scalp_trade_outcomes) AS outcomes,
              (SELECT COUNT(*) FROM scalp_context_stats) AS contexts,
              (SELECT COUNT(*) FROM scalp_learning_actions) AS actions
            """
        ).fetchone()
    return {
        "contexts": [dict(row) for row in contexts],
        "recent_actions": [dict(row) for row in actions],
        "recent_outcomes": [dict(row) for row in outcomes],
        "ml_champion": dict(champion) if champion else None,
        "ml_evaluations": [dict(row) for row in evaluations],
        "counts": dict(counts) if counts else {
            "outcomes": 0, "contexts": 0, "actions": 0,
        },
    }


def scalp_outcome_count() -> int:
    """Return the durable canonical outcome count for learner sample gating."""
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM scalp_trade_outcomes"
        ).fetchone()
    return int((row or {}).get("count") or 0)


def record_ml_evaluation(metadata: dict[str, Any]) -> None:
    init_scalp_tables()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scalp_ml_models
              (version_id, created_at, status, feature_schema_version,
               sample_count, train_count, holdout_count, selected_count,
               trained_through, evaluated_from, evaluated_through,
               metrics_json, rejection_reason, artifact_path, artifact_sha256)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                metadata["version_id"], metadata["created_at"], metadata["status"],
                metadata["feature_schema_version"], metadata.get("sample_count", 0),
                metadata.get("train_count", 0), metadata.get("holdout_count", 0),
                metadata.get("selected_count", 0), metadata.get("trained_through"),
                metadata.get("evaluated_from"), metadata.get("evaluated_through"),
                json.dumps(metadata.get("metrics") or {}, separators=(",", ":")),
                metadata.get("rejection_reason", ""), metadata.get("artifact_path", ""),
                metadata.get("artifact_sha256", ""),
            ),
        )


def promote_ml_model(metadata: dict[str, Any]) -> None:
    """Atomically archive the old champion and register the validated challenger."""
    init_scalp_tables()
    with get_conn() as conn:
        conn.execute("UPDATE scalp_ml_models SET status='ARCHIVED' WHERE status='CHAMPION'")
        conn.execute(
            """
            INSERT INTO scalp_ml_models
              (version_id, created_at, status, feature_schema_version,
               sample_count, train_count, holdout_count, selected_count,
               trained_through, evaluated_from, evaluated_through,
               metrics_json, rejection_reason, artifact_path, artifact_sha256)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                metadata["version_id"], metadata["created_at"], "CHAMPION",
                metadata["feature_schema_version"], metadata.get("sample_count", 0),
                metadata.get("train_count", 0), metadata.get("holdout_count", 0),
                metadata.get("selected_count", 0), metadata.get("trained_through"),
                metadata.get("evaluated_from"), metadata.get("evaluated_through"),
                json.dumps(metadata.get("metrics") or {}, separators=(",", ":")),
                "", metadata.get("artifact_path", ""), metadata.get("artifact_sha256", ""),
            ),
        )


def champion_ml_model() -> dict[str, Any] | None:
    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        row = conn.execute(
            "SELECT * FROM scalp_ml_models WHERE status='CHAMPION' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def record_ml_prediction(
    *, plan_id: str, predicted_at: str, model_version: str,
    tp1_probability: float, tp2_probability: float, expected_r: float,
    confidence_adjustment: float, applied: bool,
) -> None:
    init_scalp_tables()
    values = (
        predicted_at, model_version, tp1_probability, tp2_probability,
        expected_r, confidence_adjustment, int(applied), plan_id,
    )
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT plan_id FROM scalp_ml_predictions WHERE plan_id=?", (plan_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE scalp_ml_predictions
                SET predicted_at=?, model_version=?, tp1_probability=?,
                    tp2_probability=?, expected_r=?, confidence_adjustment=?, applied=?
                WHERE plan_id=?
                """,
                values,
            )
        else:
            conn.execute(
                """
                INSERT INTO scalp_ml_predictions
                  (predicted_at, model_version, tp1_probability, tp2_probability,
                   expected_r, confidence_adjustment, applied, plan_id)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                values,
            )
