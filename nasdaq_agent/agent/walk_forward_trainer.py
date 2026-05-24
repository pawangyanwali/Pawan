"""
Multi-timeframe Walk-Forward Trainer (Phase 3 Part 1).

Runs replay_signals() on historical OHLCV for a set of tickers across
multiple timeframes, then translates labeled outcomes into parameter
recommendations for AlgoLearningEngine.

This is a BATCH process — run after market close or by weekend_learner,
NOT in the hot scanner path.

Persists results to data/wf_trainer.json.

Public API
----------
WalkForwardTrainer.run(tickers, ohlcv_map, cycle_num) → summary dict
WalkForwardTrainer.get_status() → last-run summary for dashboard
get_walk_forward_trainer() → module-level singleton
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_STATE_PATH = _DATA_DIR / "wf_trainer.json"

# Timeframes processed by the trainer (must be present in ohlcv_map)
_TIMEFRAMES = ("5min", "15min")

# Minimum number of replayed signals required to consider recommendation logic
_MIN_N_FOR_CONF_RVOL = 20   # conf_gate / rvol_gate adjustments
_MIN_N_FOR_TARGET    = 15   # target_mult adjustments

# Win-rate / pnl thresholds for parameter recommendation
_WR_LOW       = 0.35   # below → raise conf_gate
_WR_HIGH      = 0.60   # above → lower rvol_gate
_PNL_LOW      = -0.5   # below → lower target_mult
_PNL_HIGH     = 1.2    # above → raise target_mult

# Adjustment magnitudes (the registry clamps to max_change automatically)
_CONF_GATE_RAISE  = 1.0
_RVOL_GATE_LOWER  = 0.05
_TARGET_MULT_DEC  = 0.1
_TARGET_MULT_INC  = 0.1


def _fetch_ohlcv_for_ticker(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Helper: fetch 5min and 15min OHLCV for a single ticker from the
    historical cache (already populated by weekend_learner Phase 1).

    Returns a dict {timeframe: DataFrame} — missing timeframes are omitted.
    Gracefully returns {} on any error so the caller can skip this ticker.
    """
    result: dict[str, pd.DataFrame] = {}
    try:
        from agent.historical_cache import get_bars
        for tf, interval in (("5min", "5min"), ("15min", "15min")):
            try:
                df = get_bars(ticker, interval, min_bars=60)
                if df is not None and not df.empty:
                    result[tf] = df
            except Exception as exc:
                logger.warning(
                    f"[WFTrainer] fetch_ohlcv {ticker}/{tf} failed: {exc}"
                )
    except Exception as exc:
        logger.warning(f"[WFTrainer] historical_cache unavailable for {ticker}: {exc}")
    return result


class WalkForwardTrainer:
    """
    Runs replay_signals() on historical OHLCV for a set of tickers across
    multiple timeframes, then translates the labeled outcomes into parameter
    recommendations for AlgoLearningEngine.

    This is a BATCH process — run after market close or by weekend_learner,
    NOT in the hot scanner path.

    Persists results to data/wf_trainer.json.
    """

    def __init__(self) -> None:
        self._last_summary: dict[str, Any] = {}

    # ── core ──────────────────────────────────────────────────────────────────

    def run(
        self,
        tickers: list[str],
        ohlcv_map: dict[str, dict[str, pd.DataFrame]],
        # ohlcv_map[ticker]["5min"]  = DataFrame with OHLCV
        # ohlcv_map[ticker]["15min"] = DataFrame with OHLCV
        cycle_num: int = 0,
    ) -> dict[str, Any]:
        """
        For each ticker + timeframe:
          1. Call replay_signals() to get labeled records
          2. Compute win rate and avg pnl_r per family
          3. Aggregate across all tickers
          4. Apply param recommendations to AlgoLearningEngine registry
        Returns a summary dict with per-family stats.
        """
        from agent.walk_forward import replay_signals

        started_at = datetime.now(timezone.utc).isoformat()
        all_records: list[dict[str, Any]] = []
        ticker_errors: list[str] = []

        # ── Step 1 & 2: replay signals for every ticker × timeframe ──────────
        for ticker in tickers:
            tf_map = ohlcv_map.get(ticker, {})
            if not tf_map:
                logger.warning(f"[WFTrainer] {ticker}: no OHLCV data — skipping")
                ticker_errors.append(ticker)
                continue

            for tf in _TIMEFRAMES:
                df = tf_map.get(tf)
                if df is None or df.empty:
                    logger.debug(f"[WFTrainer] {ticker}/{tf}: empty DataFrame — skipping")
                    continue

                # Guard against all-NaN DataFrames
                try:
                    if df.isnull().all(axis=None).all() if hasattr(df.isnull().all(axis=None), 'all') else df.isnull().all().all():
                        logger.warning(
                            f"[WFTrainer] {ticker}/{tf}: all-NaN DataFrame — skipping"
                        )
                        continue
                except Exception:
                    pass

                try:
                    records = replay_signals(
                        ticker=ticker,
                        df=df,
                        timeframe=tf,
                    )
                    all_records.extend(records)
                    logger.debug(
                        f"[WFTrainer] {ticker}/{tf}: {len(records)} records replayed"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[WFTrainer] {ticker}/{tf} replay_signals failed: {exc}"
                    )

        # ── Step 3: aggregate per-family stats ────────────────────────────────
        family_stats = self._aggregate_by_family(all_records)

        # ── Step 4: apply parameter recommendations ───────────────────────────
        try:
            recommendations = self._apply_recommendations(
                family_stats, cycle_num=cycle_num
            )
        except Exception as exc:
            logger.warning(f"[WFTrainer] _apply_recommendations failed: {exc}")
            recommendations = []

        finished_at = datetime.now(timezone.utc).isoformat()
        total = len(all_records)
        wins  = sum(1 for r in all_records if r.get("won"))
        overall_wr = round(wins / total, 3) if total else 0.0
        avg_pnl_r  = (
            round(sum(r["pnl_r"] for r in all_records) / total, 3)
            if total else 0.0
        )

        summary: dict[str, Any] = {
            "started_at":       started_at,
            "finished_at":      finished_at,
            "tickers_processed": len(tickers),
            "tickers_skipped":  len(ticker_errors),
            "total_records":    total,
            "overall_win_rate": overall_wr,
            "overall_avg_pnl_r": avg_pnl_r,
            "family_stats":     family_stats,
            "recommendations":  recommendations,
            "cycle_num":        cycle_num,
        }

        self._last_summary = summary
        self._persist(summary)
        logger.info(
            f"[WFTrainer] run complete — {total} records, "
            f"{overall_wr*100:.1f}% WR, {len(recommendations)} recommendations"
        )
        return summary

    # ── helpers ───────────────────────────────────────────────────────────────

    def _aggregate_by_family(
        self, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Group records by algo family (derived from signal direction + RSI zone
        heuristic — we use a neutral 'REPLAY' family for walk-forward signals
        because walk_forward.py generates its own rule-based signals rather than
        named algo signals).

        Since walk_forward records don't carry an algo_name, we group by
        direction as a proxy family key so the stats are still useful.
        """
        # walk_forward records don't have algo family directly; we bucket by
        # direction as a primary grouping for recommendations.
        from_direction: dict[str, list[dict]] = {}
        for r in records:
            key = r.get("direction", "UNKNOWN")
            from_direction.setdefault(key, []).append(r)

        stats: dict[str, dict[str, Any]] = {}
        for family, recs in from_direction.items():
            n    = len(recs)
            wins = sum(1 for r in recs if r.get("won"))
            pnl_sum = sum(r.get("pnl_r", 0.0) for r in recs)
            stats[family] = {
                "n":        n,
                "wins":     wins,
                "win_rate": round(wins / n, 3) if n else 0.0,
                "avg_pnl_r": round(pnl_sum / n, 3) if n else 0.0,
            }
        return stats

    def _apply_recommendations(
        self,
        family_stats: dict[str, dict[str, Any]],
        cycle_num: int,
    ) -> list[dict[str, Any]]:
        """
        Translate per-family win rate / pnl stats into ParameterControlRegistry
        updates. Uses get_engine()._registry.update(...).

        The walk-forward signal families (BUY / SELL) are mapped to all algo
        families so recommendations propagate broadly.  The registry enforces
        its own bounds and cooldown.
        """
        try:
            from agent.algo_learning_engine import get_engine, _ALL_FAMILIES
        except Exception as exc:
            logger.warning(f"[WFTrainer] cannot import AlgoLearningEngine: {exc}")
            return []

        try:
            engine   = get_engine()
            registry = engine._registry
        except Exception as exc:
            logger.warning(f"[WFTrainer] get_engine() failed: {exc}")
            return []

        applied: list[dict[str, Any]] = []

        # For each direction-based family bucket, apply rules to all algo families
        for direction_key, stats in family_stats.items():
            n        = stats.get("n", 0)
            win_rate = stats.get("win_rate", 0.0)
            avg_pnl  = stats.get("avg_pnl_r", 0.0)

            # Determine which recommendations fire
            param_updates: list[tuple[str, float, str]] = []

            if n >= _MIN_N_FOR_CONF_RVOL and win_rate < _WR_LOW:
                param_updates.append((
                    "conf_gate",
                    None,   # sentinel → add delta to current value
                    f"wf_trainer: low win_rate {win_rate:.2f} (n={n}), raise conf_gate",
                ))

            if n >= _MIN_N_FOR_CONF_RVOL and win_rate > _WR_HIGH:
                param_updates.append((
                    "rvol_gate",
                    None,
                    f"wf_trainer: high win_rate {win_rate:.2f} (n={n}), lower rvol_gate",
                ))

            if n >= _MIN_N_FOR_TARGET and avg_pnl < _PNL_LOW:
                param_updates.append((
                    "target_mult",
                    None,
                    f"wf_trainer: low avg_pnl_r {avg_pnl:.2f} (n={n}), lower target_mult",
                ))

            if n >= _MIN_N_FOR_TARGET and avg_pnl > _PNL_HIGH:
                param_updates.append((
                    "target_mult",
                    None,
                    f"wf_trainer: high avg_pnl_r {avg_pnl:.2f} (n={n}), raise target_mult",
                ))

            if not param_updates:
                continue

            # Apply to every known algo family
            for algo_family in _ALL_FAMILIES:
                for param, _, reason in param_updates:
                    try:
                        current = registry.get(algo_family, param)
                        if "lower rvol_gate" in reason:
                            new_val = current - _RVOL_GATE_LOWER
                        elif "raise conf_gate" in reason:
                            new_val = current + _CONF_GATE_RAISE
                        elif "lower target_mult" in reason:
                            new_val = current - _TARGET_MULT_DEC
                        elif "raise target_mult" in reason:
                            new_val = current + _TARGET_MULT_INC
                        else:
                            continue

                        ok = registry.update(
                            algo_family, param, new_val, reason, cycle_num
                        )
                        if ok:
                            applied.append({
                                "algo_family":  algo_family,
                                "param":        param,
                                "new_val":      new_val,
                                "reason":       reason,
                                "direction_key": direction_key,
                            })
                    except Exception as exc:
                        logger.warning(
                            f"[WFTrainer] registry.update "
                            f"{algo_family}.{param} failed: {exc}"
                        )

        return applied

    # ── persistence ───────────────────────────────────────────────────────────

    def _persist(self, summary: dict[str, Any]) -> None:
        try:
            _STATE_PATH.write_text(
                json.dumps(summary, indent=2, default=str)
            )
        except Exception as exc:
            logger.warning(f"[WFTrainer] persist failed: {exc}")

    def _load_persisted(self) -> dict[str, Any]:
        try:
            if _STATE_PATH.exists():
                return json.loads(_STATE_PATH.read_text())
        except Exception as exc:
            logger.warning(f"[WFTrainer] load persisted state failed: {exc}")
        return {}

    # ── public API ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Returns last run summary for dashboard API."""
        if not self._last_summary:
            self._last_summary = self._load_persisted()
        return dict(self._last_summary)


# ── Module-level singleton ────────────────────────────────────────────────────

_trainer_instance: WalkForwardTrainer | None = None
_trainer_lock = __import__("threading").Lock()


def get_walk_forward_trainer() -> WalkForwardTrainer:
    """Return the module-level WalkForwardTrainer singleton."""
    global _trainer_instance
    if _trainer_instance is None:
        with _trainer_lock:
            if _trainer_instance is None:
                _trainer_instance = WalkForwardTrainer()
    return _trainer_instance
