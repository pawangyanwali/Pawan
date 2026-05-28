"""
Phase 2 Adaptive Trading Learning Engine — AI-TRD-CL-002.

Builds on Phase 1 (algo_learning_engine.py) with five advanced components:

  1. ConceptDriftDetector      — PSI-based feature drift monitoring (60-day window)
  2. WalkForwardValidator      — periodic performance validation → ModelVersionRegistry
  3. CrossTickerTransferEngine — hierarchical knowledge transfer (Global→Sector→Ticker)
  4. StagedDeploymentController— SHADOW/PAPER_ONLY/PARTIAL_LIVE/FULL_LIVE routing
  5. OperatorNotificationService— throttled webhook + in-app WebSocket notifications

Phase2Engine — top-level coordinator called by AlgoLearningEngine.run_cycle().

Design constraints:
- Self-contained: no circular imports with Phase 1 (imports lazily)
- All state persisted to data/ (gitignored); survives restarts
- Every component fails silently — never crashes the trading system
- Thread-safe throughout
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── PSI thresholds ─────────────────────────────────────────────────────────────
PSI_WARNING  = 0.10   # moderate change — monitor
PSI_MATERIAL = 0.25   # significant change — trigger model review
PSI_BINS     = 10     # number of histogram bins for PSI

# ── Deployment modes (ordered from least to most permissive) ───────────────────
DEPLOYMENT_MODES = ["SHADOW", "PAPER_ONLY", "PARTIAL_LIVE", "FULL_LIVE"]
_MODE_INDEX      = {m: i for i, m in enumerate(DEPLOYMENT_MODES)}

# ── Tier thresholds for cross-ticker transfer ──────────────────────────────────
_TIER_GLOBAL     = 10    # ticker has < 10 trades → use global only
_TIER_BLEND      = 30    # 10-29 trades → blend
_BLEND_TICKER_WT = 0.30  # weight of ticker-specific data in blend zone
_TICKER_WT       = 0.80  # weight of ticker-specific data when ≥30 trades

# ── Notification throttle ──────────────────────────────────────────────────────
_THROTTLE_SECS_DEFAULT = 900  # 15 minutes
_THROTTLE_SECS_DRIFT   = 300  # 5 minutes for drift alerts


# ── Internal DB query ──────────────────────────────────────────────────────────

def _query_p2_outcomes(days: int = 60) -> pd.DataFrame:
    """
    Full-column query of resolved bt_signals for Phase 2 analytics.
    Returns empty DataFrame on any error.
    """
    try:
        from agent.db import get_conn
        with get_conn() as c:
            rows = c.execute("""
                SELECT ticker, direction, confidence, session, regime,
                       vwap_event, rsi_zone, rsi_value, entry_type,
                       rr_ratio, algo_name, sector_etf, status,
                       pnl_pct, r_multiple, bars_tracked, fired_at
                FROM bt_signals
                WHERE status IN ('WIN', 'LOSS', 'TIMEOUT')
                  AND is_counterfactual = 0
                  AND fired_at >= datetime('now', ? || ' days')
                ORDER BY fired_at DESC
                LIMIT 2000
            """, (f"-{days}",)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["won"] = (df["status"] == "WIN").astype(int)
        return df
    except Exception as exc:
        logger.debug(f"[P2] _query_p2_outcomes error: {exc}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ConceptDriftDetector
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed bin ranges for each monitored feature
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "confidence": (0.0,  100.0),
    "rr_ratio":   (0.0,    5.0),
    "rsi_value":  (0.0,  100.0),
    "pnl_pct":    (-5.0,   5.0),
}


def _compute_psi(reference: np.ndarray, current: np.ndarray,
                 lo: float, hi: float, bins: int = PSI_BINS) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    PSI = Σ (current_pct - ref_pct) * ln(current_pct / ref_pct)

    Returns 0.0 if either array is empty or an error occurs.
    """
    try:
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current,   bins=edges)
        # Convert to proportions, floor at 0.0001 to avoid log(0)
        ref_pct = np.maximum(ref_counts / max(len(reference), 1), 1e-4)
        cur_pct = np.maximum(cur_counts / max(len(current),   1), 1e-4)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(max(psi, 0.0), 6)
    except Exception as exc:
        logger.debug(f"[PSI] compute error: {exc}")
        return 0.0


class ConceptDriftDetector:
    """
    Monitors feature distributions for concept drift using Population Stability Index.

    - Builds a reference distribution from the first REFERENCE_MIN resolved trades.
    - Computes PSI for each tracked feature on every CHECK_INTERVAL cycles.
    - Emits WARNING (PSI 0.10-0.25) or MATERIAL (PSI ≥0.25) drift events.
    - Persisted to data/drift_state.json.
    """

    _PATH              = _DATA_DIR / "drift_state.json"
    REFERENCE_MIN      = 30    # minimum trades before reference is built
    CHECK_INTERVAL     = 5     # run PSI check every N cycles
    CURRENT_WINDOW_MIN = 20    # minimum current-window trades for valid PSI

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {
            "reference_built":   False,
            "reference_n":       0,
            "reference_arrays":  {},   # {feature: [values]}
            "last_psi":          {},   # {feature: psi_score}
            "drift_events":      [],   # [{ts, feature, psi, severity}]
            "last_check_cycle":  0,
        }

    def load(self) -> None:
        try:
            if self._PATH.exists():
                raw = json.loads(self._PATH.read_text())
                with self._lock:
                    self._state.update(raw)
                if self._validate_reference():
                    self.save()   # persist the cleared state immediately
        except Exception as exc:
            logger.warning(f"[DriftDetector] load error: {exc}")

    def _validate_reference(self) -> bool:
        """Reset reference if arrays are corrupt (all-zero or zero-variance). Returns True if reset."""
        with self._lock:
            if not self._state.get("reference_built"):
                return False
            arrays = self._state.get("reference_arrays", {})
            corrupt = False
            for feat, vals in arrays.items():
                if not vals:
                    corrupt = True
                    break
                arr = np.array(vals, dtype=float)
                if np.std(arr) < 1e-6:
                    corrupt = True
                    logger.warning(
                        f"[DriftDetector] Corrupt reference for '{feat}' "
                        f"(std={np.std(arr):.6f}) — resetting drift reference"
                    )
                    break
            if corrupt:
                self._state["reference_built"]  = False
                self._state["reference_n"]      = 0
                self._state["reference_arrays"] = {}
                self._state["last_psi"]         = {}
                self._state["drift_events"]     = []
            return corrupt

    def save(self) -> None:
        try:
            with self._lock:
                data = dict(self._state)
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[DriftDetector] save error: {exc}")

    def update(self, df: pd.DataFrame, cycle_num: int) -> dict[str, tuple[float, str]]:
        """
        Update drift detection state. Returns {feature: (psi, severity)} for any
        features that crossed PSI thresholds this cycle.
        Severity: "WARNING" | "MATERIAL" | "NONE"
        """
        alerts: dict[str, tuple[float, str]] = {}
        try:
            if df.empty:
                return alerts

            with self._lock:
                ref_built = self._state["reference_built"]
                ref_n     = self._state["reference_n"]

            # ── Build reference distribution on first REFERENCE_MIN trades ──────
            if not ref_built:
                if len(df) >= self.REFERENCE_MIN:
                    ref_rows = df.tail(self.REFERENCE_MIN)
                    new_refs: dict[str, list] = {}
                    for feat in _FEATURE_RANGES:
                        col = feat
                        if col in ref_rows.columns:
                            vals = ref_rows[col].dropna().tolist()
                            if vals:
                                new_refs[feat] = vals
                    with self._lock:
                        self._state["reference_arrays"] = new_refs
                        self._state["reference_built"]  = True
                        self._state["reference_n"]      = len(ref_rows)
                return alerts

            # ── Check PSI every CHECK_INTERVAL cycles ────────────────────────
            with self._lock:
                last_check = self._state.get("last_check_cycle", 0)
            if cycle_num - last_check < self.CHECK_INTERVAL:
                return alerts

            # ── Compute PSI for each feature ─────────────────────────────────
            recent = df.head(max(self.CURRENT_WINDOW_MIN, 50))  # last N trades
            if len(recent) < self.CURRENT_WINDOW_MIN:
                return alerts

            psi_scores: dict[str, float] = {}
            for feat, (lo, hi) in _FEATURE_RANGES.items():
                if feat not in recent.columns:
                    continue
                with self._lock:
                    ref_vals = self._state["reference_arrays"].get(feat, [])
                if len(ref_vals) < 5:
                    continue
                cur_vals = recent[feat].dropna().to_numpy()
                ref_arr  = np.array(ref_vals, dtype=float)
                psi = _compute_psi(ref_arr, cur_vals, lo, hi)
                psi_scores[feat] = psi

                severity = "NONE"
                if psi >= PSI_MATERIAL:
                    severity = "MATERIAL"
                elif psi >= PSI_WARNING:
                    severity = "WARNING"

                if severity != "NONE":
                    alerts[feat] = (psi, severity)
                    drift_event = {
                        "ts":       datetime.now(timezone.utc).isoformat(),
                        "feature":  feat,
                        "psi":      psi,
                        "severity": severity,
                        "cycle":    cycle_num,
                    }
                    with self._lock:
                        self._state["drift_events"].append(drift_event)
                        # Keep only last 100 events
                        self._state["drift_events"] = self._state["drift_events"][-100:]

            with self._lock:
                self._state["last_psi"]        = psi_scores
                self._state["last_check_cycle"] = cycle_num

        except Exception as exc:
            logger.warning(f"[DriftDetector] update error: {exc}")

        return alerts

    def get_drift_summary(self) -> dict:
        """Returns current drift state for API/dashboard."""
        with self._lock:
            return {
                "reference_built": self._state.get("reference_built", False),
                "reference_n":     self._state.get("reference_n", 0),
                "last_psi":        dict(self._state.get("last_psi", {})),
                "recent_events":   list(self._state.get("drift_events", []))[-10:],
                "material_drifts": [
                    f for f, psi in self._state.get("last_psi", {}).items()
                    if psi >= PSI_MATERIAL
                ],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WalkForwardValidator
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Computes performance metrics from resolved outcomes and registers new versions
    in the Phase 1 ModelVersionRegistry for champion/challenger evaluation.

    Runs every CHECK_CYCLE_INTERVAL learning cycles (not more often than that).
    Requires MIN_TRADES resolved outcomes before running.
    Persisted to data/wf_validation.json.
    """

    _PATH               = _DATA_DIR / "wf_validation.json"
    CHECK_CYCLE_INTERVAL = 10   # run validation every N cycles
    MIN_TRADES           = 30   # minimum resolved trades for meaningful metrics

    def __init__(self):
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._last_run_cycle: int  = 0

    def load(self) -> None:
        try:
            if self._PATH.exists():
                data = json.loads(self._PATH.read_text())
                with self._lock:
                    self._history       = data.get("history", [])
                    self._last_run_cycle = int(data.get("last_run_cycle", 0))
        except Exception as exc:
            logger.warning(f"[WFValidator] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = {
                    "history":        self._history[-50:],  # keep last 50
                    "last_run_cycle": self._last_run_cycle,
                }
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[WFValidator] save error: {exc}")

    def run_validation(self, df: pd.DataFrame, cycle_num: int) -> Optional[dict]:
        """
        Compute performance metrics from resolved outcomes.
        Registers result as a challenger in ModelVersionRegistry.
        Returns metrics dict, or None if not enough data or wrong cycle.
        """
        try:
            with self._lock:
                last = self._last_run_cycle

            if cycle_num - last < self.CHECK_CYCLE_INTERVAL:
                return None

            if df.empty or len(df) < self.MIN_TRADES:
                return None

            metrics = self._compute_metrics(df)
            if not metrics:
                return None

            # Register with Phase 1 ModelVersionRegistry
            version_id = f"wf_v{cycle_num}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}"
            try:
                from agent.algo_learning_engine import get_engine as _get_ale
                _get_ale()._model_reg.register_version(version_id, metrics)
                can_promote, reasons = _get_ale()._model_reg.evaluate_promotion(version_id)
                metrics["version_id"]   = version_id
                metrics["can_promote"]  = can_promote
                metrics["gate_reasons"] = reasons
                if can_promote:
                    _get_ale()._model_reg.promote(version_id)
                    metrics["promoted"] = True
                    logger.info(f"[WFValidator] Promoted {version_id} to champion")
            except Exception as reg_exc:
                logger.debug(f"[WFValidator] registry error: {reg_exc}")
                metrics["version_id"] = version_id

            metrics["cycle_num"] = cycle_num
            metrics["ts"]        = datetime.now(timezone.utc).isoformat()

            with self._lock:
                self._history.append(metrics)
                self._last_run_cycle = cycle_num

            logger.info(
                f"[WFValidator] Cycle {cycle_num}: WR={metrics.get('win_rate', 0):.1%}  "
                f"PF={metrics.get('profit_factor', 0):.2f}  "
                f"Sharpe={metrics.get('sharpe', 0):.2f}  "
                f"n={metrics.get('n_trades', 0)}"
            )
            return metrics

        except Exception as exc:
            logger.warning(f"[WFValidator] run_validation error: {exc}")
            return None

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        """Compute win_rate, profit_factor, expectancy, sharpe, max_drawdown."""
        try:
            resolved = df[df["status"].isin(["WIN", "LOSS", "TIMEOUT"])].copy()
            if len(resolved) < self.MIN_TRADES:
                return {}

            # Filter out null pnl_pct
            resolved = resolved.dropna(subset=["pnl_pct"])
            n = len(resolved)
            if n < 10:
                return {}

            pnl    = resolved["pnl_pct"].astype(float)
            wins   = (resolved["status"] == "WIN").sum()
            losses = n - wins

            win_rate     = float(wins / n)
            expectancy   = float(pnl.mean())

            gross_profit = float(pnl[pnl > 0].sum()) if (pnl > 0).any() else 0.0
            gross_loss   = float(abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit > 0) * 9.99

            sharpe = self._compute_sharpe(pnl)
            max_dd = self._compute_max_drawdown(pnl)

            return {
                "n_trades":      n,
                "wins":          int(wins),
                "losses":        int(losses),
                "win_rate":      round(win_rate,     4),
                "expectancy":    round(expectancy,   6),
                "profit_factor": round(profit_factor, 4),
                "sharpe":        round(sharpe,        4),
                "max_drawdown":  round(max_dd,        6),
            }
        except Exception as exc:
            logger.debug(f"[WFValidator] _compute_metrics error: {exc}")
            return {}

    @staticmethod
    def _compute_sharpe(returns: pd.Series) -> float:
        """Annualized Sharpe treating each trade as an independent observation."""
        try:
            if len(returns) < 3:
                return 0.0
            std = returns.std()
            if std < 1e-9:
                return 0.0
            # Annualize assuming ~500 trades/year (common for day-trading)
            return float(returns.mean() / std * math.sqrt(500))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_max_drawdown(returns: pd.Series) -> float:
        """Max peak-to-trough drawdown as a positive fraction (e.g. 0.12 = 12%)."""
        try:
            equity = (1 + returns / 100).cumprod()
            peak   = equity.cummax()
            dd     = (equity - peak) / peak
            return float(abs(dd.min()))
        except Exception:
            return 0.0

    def get_latest_metrics(self) -> dict:
        """Return the most recent validation run metrics."""
        with self._lock:
            return dict(self._history[-1]) if self._history else {}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CrossTickerTransferEngine
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sector(ticker: str) -> str:
    """Return the sector ETF symbol for a ticker, or 'QQQ' as default."""
    try:
        from agent.sector_etf import SECTOR_MAP, _DEFAULT_SECTOR_ETF
        return SECTOR_MAP.get(ticker, _DEFAULT_SECTOR_ETF)
    except Exception:
        return "QQQ"


class CrossTickerTransferEngine:
    """
    Hierarchical knowledge transfer across tickers.

    Three-tier system — blend weights depend on how many trades a ticker has:
      GLOBAL tier  (<10 trades): 100% global patterns
      BLEND tier   (10–29):      30% ticker + 70% global
      TICKER tier  (≥30 trades): 80% ticker + 20% global

    Sector patterns are computed as a middle layer between global and ticker,
    but for parameter blending we use the simpler Global↔Ticker blend to keep
    the math tractable.

    Persisted to data/transfer_state.json.
    """

    _PATH = _DATA_DIR / "transfer_state.json"

    def __init__(self):
        self._lock = threading.Lock()
        # {algo_family: {"win_rate": float, "n_trades": int}}
        self._global:  dict[str, dict] = {}
        # {sector_etf: {algo_family: {"win_rate": float, "n_trades": int}}}
        self._sector:  dict[str, dict[str, dict]] = {}
        # {ticker: {algo_family: {"win_rate": float, "n_trades": int}}}
        self._ticker:  dict[str, dict[str, dict]] = {}
        # {ticker: int}
        self._ticker_counts: dict[str, int] = {}

    def load(self) -> None:
        try:
            if self._PATH.exists():
                raw = json.loads(self._PATH.read_text())
                with self._lock:
                    self._global       = raw.get("global", {})
                    self._sector       = raw.get("sector", {})
                    self._ticker       = raw.get("ticker", {})
                    self._ticker_counts = raw.get("ticker_counts", {})
        except Exception as exc:
            logger.warning(f"[TransferEngine] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = {
                    "global":        self._global,
                    "sector":        self._sector,
                    "ticker":        self._ticker,
                    "ticker_counts": self._ticker_counts,
                }
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[TransferEngine] save error: {exc}")

    def transfer(self, df: pd.DataFrame, cycle_num: int) -> dict:
        """
        Aggregate per-ticker/sector/global outcome patterns from resolved trades.
        Returns a summary of tiers updated and patterns transferred.
        """
        summary = {"tickers_updated": 0, "sectors_updated": 0, "global_updated": False}
        try:
            if df.empty or "algo_name" not in df.columns:
                return summary

            from agent.algo_learning_engine import _ALGO_FAMILY_MAP

            # ── Global aggregation ────────────────────────────────────────────
            global_new = self._aggregate_by_family(df, _ALGO_FAMILY_MAP)
            with self._lock:
                self._global = self._blend_aggregates(self._global, global_new)
            summary["global_updated"] = bool(global_new)

            # ── Sector aggregation ────────────────────────────────────────────
            if "sector_etf" in df.columns:
                for sector, grp in df.groupby("sector_etf"):
                    if not sector or len(grp) < 3:
                        continue
                    sec_new = self._aggregate_by_family(grp, _ALGO_FAMILY_MAP)
                    with self._lock:
                        if sector not in self._sector:
                            self._sector[sector] = {}
                        self._sector[sector] = self._blend_aggregates(
                            self._sector[sector], sec_new
                        )
                    summary["sectors_updated"] += 1

            # ── Ticker aggregation ────────────────────────────────────────────
            if "ticker" in df.columns:
                for ticker, grp in df.groupby("ticker"):
                    if not ticker:
                        continue
                    tkr_new = self._aggregate_by_family(grp, _ALGO_FAMILY_MAP)
                    with self._lock:
                        if ticker not in self._ticker:
                            self._ticker[ticker] = {}
                        self._ticker[ticker] = self._blend_aggregates(
                            self._ticker[ticker], tkr_new
                        )
                        self._ticker_counts[ticker] = len(grp)
                    summary["tickers_updated"] += 1

        except Exception as exc:
            logger.warning(f"[TransferEngine] transfer error: {exc}")

        return summary

    @staticmethod
    def _aggregate_by_family(df: pd.DataFrame,
                              family_map: dict[str, str]) -> dict[str, dict]:
        """Compute per-family win rates from a trades DataFrame."""
        result: dict[str, dict] = {}
        try:
            if "algo_name" not in df.columns or "won" not in df.columns:
                return result
            for algo_name, grp in df.groupby("algo_name"):
                if not algo_name:
                    continue
                family = family_map.get(str(algo_name), "")
                if not family:
                    continue
                n     = len(grp)
                wins  = int(grp["won"].sum())
                wr    = wins / n if n else 0.0
                if family not in result:
                    result[family] = {"win_rate": 0.0, "n_trades": 0}
                # Accumulate (will be blended via EWMA below)
                old = result[family]
                combined_n = old["n_trades"] + n
                combined_wr = (
                    (old["win_rate"] * old["n_trades"] + wr * n) / combined_n
                    if combined_n > 0 else 0.0
                )
                result[family] = {"win_rate": round(combined_wr, 4),
                                   "n_trades": combined_n}
        except Exception as exc:
            logger.debug(f"[TransferEngine] _aggregate_by_family error: {exc}")
        return result

    @staticmethod
    def _blend_aggregates(old: dict[str, dict],
                           new: dict[str, dict],
                           alpha: float = 0.20) -> dict[str, dict]:
        """EWMA blend of new aggregate data into existing state."""
        result = dict(old)
        for family, new_stats in new.items():
            if family in result:
                old_wr = float(result[family].get("win_rate", 0.0))
                new_wr = float(new_stats.get("win_rate", 0.0))
                blended_wr = alpha * new_wr + (1 - alpha) * old_wr
                old_n  = int(result[family].get("n_trades", 0))
                new_n  = int(new_stats.get("n_trades", 0))
                result[family] = {
                    "win_rate": round(blended_wr, 4),
                    "n_trades": old_n + new_n,
                }
            else:
                result[family] = new_stats
        return result

    def get_ticker_tier(self, ticker: str) -> str:
        """Return tier name: GLOBAL | BLEND | TICKER."""
        with self._lock:
            n = self._ticker_counts.get(ticker, 0)
        if n < _TIER_GLOBAL:
            return "GLOBAL"
        if n < _TIER_BLEND:
            return "BLEND"
        return "TICKER"

    def get_blended_win_rate(self, ticker: str, algo_family: str) -> float:
        """
        Return blended win rate for (ticker, algo_family) using hierarchical tiers.
        Falls back gracefully at each level.
        """
        try:
            with self._lock:
                global_wr  = self._global.get(algo_family, {}).get("win_rate", 0.5)
                ticker_n   = self._ticker_counts.get(ticker, 0)
                ticker_wr  = (self._ticker.get(ticker, {})
                              .get(algo_family, {}).get("win_rate", global_wr))

            if ticker_n < _TIER_GLOBAL:
                return round(global_wr, 4)
            elif ticker_n < _TIER_BLEND:
                blended = _BLEND_TICKER_WT * ticker_wr + (1 - _BLEND_TICKER_WT) * global_wr
            else:
                blended = _TICKER_WT * ticker_wr + (1 - _TICKER_WT) * global_wr
            return round(blended, 4)
        except Exception:
            return 0.5

    def get_transfer_summary(self) -> dict:
        """Dashboard summary of transfer state."""
        with self._lock:
            return {
                "global_families": len(self._global),
                "sector_count":    len(self._sector),
                "ticker_count":    len(self._ticker),
                "tier_distribution": {
                    "GLOBAL": sum(1 for t in self._ticker_counts.values() if t < _TIER_GLOBAL),
                    "BLEND":  sum(1 for t in self._ticker_counts.values()
                                  if _TIER_GLOBAL <= t < _TIER_BLEND),
                    "TICKER": sum(1 for t in self._ticker_counts.values() if t >= _TIER_BLEND),
                },
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. StagedDeploymentController
# ═══════════════════════════════════════════════════════════════════════════════

# Advancement gates per mode transition
_ADVANCEMENT_GATES: dict[str, dict] = {
    "SHADOW": {
        "target_mode":            "PAPER_ONLY",
        "min_win_rate":           0.45,
        "min_profit_factor":      0.0,    # not required for shadow→paper
        "consecutive_required":   3,
        "description":            "counterfactual win_rate > 45% for 3 cycles",
    },
    "PAPER_ONLY": {
        "target_mode":            "PARTIAL_LIVE",
        "min_win_rate":           0.50,
        "min_profit_factor":      1.0,
        "consecutive_required":   3,
        "description":            "paper win_rate > 50% AND PF > 1.0 for 3 cycles",
    },
    "PARTIAL_LIVE": {
        "target_mode":            "FULL_LIVE",
        "min_win_rate":           0.55,
        "min_profit_factor":      1.10,
        "consecutive_required":   5,
        "description":            "paper win_rate > 55% AND PF > 1.10 for 5 cycles",
    },
}

# UCB weight threshold for PARTIAL_LIVE algo selection
_PARTIAL_LIVE_UCB_THRESHOLD = 1.20


class StagedDeploymentController:
    """
    State machine that governs how algo signals are routed.

    Modes (ordered from most conservative to most permissive):
      SHADOW       → signals logged as counterfactual only (no paper trades)
      PAPER_ONLY   → all signals go to paper trading
      PARTIAL_LIVE → high-UCB algos (weight > 1.20) route to live; others paper
      FULL_LIVE    → all algos route to live trading

    Auto-advancement: mode advances when performance gates are met for
    `consecutive_required` consecutive validation cycles.

    Persisted to data/deployment_state.json.
    """

    _PATH = _DATA_DIR / "deployment_state.json"

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {
            "current_mode":         "PAPER_ONLY",   # safe default
            "consecutive_passes":   0,
            "last_mode_change":     None,
            "last_mode_change_cycle": 0,
            "mode_history":         [],
        }

    def load(self) -> None:
        try:
            if self._PATH.exists():
                raw = json.loads(self._PATH.read_text())
                with self._lock:
                    self._state.update(raw)
        except Exception as exc:
            logger.warning(f"[DeployController] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = dict(self._state)
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[DeployController] save error: {exc}")

    def get_routing(self, algo_name: str, ucb_weight: float) -> str:
        """
        Return routing decision for a signal: SHADOW | PAPER | LIVE.
        Called by scanner.py before routing each algo signal.
        """
        with self._lock:
            mode = self._state["current_mode"]

        if mode == "SHADOW":
            return "SHADOW"
        if mode == "PAPER_ONLY":
            return "PAPER"
        if mode == "PARTIAL_LIVE":
            return "LIVE" if ucb_weight >= _PARTIAL_LIVE_UCB_THRESHOLD else "PAPER"
        if mode == "FULL_LIVE":
            return "LIVE"
        return "PAPER"  # safe default

    def evaluate_advancement(self, metrics: dict, cycle_num: int) -> bool:
        """
        Check if advancement criteria are met. Advances mode if consecutive
        gates are satisfied. Returns True if mode advanced.
        """
        try:
            with self._lock:
                current_mode = self._state["current_mode"]

            gates = _ADVANCEMENT_GATES.get(current_mode)
            if not gates:
                return False   # FULL_LIVE has no further advancement

            win_rate      = float(metrics.get("win_rate", 0) or 0)
            profit_factor = float(metrics.get("profit_factor", 0) or 0)
            n_trades      = int(metrics.get("n_trades", 0) or 0)

            # Need minimum sample for meaningful evaluation
            if n_trades < 20:
                return False

            gate_passed = (
                win_rate >= gates["min_win_rate"] and
                (gates["min_profit_factor"] == 0.0 or
                 profit_factor >= gates["min_profit_factor"])
            )

            with self._lock:
                if gate_passed:
                    self._state["consecutive_passes"] += 1
                else:
                    self._state["consecutive_passes"] = 0

                passes   = self._state["consecutive_passes"]
                required = gates["consecutive_required"]

                if passes >= required:
                    old_mode  = self._state["current_mode"]
                    new_mode  = gates["target_mode"]
                    self._state["current_mode"]            = new_mode
                    self._state["consecutive_passes"]      = 0
                    self._state["last_mode_change"]        = datetime.now(timezone.utc).isoformat()
                    self._state["last_mode_change_cycle"]  = cycle_num
                    self._state["mode_history"].append({
                        "ts":        self._state["last_mode_change"],
                        "from_mode": old_mode,
                        "to_mode":   new_mode,
                        "cycle_num": cycle_num,
                        "metrics":   metrics,
                    })
                    # Keep last 20 transitions
                    self._state["mode_history"] = self._state["mode_history"][-20:]
                    logger.info(
                        f"[DeployController] Mode advanced: {old_mode} → {new_mode}"
                        f" (cycle {cycle_num}, WR={win_rate:.1%}, PF={profit_factor:.2f})"
                    )
                    return True

        except Exception as exc:
            logger.warning(f"[DeployController] evaluate_advancement error: {exc}")

        return False

    def get_current_mode(self) -> str:
        with self._lock:
            return self._state["current_mode"]

    def get_status(self) -> dict:
        with self._lock:
            return {
                "current_mode":       self._state["current_mode"],
                "consecutive_passes": self._state["consecutive_passes"],
                "last_mode_change":   self._state.get("last_mode_change"),
                "mode_history":       list(self._state.get("mode_history", []))[-5:],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OperatorNotificationService
# ═══════════════════════════════════════════════════════════════════════════════

_THROTTLE_MAP: dict[str, int] = {
    "DRIFT_ALERT":            _THROTTLE_SECS_DRIFT,
    "MODEL_PROMOTED":         _THROTTLE_SECS_DEFAULT,
    "DEPLOYMENT_ADVANCED":    _THROTTLE_SECS_DEFAULT,
    "PARAM_CHANGED":          _THROTTLE_SECS_DEFAULT,
    "FILTER_RELAXED":         _THROTTLE_SECS_DEFAULT,
    "WALK_FORWARD_COMPLETE":  _THROTTLE_SECS_DEFAULT,
    "TRANSFER_APPLIED":       _THROTTLE_SECS_DEFAULT * 2,  # less urgent — 30 min
}


class OperatorNotificationService:
    """
    Unified notification layer for all Phase 1 + Phase 2 significant events.

    Channels:
      1. AuditLogger JSONL (always) via Phase 1 AuditLogger
      2. Webhook POST to ALERT_WEBHOOK_URL (if env var set)
      3. In-app WebSocket broadcast via main.py ConnectionManager

    Per-event throttling prevents notification storms during rapid adaptation.
    Persisted notification history to data/notifications.jsonl.
    """

    _PATH = _DATA_DIR / "notifications.jsonl"

    def __init__(self):
        self._lock       = threading.Lock()
        self._last_sent: dict[str, float] = {}   # {event_type: unix_timestamp}

    def notify(self, event_type: str, message: str, data: dict,
               severity: str = "INFO") -> bool:
        """
        Emit a notification if not throttled.
        Returns True if notification was actually sent (not throttled).
        """
        import time
        try:
            throttle = _THROTTLE_MAP.get(event_type, _THROTTLE_SECS_DEFAULT)
            now      = time.time()
            with self._lock:
                last = self._last_sent.get(event_type, 0.0)
                if now - last < throttle:
                    return False
                self._last_sent[event_type] = now

            entry = {
                "ts":         datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "severity":   severity,
                "message":    message,
                "data":       data,
            }

            # ── 1. Append to JSONL ───────────────────────────────────────────
            try:
                self._PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(self._PATH, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

            # ── 2. Webhook ───────────────────────────────────────────────────
            webhook = os.environ.get("ALERT_WEBHOOK_URL", "")
            if webhook:
                try:
                    import urllib.request
                    payload = json.dumps(entry).encode()
                    req = urllib.request.Request(
                        webhook, data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=5)
                except Exception as exc:
                    logger.debug(f"[Notifier] webhook error: {exc}")

            # ── 3. In-app WebSocket broadcast ────────────────────────────────
            self._broadcast_in_app(event_type, message, entry)

            logger.info(f"[Notifier] {severity} {event_type}: {message}")
            return True

        except Exception as exc:
            logger.debug(f"[Notifier] notify error: {exc}")
            return False

    def _broadcast_in_app(self, event_type: str, message: str, entry: dict) -> None:
        """Push notification to all connected WebSocket clients."""
        try:
            import asyncio
            import main as _main
            manager = getattr(_main, "manager", None)
            if manager is None:
                return
            payload = json.dumps({
                "type":       "LEARNING_EVENT",
                "event_type": event_type,
                "message":    message,
                "severity":   entry.get("severity", "INFO"),
                "ts":         entry.get("ts", ""),
                "data":       entry.get("data", {}),
            })
            # Run broadcast in the existing event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), loop)
            except RuntimeError:
                pass   # no running loop — WebSocket not available
        except Exception as exc:
            logger.debug(f"[Notifier] broadcast_in_app error: {exc}")

    def get_recent_notifications(self, limit: int = 50) -> list[dict]:
        """Return recent notifications from the JSONL log."""
        try:
            if not self._PATH.exists():
                return []
            lines = self._PATH.read_text().strip().splitlines()
            entries = []
            for line in reversed(lines[-limit * 2:]):
                try:
                    entries.append(json.loads(line))
                    if len(entries) >= limit:
                        break
                except Exception:
                    pass
            return list(reversed(entries))
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# Phase2Engine — top-level coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class Phase2Engine:
    """
    Orchestrates all 5 Phase 2 learning components.
    Called by AlgoLearningEngine.run_cycle() after Phase 1 processing.

    The engine is stateless at the call level — all state lives inside
    the individual components, persisted to data/.
    """

    def __init__(self):
        self._drift      = ConceptDriftDetector()
        self._validator  = WalkForwardValidator()
        self._transfer   = CrossTickerTransferEngine()
        self._deployment = StagedDeploymentController()
        self._notifier   = OperatorNotificationService()
        self._lock       = threading.Lock()

    def load(self) -> None:
        """Load all persistent state."""
        for comp in (self._drift, self._validator, self._transfer, self._deployment):
            try:
                comp.load()
            except Exception as exc:
                logger.warning(f"[Phase2Engine] load error ({type(comp).__name__}): {exc}")

    def save(self) -> None:
        """Save all persistent state."""
        for comp in (self._drift, self._validator, self._transfer, self._deployment):
            try:
                comp.save()
            except Exception as exc:
                logger.warning(f"[Phase2Engine] save error ({type(comp).__name__}): {exc}")

    def run_cycle(self, outcomes_df: pd.DataFrame, cycle_num: int) -> None:
        """
        Full Phase 2 learning cycle. Called after AlgoLearningEngine (Phase 1)
        processes its cycle.

        Steps:
          1. Fetch enriched outcome data from bt_signals (needs pnl_pct, ticker, etc.)
          2. ConceptDriftDetector — check for feature distribution shifts
          3. WalkForwardValidator — compute metrics, register challenger
          4. CrossTickerTransferEngine — update hierarchical patterns
          5. StagedDeploymentController — evaluate mode advancement
          6. Emit notifications for significant events
          7. Save all state
        """
        try:
            # Use enriched data from DB (includes pnl_pct, ticker, sector_etf)
            rich_df = _query_p2_outcomes(days=60)
            if rich_df.empty:
                # Fall back to the Phase 1 outcomes_df if DB unavailable
                rich_df = outcomes_df.copy() if outcomes_df is not None else pd.DataFrame()

            # ── 1. Concept Drift ─────────────────────────────────────────────
            drift_alerts = {}
            try:
                drift_alerts = self._drift.update(rich_df, cycle_num)
                for feat, (psi, severity) in drift_alerts.items():
                    self._notifier.notify(
                        "DRIFT_ALERT",
                        f"Feature '{feat}' distribution shifted (PSI={psi:.3f}, {severity})",
                        {"feature": feat, "psi": psi, "severity": severity, "cycle": cycle_num},
                        severity=severity,
                    )
            except Exception as exc:
                logger.debug(f"[Phase2] drift error: {exc}")

            # ── 2. Walk-Forward Validation ───────────────────────────────────
            wf_metrics = None
            try:
                wf_metrics = self._validator.run_validation(rich_df, cycle_num)
                if wf_metrics:
                    self._notifier.notify(
                        "WALK_FORWARD_COMPLETE",
                        f"Validation cycle {cycle_num}: WR={wf_metrics.get('win_rate', 0):.1%}  "
                        f"PF={wf_metrics.get('profit_factor', 0):.2f}",
                        wf_metrics,
                    )
                    if wf_metrics.get("promoted"):
                        self._notifier.notify(
                            "MODEL_PROMOTED",
                            f"Version {wf_metrics.get('version_id')} promoted to champion",
                            wf_metrics,
                            severity="WARNING",
                        )
            except Exception as exc:
                logger.debug(f"[Phase2] validator error: {exc}")

            # ── 3. Cross-Ticker Transfer ─────────────────────────────────────
            transfer_summary = {}
            try:
                transfer_summary = self._transfer.transfer(rich_df, cycle_num)
                if transfer_summary.get("tickers_updated", 0) > 0:
                    self._notifier.notify(
                        "TRANSFER_APPLIED",
                        f"Transfer learning updated {transfer_summary['tickers_updated']} tickers",
                        transfer_summary,
                    )
            except Exception as exc:
                logger.debug(f"[Phase2] transfer error: {exc}")

            # ── 4. Staged Deployment Advancement ────────────────────────────
            try:
                metrics_for_gate = wf_metrics or {}
                if not metrics_for_gate and not rich_df.empty:
                    # Compute quick win rate from available data
                    n_rich   = len(rich_df)
                    wins_rich = int(rich_df.get("won", rich_df.get("status", pd.Series()).eq("WIN")).sum())
                    if n_rich > 0:
                        metrics_for_gate = {
                            "win_rate":      wins_rich / n_rich,
                            "profit_factor": 1.0,  # neutral default
                            "n_trades":      n_rich,
                        }
                advanced = self._deployment.evaluate_advancement(metrics_for_gate, cycle_num)
                if advanced:
                    self._notifier.notify(
                        "DEPLOYMENT_ADVANCED",
                        f"Deployment mode advanced to {self._deployment.get_current_mode()}",
                        {**metrics_for_gate,
                         "new_mode": self._deployment.get_current_mode(),
                         "cycle_num": cycle_num},
                        severity="WARNING",
                    )
            except Exception as exc:
                logger.debug(f"[Phase2] deployment error: {exc}")

            # ── 5. Save ──────────────────────────────────────────────────────
            self.save()

        except Exception as exc:
            logger.warning(f"[Phase2Engine] run_cycle error: {exc}")

    # ── Public API used by algo_learning_engine.py + scanner.py ──────────────

    def get_routing(self, algo_name: str, ucb_weight: float) -> str:
        """Routing decision for scanner.py: SHADOW | PAPER | LIVE."""
        try:
            return self._deployment.get_routing(algo_name, ucb_weight)
        except Exception:
            return "PAPER"

    def get_drift_summary(self) -> dict:
        """Drift state for dashboard API."""
        try:
            return self._drift.get_drift_summary()
        except Exception:
            return {}

    def get_deployment_mode(self) -> str:
        """Current deployment mode."""
        try:
            return self._deployment.get_current_mode()
        except Exception:
            return "PAPER_ONLY"

    def get_status(self) -> dict:
        """Combined Phase 2 status for dashboard."""
        try:
            return {
                "drift":          self._drift.get_drift_summary(),
                "deployment":     self._deployment.get_status(),
                "validation":     self._validator.get_latest_metrics(),
                "transfer":       self._transfer.get_transfer_summary(),
                "notifications":  self._notifier.get_recent_notifications(limit=10),
            }
        except Exception as exc:
            logger.debug(f"[Phase2Engine] get_status error: {exc}")
            return {}

    def get_blended_win_rate(self, ticker: str, algo_family: str) -> float:
        """Cross-ticker blended win rate for (ticker, algo_family)."""
        try:
            return self._transfer.get_blended_win_rate(ticker, algo_family)
        except Exception:
            return 0.5

    def get_ticker_tier(self, ticker: str) -> str:
        """Returns GLOBAL | BLEND | TICKER for a given ticker."""
        try:
            return self._transfer.get_ticker_tier(ticker)
        except Exception:
            return "GLOBAL"


# ── Module-level singleton ─────────────────────────────────────────────────────

_p2_engine:      Optional[Phase2Engine] = None
_p2_engine_lock  = threading.Lock()


def get_phase2_engine() -> Phase2Engine:
    global _p2_engine
    if _p2_engine is None:
        with _p2_engine_lock:
            if _p2_engine is None:
                _p2_engine = Phase2Engine()
                _p2_engine.load()
    return _p2_engine


def get_deployment_mode() -> str:
    """Safe public accessor for current deployment mode."""
    try:
        return get_phase2_engine().get_deployment_mode()
    except Exception:
        return "PAPER_ONLY"


def get_routing(algo_name: str, ucb_weight: float = 1.0) -> str:
    """Safe public accessor for signal routing decision."""
    try:
        return get_phase2_engine().get_routing(algo_name, ucb_weight)
    except Exception:
        return "PAPER"
