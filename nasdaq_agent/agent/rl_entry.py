"""
Reinforcement Learning entry timing optimizer.

Uses a simple Q-learning agent to learn WHEN to enter a trade within a
confirmed signal zone. Instead of entering immediately on every signal,
the RL agent observes the next few bars and decides: ENTER_NOW or WAIT.

State (8 features):
  - momentum_1bar: last bar return (positive = price moving up)
  - rsi_norm: RSI/100 (0=oversold, 1=overbought)
  - vol_ratio: current volume / 20-bar avg
  - spread_to_support: % distance to nearest support
  - spread_to_resistance: % distance to nearest resistance
  - vwap_deviation: % above/below VWAP
  - bars_since_signal: how long since the signal (0=fresh, 1=stale)
  - time_of_day_norm: 0=open, 1=close

Actions:
  0 = WAIT  (don't enter yet, observe one more bar)
  1 = ENTER (enter the trade now)

Reward:
  +1.0 if entered and price moved ≥ 0.35% in correct direction within 6 bars
  -0.5 if entered and price moved ≤ 0% within 6 bars (no follow-through)
  -0.1 for each WAIT action (opportunity cost of waiting)
   0.0 if never entered within 5 bars (signal expired)

Learning: epsilon-greedy Q-table updated after each episode.
"""
from __future__ import annotations

import logging
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_QTABLE_PATH = Path(__file__).parent.parent / "data" / "models" / "rl_qtable.json"
_QTABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
ALPHA        = 0.10    # learning rate
GAMMA        = 0.90    # discount factor
EPSILON_INIT = 0.30    # initial exploration rate
EPSILON_MIN  = 0.05    # minimum exploration
EPSILON_DECAY = 0.995  # decay per episode
N_STATE_BINS  = 5      # discretize each feature into N bins
N_ACTIONS     = 2      # 0=WAIT, 1=ENTER
WAIT_COST     = 0.10   # reward penalty per bar of waiting
WIN_REWARD    = 1.00   # reward for profitable entry
LOSS_REWARD   = -0.50  # reward for losing entry
MAX_WAIT_BARS = 5      # signal expires after this many WAIT steps


def _discretize(value: float, n_bins: int = N_STATE_BINS, lo: float = 0.0, hi: float = 1.0) -> int:
    """Map a continuous value to a discrete bin index [0, n_bins-1]."""
    clipped = float(np.clip(value, lo, hi))
    norm    = (clipped - lo) / (hi - lo + 1e-9)
    return min(int(norm * n_bins), n_bins - 1)


def _state_key(state: dict) -> str:
    """Convert state dict to a string key for the Q-table."""
    bins = [
        _discretize(state.get("momentum_1bar",       0.0), lo=-0.01, hi=0.01),
        _discretize(state.get("rsi_norm",            0.5), lo=0.0,   hi=1.0),
        _discretize(state.get("vol_ratio",           1.0), lo=0.0,   hi=3.0),
        _discretize(state.get("spread_to_support",   0.5), lo=0.0,   hi=0.02),
        _discretize(state.get("vwap_deviation",      0.0), lo=-0.01, hi=0.01),
        _discretize(state.get("bars_since_signal",   0.0), lo=0.0,   hi=5.0),
        _discretize(state.get("time_of_day_norm",    0.5), lo=0.0,   hi=1.0),
    ]
    return "|".join(map(str, bins))


class RLEntryAgent:
    """
    Tabular Q-learning agent for entry timing.
    Persists Q-table to disk across restarts.
    """

    def __init__(self):
        self.q_table:  dict[str, list[float]] = {}
        self.epsilon:  float = EPSILON_INIT
        self.episodes: int   = 0
        self._load()

    def _load(self) -> None:
        try:
            if _QTABLE_PATH.exists():
                data = json.loads(_QTABLE_PATH.read_text())
                self.q_table  = data.get("q_table", {})
                self.epsilon  = float(data.get("epsilon", EPSILON_INIT))
                self.episodes = int(data.get("episodes", 0))
                logger.info(f"[RLEntry] Loaded Q-table: {len(self.q_table)} states, "
                            f"epsilon={self.epsilon:.3f}, episodes={self.episodes}")
        except Exception as e:
            logger.debug(f"[RLEntry] Load failed: {e}")

    def _save(self) -> None:
        try:
            data = {"q_table": self.q_table, "epsilon": self.epsilon, "episodes": self.episodes}
            _QTABLE_PATH.write_text(json.dumps(data))
        except Exception as e:
            logger.debug(f"[RLEntry] Save failed: {e}")

    def _q(self, key: str) -> list[float]:
        """Get Q-values for a state, initialising to zeros if unseen."""
        if key not in self.q_table:
            self.q_table[key] = [0.0] * N_ACTIONS
        return self.q_table[key]

    def choose_action(self, state: dict) -> int:
        """Epsilon-greedy action selection. Returns 0 (WAIT) or 1 (ENTER)."""
        key = _state_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q = self._q(key)
        return int(np.argmax(q))

    def update(self, state: dict, action: int, reward: float, next_state: Optional[dict]) -> None:
        """Q-learning update step."""
        key  = _state_key(state)
        q    = self._q(key)
        if next_state is not None:
            next_key = _state_key(next_state)
            max_next = max(self._q(next_key))
        else:
            max_next = 0.0
        q[action] += ALPHA * (reward + GAMMA * max_next - q[action])
        self.q_table[key] = q

    def record_episode(self, episode_result: dict) -> None:
        """
        Process a completed episode (signal resolved).

        episode_result keys:
          states:  list of state dicts (one per bar observed)
          actions: list of int (0=WAIT, 1=ENTER)
          entered_at_bar: int or None (None if never entered)
          final_pnl_pct:  float (actual pct move after entry, positive=win)
          direction:      "BUY" or "SELL"
        """
        states   = episode_result.get("states", [])
        actions  = episode_result.get("actions", [])
        entered  = episode_result.get("entered_at_bar")
        pnl      = float(episode_result.get("final_pnl_pct", 0.0))
        d        = episode_result.get("direction", "BUY")

        if not states or not actions:
            return

        # Compute rewards
        from agent.signal_tracker import SHORT_WIN_PCT, SLIPPAGE_PCT
        win_threshold = SHORT_WIN_PCT + SLIPPAGE_PCT
        d_sign = 1 if "BUY" in d else -1

        for i, (s, a) in enumerate(zip(states, actions)):
            is_last = (i == len(states) - 1)
            if a == 1 and i == entered:
                # Entry action: reward based on subsequent pnl
                adj_pnl = pnl * d_sign
                reward  = WIN_REWARD if adj_pnl >= win_threshold else LOSS_REWARD
            elif a == 0:
                # Wait action: small cost
                reward = -WAIT_COST
            else:
                reward = 0.0
            next_s = states[i + 1] if i + 1 < len(states) else None
            self.update(s, a, reward, next_s)

        self.episodes += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # Save periodically
        if self.episodes % 50 == 0:
            self._save()
            logger.info(f"[RLEntry] Checkpoint: {self.episodes} episodes, "
                        f"epsilon={self.epsilon:.3f}, states={len(self.q_table)}")

    def get_entry_recommendation(
        self,
        momentum_1bar:    float = 0.0,
        rsi_norm:         float = 0.5,
        vol_ratio:        float = 1.0,
        spread_to_support: float = 0.005,
        vwap_deviation:   float = 0.0,
        bars_since_signal: float = 0.0,
        time_of_day_norm:  float = 0.5,
    ) -> tuple[str, float]:
        """
        Get entry recommendation for current market state.
        Returns ("ENTER" or "WAIT", confidence 0-1).
        """
        state = {
            "momentum_1bar":    momentum_1bar,
            "rsi_norm":         rsi_norm,
            "vol_ratio":        vol_ratio,
            "spread_to_support": spread_to_support,
            "vwap_deviation":   vwap_deviation,
            "bars_since_signal": bars_since_signal,
            "time_of_day_norm": time_of_day_norm,
        }
        key = _state_key(state)
        q   = self._q(key)

        if all(v == 0.0 for v in q):
            # Untrained state — default to ENTER (don't hold back early on)
            return "ENTER", 0.5

        action = int(np.argmax(q))
        # Confidence: margin between best and worst action, scaled 0-1
        margin = abs(q[0] - q[1]) / (abs(q[0]) + abs(q[1]) + 1e-9)
        conf   = round(float(np.clip(0.5 + margin * 0.5, 0.5, 1.0)), 3)
        return ("ENTER" if action == 1 else "WAIT"), conf

    def get_stats(self) -> dict:
        return {
            "episodes":    self.episodes,
            "epsilon":     round(self.epsilon, 4),
            "states_seen": len(self.q_table),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_rl_agent = RLEntryAgent()


def get_entry_recommendation(**kwargs) -> tuple[str, float]:
    """Get RL-based entry timing recommendation."""
    return _rl_agent.get_entry_recommendation(**kwargs)


def record_trade_episode(episode_result: dict) -> None:
    """Feed a completed trade episode back to the RL agent for learning."""
    _rl_agent.record_episode(episode_result)


def get_rl_stats() -> dict:
    return _rl_agent.get_stats()
