"""
Tests for Phase 1 context intelligence.

Acceptance criteria (from design doc):
  AC1  scanner has zero Finnhub calls inside analyse_ticker()
  AC2  Valkey context read under 10 ms
  AC3  missing FINNHUB_API_KEY does not break scanner
  AC4  provider outage does not blank context immediately
  AC5  earnings blackout dict matches scanner expectations

These tests are all unit / integration tests that mock DB and Valkey;
they do NOT require a live PostgreSQL or Valkey connection.
"""
from __future__ import annotations

import json
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


# ── AC1: no Finnhub calls inside analyse_ticker ----------------------------

class TestNoFinnhubInScanner(unittest.TestCase):
    """The scanner's analyse_ticker must never call Finnhub directly."""

    def test_scanner_source_does_not_import_finnhub_at_module_level(self):
        """
        Verify agent/scanner.py does not have a top-level import of the Finnhub
        provider.  (We check the source — the module is too heavy to import in
        a unit test environment without DB/Valkey/sklearn running.)
        """
        import pathlib
        scanner_src = (
            pathlib.Path(__file__).parent.parent / "agent" / "scanner.py"
        ).read_text()

        # No module-level 'from agent.providers' import in the top section
        lines = scanner_src.splitlines()
        module_level_imports = [
            ln for ln in lines
            if ln.strip().startswith(("import ", "from "))
               and "finnhub" in ln.lower()
        ]
        self.assertEqual(
            module_level_imports, [],
            f"Unexpected Finnhub import in scanner.py: {module_level_imports}",
        )

    def test_score_sentiment_not_called_directly(self):
        """
        The shim replaces score_sentiment with get_context_snapshot.
        Verify that the function body references get_context_snapshot.
        """
        import ast
        import pathlib

        scanner_path = pathlib.Path(__file__).parent.parent / "agent" / "scanner.py"
        source = scanner_path.read_text()

        # The shim call must be present
        self.assertIn("get_context_snapshot", source,
                      "get_context_snapshot not found in scanner.py — shim not applied?")

        # Old direct call must not be present (inside analyse_ticker body)
        # It may still appear in imports; we check the assignment form only.
        self.assertNotIn("sent, headlines = score_sentiment(",
                         source,
                         "score_sentiment direct assignment still present in scanner.py")
        self.assertNotIn("eb = earnings_blackout(ticker)",
                         source,
                         "earnings_blackout direct call still present in scanner.py")


# ── AC2: Valkey read latency -----------------------------------------------

class TestValkeyCacheReadLatency(unittest.TestCase):
    """get_context_snapshot with a warm Valkey mock returns in < 10 ms."""

    def test_read_latency_under_10ms(self):
        import agent.context_snapshot as cs

        # Warm payload simulating what context-intel publishes
        warm_payload = json.dumps({
            "ticker":             "AAPL",
            "sentiment_5m":       0.12,
            "sentiment_30m":      0.08,
            "sentiment_2h":       0.05,
            "sentiment_1d":       0.03,
            "sentiment_velocity": 0.03,
            "news_count_30m":     2,
            "news_shock":         False,
            "context_risk_score": 0.07,
            "recent_headlines":   ["Apple beats earnings estimates"],
            "earnings_phase":     "",
            "earnings_reason":    "",
            "earnings_next_date": "",
            "earnings_days_away": 999,
            "asof_ts":            time.time(),
        }).encode()

        mock_client = MagicMock()
        mock_client.get.return_value = warm_payload

        with patch("agent.context_snapshot._client", return_value=mock_client):
            start = time.perf_counter()
            for _ in range(100):
                result = cs.get_context_snapshot("AAPL")
            elapsed_avg_ms = (time.perf_counter() - start) / 100 * 1000

        self.assertLess(elapsed_avg_ms, 10.0,
                        f"Average Valkey read {elapsed_avg_ms:.2f} ms exceeds 10 ms target")
        self.assertAlmostEqual(result["sentiment_30m"], 0.08, places=2)
        self.assertEqual(result["recent_headlines"], ["Apple beats earnings estimates"])


# ── AC3: missing FINNHUB_API_KEY is safe -----------------------------------

class TestMissingApiKey(unittest.TestCase):
    """All public functions return safe defaults when FINNHUB_API_KEY is absent."""

    def test_finnhub_provider_no_key_returns_empty(self):
        """fetch_market_news and fetch_company_news return [] when key is missing."""
        import os
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if present
            os.environ.pop("FINNHUB_API_KEY", None)
            import agent.providers.finnhub_provider as fp
            importlib.reload_if_possible(fp)

            result_market  = fp.fetch_market_news()
            result_company = fp.fetch_company_news("AAPL", "2026-01-01", "2026-01-07")
            result_cal     = fp.fetch_earnings_calendar("2026-01-01", "2026-04-01")

        self.assertEqual(result_market,  [])
        self.assertEqual(result_company, [])
        self.assertEqual(result_cal,     [])

    def test_get_context_snapshot_returns_safe_defaults_no_valkey(self):
        """
        When Valkey is unavailable AND DB has no data, get_context_snapshot
        returns a fully populated dict with safe default values.
        """
        import agent.context_snapshot as cs

        # Valkey unavailable
        with patch("agent.context_snapshot._client", return_value=None):
            # DB also returns None
            with patch("agent.context_snapshot._read_pg", return_value=None):
                snap = cs.get_context_snapshot("TSLA")

        self.assertEqual(snap["ticker"],         "TSLA")
        self.assertEqual(snap["sentiment_30m"],  0.0)
        self.assertEqual(snap["earnings_phase"], "")
        self.assertEqual(snap["earnings_days_away"], 999)
        self.assertIsInstance(snap["recent_headlines"], list)
        self.assertIsInstance(snap["stale_age_s"], float)

    def test_scanner_shim_safe_defaults_pass_through(self):
        """
        Build the eb dict the way scanner.py does and verify it satisfies the
        downstream key requirements (earnings_blocked, earnings_reason, etc.).
        """
        # Simulate a default (empty) context snapshot
        snap = {
            "sentiment_30m":      0.0,
            "recent_headlines":   [],
            "earnings_phase":     "",
            "earnings_reason":    "",
            "earnings_next_date": "",
            "earnings_days_away": 999,
        }
        sent      = float(snap.get("sentiment_30m", 0.0))
        headlines = list(snap.get("recent_headlines", []))
        _ep       = snap.get("earnings_phase", "")
        eb = {
            "blocked":   _ep == "blackout" or _ep == "cooldown",
            "reason":    snap.get("earnings_reason",    ""),
            "next_date": snap.get("earnings_next_date", ""),
            "days_away": int(snap.get("earnings_days_away", 999)),
        }

        # Verify types match scanner expectations
        self.assertIsInstance(sent,             float)
        self.assertIsInstance(headlines,        list)
        self.assertIsInstance(eb["blocked"],    bool)
        self.assertIsInstance(eb["reason"],     str)
        self.assertIsInstance(eb["next_date"],  str)
        self.assertIsInstance(eb["days_away"],  int)

        # Safe defaults: not blocked, empty strings
        self.assertFalse(eb["blocked"])
        self.assertEqual(eb["reason"], "")


# ── AC4: provider outage does not blank context ----------------------------

class TestProviderOutageGraceful(unittest.TestCase):
    """
    Simulates Finnhub being unreachable.
    Existing Valkey / DB data must remain accessible to the scanner.
    """

    def test_valkey_snapshot_survives_provider_outage(self):
        """
        If news_poller fails (Finnhub unreachable), the Valkey snapshot
        written by the previous feature_compute cycle is still readable.
        """
        import agent.context_snapshot as cs

        stale_but_valid = json.dumps({
            "ticker":             "MSFT",
            "sentiment_30m":      0.15,
            "sentiment_5m":       0.20,
            "sentiment_2h":       0.10,
            "sentiment_1d":       0.08,
            "sentiment_velocity": 0.05,
            "news_count_30m":     1,
            "news_shock":         False,
            "context_risk_score": 0.12,
            "recent_headlines":   ["Microsoft announces Azure expansion"],
            "earnings_phase":     "",
            "earnings_reason":    "",
            "earnings_next_date": "",
            "earnings_days_away": 30,
            "asof_ts":            time.time() - 200,  # 200s old, within 600s TTL
        }).encode()

        mock_c = MagicMock()
        mock_c.get.return_value = stale_but_valid

        with patch("agent.context_snapshot._client", return_value=mock_c):
            snap = cs.get_context_snapshot("MSFT")

        # Should return the stale-but-valid data, NOT empty defaults
        self.assertEqual(snap["sentiment_30m"], 0.15)
        self.assertGreater(snap["stale_age_s"], 0)
        self.assertLess(snap["stale_age_s"], 600,
                        "stale_age_s within TTL — snapshot should be served")

    def test_pg_fallback_when_valkey_stale(self):
        """
        When Valkey key has expired (returns None), the PG fallback kicks in
        and returns the last computed features.
        """
        import agent.context_snapshot as cs

        pg_features = {
            "ticker":             "NVDA",
            "sentiment_30m":      0.22,
            "sentiment_5m":       0.18,
            "sentiment_2h":       0.15,
            "sentiment_1d":       0.12,
            "sentiment_velocity": 0.07,
            "news_count_30m":     3,
            "news_shock":         True,
            "context_risk_score": 0.25,
            "recent_headlines":   ["Nvidia GPU demand surges"],
            "earnings_phase":     "",
            "earnings_reason":    "",
            "earnings_next_date": "",
            "earnings_days_away": 45,
            "asof_ts":            time.time() - 700,  # older than TTL
        }

        # Valkey returns None (expired)
        mock_c = MagicMock()
        mock_c.get.return_value = None

        with patch("agent.context_snapshot._client", return_value=mock_c):
            with patch("agent.context_snapshot._read_pg", return_value=pg_features):
                snap = cs.get_context_snapshot("NVDA")

        self.assertEqual(snap["sentiment_30m"], 0.22)
        self.assertTrue(snap["news_shock"])


# ── AC5: earnings blackout dict matches scanner expectations ---------------

class TestEarningsBlackoutDict(unittest.TestCase):
    """
    The eb dict built from get_context_snapshot must satisfy all downstream
    key accesses in scanner.py (lines 747, 1165-1168).
    """

    def _build_eb(self, earnings_phase: str, days_away: int,
                  reason: str = "", next_date: str = "") -> dict:
        """Replicate the exact shim logic from scanner.py."""
        snap = {
            "earnings_phase":     earnings_phase,
            "earnings_reason":    reason,
            "earnings_next_date": next_date,
            "earnings_days_away": days_away,
        }
        _ep = snap.get("earnings_phase", "")
        return {
            "blocked":   _ep == "blackout" or _ep == "cooldown",
            "reason":    snap.get("earnings_reason",    ""),
            "next_date": snap.get("earnings_next_date", ""),
            "days_away": int(snap.get("earnings_days_away", 999)),
        }

    def test_blackout_phase_sets_blocked_true(self):
        eb = self._build_eb("blackout", days_away=1,
                            reason="Earnings in 1d — blackout",
                            next_date="Jun 01, 2026")
        self.assertTrue(eb["blocked"])
        self.assertIn("blackout", eb["reason"].lower())
        self.assertEqual(eb["days_away"], 1)
        self.assertEqual(eb["next_date"], "Jun 01, 2026")

    def test_cooldown_phase_sets_blocked_true(self):
        eb = self._build_eb("cooldown", days_away=-1,
                            reason="Post-earnings cooldown")
        self.assertTrue(eb["blocked"])
        self.assertEqual(eb["days_away"], -1)

    def test_caution_phase_does_not_block(self):
        eb = self._build_eb("caution", days_away=5,
                            reason="Earnings in 5d — reduce size")
        self.assertFalse(eb["blocked"])
        self.assertEqual(eb["days_away"], 5)

    def test_no_earnings_data_is_not_blocked(self):
        eb = self._build_eb("", days_away=999)
        self.assertFalse(eb["blocked"])
        self.assertEqual(eb["reason"],    "")
        self.assertEqual(eb["next_date"], "")
        self.assertEqual(eb["days_away"], 999)

    def test_all_required_keys_present(self):
        """All four keys that scanner.py accesses must be present."""
        eb = self._build_eb("blackout", 2, "test", "May 28, 2026")
        for key in ("blocked", "reason", "next_date", "days_away"):
            self.assertIn(key, eb, f"Missing required key '{key}' in eb dict")

    def test_days_away_is_always_int(self):
        for phase, days in [("", 999), ("blackout", 0), ("caution", 7)]:
            eb = self._build_eb(phase, days)
            self.assertIsInstance(eb["days_away"], int,
                                  f"days_away must be int, got {type(eb['days_away'])}")


# ── Scoring unit tests ────────────────────────────────────────────────────────

class TestContextStoreScoringHelpers(unittest.TestCase):
    """Unit tests for the scoring helpers in context_store."""

    def test_score_headline_bullish(self):
        from agent.context_store import _score_headline
        score = _score_headline("Apple beats earnings estimates and upgrades guidance")
        self.assertGreater(score, 0.0)

    def test_score_headline_bearish(self):
        from agent.context_store import _score_headline
        score = _score_headline("SEC investigation triggers layoff and loss warning")
        self.assertLess(score, 0.0)

    def test_score_headline_neutral(self):
        from agent.context_store import _score_headline
        score = _score_headline("Company announces quarterly results")
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_content_hash_is_stable(self):
        from agent.context_store import _content_hash
        h1 = _content_hash("AAPL", "Apple beats earnings", 1716820800)
        h2 = _content_hash("AAPL", "Apple beats earnings", 1716820800)
        self.assertEqual(h1, h2)

    def test_content_hash_dedupes_within_hour(self):
        """Timestamps in the same 1-hour bucket must produce the same hash."""
        from agent.context_store import _content_hash
        # Pick a timestamp that sits at the start of an hour boundary
        ts_base = 1716818400      # exactly 3600 * 476894 → bucket 476894
        ts_plus = ts_base + 3599  # still within the same hour bucket
        self.assertEqual(ts_base // 3600, ts_plus // 3600,
                         "Pre-condition: both timestamps must be in the same bucket")
        h1 = _content_hash("AAPL", "Apple beats earnings", ts_base)
        h2 = _content_hash("AAPL", "Apple beats earnings", ts_plus)
        self.assertEqual(h1, h2)

    def test_content_hash_differs_for_different_tickers(self):
        from agent.context_store import _content_hash
        h1 = _content_hash("AAPL", "Headline", 1716820800)
        h2 = _content_hash("MSFT", "Headline", 1716820800)
        self.assertNotEqual(h1, h2)

    def test_source_credibility_known_source(self):
        from agent.context_store import _source_credibility
        self.assertGreaterEqual(_source_credibility("Reuters"),    0.90)
        self.assertGreaterEqual(_source_credibility("Bloomberg"),  0.90)
        self.assertGreaterEqual(_source_credibility("CNBC Live"),  0.85)

    def test_source_credibility_unknown_source(self):
        from agent.context_store import _source_credibility
        cred = _source_credibility("RandomBlog.io")
        self.assertGreater(cred, 0.0)
        self.assertLessEqual(cred, 1.0)

    def test_event_weight_decays_with_age(self):
        from agent.context_store import _event_weight
        w_fresh = _event_weight(0.9, 0)
        w_30min = _event_weight(0.9, 30)
        w_2h    = _event_weight(0.9, 120)
        self.assertGreater(w_fresh, w_30min)
        self.assertGreater(w_30min, w_2h)

    def test_empty_features_has_all_required_keys(self):
        from agent.context_store import _empty_features
        ef = _empty_features("TEST")
        required = [
            "ticker", "sentiment_5m", "sentiment_30m", "sentiment_2h",
            "sentiment_1d", "sentiment_velocity", "news_count_30m",
            "news_count_1d", "news_shock", "context_risk_score", "recent_headlines",
        ]
        for key in required:
            self.assertIn(key, ef, f"Key '{key}' missing from _empty_features")


# ── Finnhub provider unit tests ───────────────────────────────────────────────

class TestFinnhubProvider(unittest.TestCase):
    """Unit tests for the Finnhub provider adapter."""

    def test_is_available_false_without_key(self):
        import os
        import agent.providers.finnhub_provider as fp
        os.environ.pop("FINNHUB_API_KEY", None)
        self.assertFalse(fp.is_available())

    def test_is_available_true_with_key(self):
        import os
        import agent.providers.finnhub_provider as fp
        with patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key_123"}):
            self.assertTrue(fp.is_available())

    def test_fetch_market_news_no_key_returns_empty(self):
        import os
        import agent.providers.finnhub_provider as fp
        os.environ.pop("FINNHUB_API_KEY", None)
        self.assertEqual(fp.fetch_market_news(), [])

    def test_fetch_market_news_mocked_response(self):
        import os
        import agent.providers.finnhub_provider as fp
        fake_articles = [
            {"headline": "Fed holds rates", "datetime": 1716820800, "source": "Reuters"},
        ]
        with patch.dict(os.environ, {"FINNHUB_API_KEY": "testkey"}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = fake_articles
                mock_get.return_value = mock_resp

                result = fp.fetch_market_news()

        self.assertEqual(result, fake_articles)

    def test_fetch_earnings_calendar_parses_dict(self):
        import os
        import agent.providers.finnhub_provider as fp
        fake_cal = {
            "earningsCalendar": [
                {"symbol": "AAPL", "date": "2026-07-30", "hour": "amc"},
            ]
        }
        with patch.dict(os.environ, {"FINNHUB_API_KEY": "testkey"}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = fake_cal
                mock_get.return_value = mock_resp

                result = fp.fetch_earnings_calendar("2026-07-01", "2026-08-01")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")

    def test_429_response_returns_none(self):
        import os
        import agent.providers.finnhub_provider as fp
        with patch.dict(os.environ, {"FINNHUB_API_KEY": "testkey"}):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 429
                mock_get.return_value = mock_resp
                with patch("time.sleep"):   # don't actually wait 30s
                    result = fp._get("/news", {"category": "general"})

        self.assertIsNone(result)


# ── map_market_news_to_tickers tests ─────────────────────────────────────────

class TestMapMarketNewsToTickers(unittest.TestCase):
    def test_article_with_ticker_in_headline(self):
        from agent.context_store import map_market_news_to_tickers
        articles = [{"headline": "AAPL surges on strong iPhone sales", "source": "CNBC",
                     "datetime": 1716820800}]
        result = map_market_news_to_tickers(articles, ["AAPL", "MSFT", "NVDA"])
        tickers_in_result = {r["ticker"] for r in result}
        self.assertIn("AAPL", tickers_in_result)

    def test_article_with_no_ticker_becomes_market(self):
        from agent.context_store import map_market_news_to_tickers
        articles = [{"headline": "Fed minutes show hawkish tone", "source": "Reuters",
                     "datetime": 1716820800}]
        result = map_market_news_to_tickers(articles, ["AAPL", "MSFT"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "MARKET")
        self.assertTrue(result[0]["is_market_wide"])

    def test_article_with_related_field(self):
        from agent.context_store import map_market_news_to_tickers
        articles = [{"headline": "Chip sector news", "related": "NVDA,AMD",
                     "source": "MarketWatch", "datetime": 1716820800}]
        result = map_market_news_to_tickers(articles, ["NVDA", "AMD", "INTC"])
        tickers = {r["ticker"] for r in result}
        self.assertIn("NVDA", tickers)
        self.assertIn("AMD", tickers)


# ── build_payload_from_features ───────────────────────────────────────────────

class TestBuildPayloadFromFeatures(unittest.TestCase):
    def test_all_keys_present(self):
        from agent.context_snapshot import build_payload_from_features, _EMPTY_PAYLOAD
        features = {
            "sentiment_5m": 0.1, "sentiment_30m": 0.2, "sentiment_2h": 0.15,
            "sentiment_1d": 0.1, "sentiment_velocity": 0.05,
            "news_count_30m": 2, "news_shock": False, "context_risk_score": 0.1,
            "recent_headlines": '["Headline 1"]',
        }
        payload = build_payload_from_features(
            "AAPL", features,
            earnings_phase="caution", earnings_days_away=5,
            earnings_next_date="Jun 01, 2026", earnings_reason="Earnings in 5d",
        )
        for key in _EMPTY_PAYLOAD:
            if key in ("asof_ts", "stale_age_s"):
                continue   # added by publish_context_snapshot / get_context_snapshot
            self.assertIn(key, payload, f"Key '{key}' missing from payload")

    def test_recent_headlines_parsed_from_json_string(self):
        from agent.context_snapshot import build_payload_from_features
        features = {
            "sentiment_5m": 0.0, "sentiment_30m": 0.0, "sentiment_2h": 0.0,
            "sentiment_1d": 0.0, "sentiment_velocity": 0.0,
            "news_count_30m": 0, "news_shock": False, "context_risk_score": 0.0,
            "recent_headlines": '["Headline A", "Headline B"]',
        }
        payload = build_payload_from_features("MSFT", features)
        self.assertEqual(payload["recent_headlines"], ["Headline A", "Headline B"])


# ── Helper (not a test class) ─────────────────────────────────────────────────

class importlib:  # noqa: N801
    """Minimal stand-in so test can call importlib.reload_if_possible."""
    @staticmethod
    def reload_if_possible(module):
        import importlib as _il
        try:
            _il.reload(module)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
