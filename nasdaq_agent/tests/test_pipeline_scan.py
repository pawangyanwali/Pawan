import sys
import time
import types

from agent import pipeline as pipeline_mod
from agent.pipeline import ScanPipeline


def _install_fake_config(monkeypatch, values):
    class FakeConfig:
        def get(self, key, default=None):
            return values.get(key, default)

    monkeypatch.setitem(
        sys.modules,
        "agent.config_manager",
        types.SimpleNamespace(config=FakeConfig()),
    )


def test_scan_records_slow_ticker_metrics(monkeypatch):
    fake_scanner = types.ModuleType("agent.scanner")

    class StockSignal:
        def __init__(self, ticker, score):
            self.ticker = ticker
            self.score = score

    def analyse_ticker(ticker, **kwargs):
        if ticker == "SLOW":
            time.sleep(0.02)
        return StockSignal(ticker, 1.0 if ticker == "SLOW" else 0.1)

    fake_scanner.StockSignal = StockSignal
    fake_scanner.analyse_ticker = analyse_ticker

    monkeypatch.setitem(sys.modules, "agent.scanner", fake_scanner)
    monkeypatch.setenv("NASDAQ_SCAN_SLOW_TICKER_S", "0.005")
    _install_fake_config(
        monkeypatch,
        {
            "scanner.ticker_timeout_s": 45,
            "scanner.cycle_budget_s": 20,
            "scanner.slow_ticker_cooldown_s": 300,
        },
    )

    pipeline = ScanPipeline(n_workers=2)
    results = pipeline.scan(["FAST", "SLOW"], {}, {}, {}, {})
    metrics = pipeline.get_metrics()

    assert [sig.ticker for sig in results] == ["SLOW", "FAST"]
    assert metrics["slow_tickers_last_cycle"][0]["ticker"] == "SLOW"
    assert metrics["slow_tickers_last_cycle"][0]["elapsed_s"] >= 0.005


def test_scan_cycle_budget_marks_slow_ticker_for_cooldown(monkeypatch):
    fake_scanner = types.ModuleType("agent.scanner")

    class StockSignal:
        def __init__(self, ticker, score):
            self.ticker = ticker
            self.score = score

    def analyse_ticker(ticker, **kwargs):
        if ticker == "SLOW":
            time.sleep(1.5)
        return StockSignal(ticker, 1.0 if ticker == "SLOW" else 0.1)

    fake_scanner.StockSignal = StockSignal
    fake_scanner.analyse_ticker = analyse_ticker

    monkeypatch.setitem(sys.modules, "agent.scanner", fake_scanner)
    _install_fake_config(
        monkeypatch,
        {
            "scanner.ticker_timeout_s": 45,
            "scanner.cycle_budget_s": 1,
            "scanner.slow_ticker_cooldown_s": 60,
        },
    )
    with pipeline_mod._slow_skip_lock:
        pipeline_mod._slow_skip_until.clear()

    pipeline = ScanPipeline(n_workers=2)
    started = time.perf_counter()
    results = pipeline.scan(["FAST", "SLOW"], {}, {}, {}, {})
    elapsed = time.perf_counter() - started
    metrics = pipeline.get_metrics()

    assert elapsed < 1.4
    assert [sig.ticker for sig in results] == ["FAST"]
    assert metrics["timeouts_last_cycle"] == 1

    results = pipeline.scan(["FAST", "SLOW"], {}, {}, {}, {})
    metrics = pipeline.get_metrics()

    assert [sig.ticker for sig in results] == ["FAST"]
    assert metrics["skipped_slow_cooldown"] == 1
