import sys
import time
import types

from agent.pipeline import ScanPipeline


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

    pipeline = ScanPipeline(n_workers=2)
    results = pipeline.scan(["FAST", "SLOW"], {}, {}, {}, {})
    metrics = pipeline.get_metrics()

    assert [sig.ticker for sig in results] == ["SLOW", "FAST"]
    assert metrics["slow_tickers_last_cycle"][0]["ticker"] == "SLOW"
    assert metrics["slow_tickers_last_cycle"][0]["elapsed_s"] >= 0.005
