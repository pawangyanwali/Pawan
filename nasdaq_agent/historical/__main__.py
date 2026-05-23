"""
Historical data backfill CLI.

Usage examples
--------------
# Full 2-year backfill for all tickers (start and leave running over the weekend):
  python -m nasdaq_agent.historical

# Custom years / specific tickers:
  python -m nasdaq_agent.historical --years 1 --tickers AAPL,MSFT,NVDA

# Resume after a crash:
  python -m nasdaq_agent.historical --resume

# Only resample 1min → derived (skip API fetch):
  python -m nasdaq_agent.historical --resample-only

# Only fetch / refresh 1day bars:
  python -m nasdaq_agent.historical --daily-only

# Show current progress and row counts, then exit:
  python -m nasdaq_agent.historical --status

# Reset all progress and start fresh:
  python -m nasdaq_agent.historical --reset
"""

import argparse
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

# ── Bootstrap: load .env and ensure the package root is importable ────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Try to load .env from the standard deployment location.
# Skip silently if the file is not readable (service user owns it).
_ENV_PATHS = [
    _REPO_ROOT / ".env",
    Path("/opt/nasdaq-agent/nasdaq_agent/.env"),
]
try:
    from dotenv import load_dotenv
    for _p in _ENV_PATHS:
        if _p.exists() and os.access(_p, os.R_OK):
            load_dotenv(_p)
            break
except (ImportError, Exception):
    pass  # env vars may already be exported in the shell


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    # File — always written regardless of verbose flag
    log_dir = Path("/opt/nasdaq-agent/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "backfill.log"
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(ch)
    root.addHandler(fh)

    # Silence noisy sub-loggers unless verbose
    if not verbose:
        for noisy in ("urllib3", "httpx", "httpcore", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("backfill.main").info("Logging to %s", log_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Schwab historical price backfill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--years",         type=int,   default=2,
                   help="Years of history to fetch (default: 2)")
    p.add_argument("--tickers",       type=str,   default=None,
                   help="Comma-separated tickers (default: full universe)")
    p.add_argument("--rate",          type=float, default=1.0,
                   help="Seconds between API calls (default: 1.0 = 1 req/s)")
    p.add_argument("--resume",        action="store_true",
                   help="Resume from saved checkpoint (default when checkpoint exists)")
    p.add_argument("--reset",         action="store_true",
                   help="Delete checkpoint and start from scratch")
    p.add_argument("--resample-only", action="store_true",
                   help="Skip API fetch; only resample stored 1min → derived intervals")
    p.add_argument("--daily-only",    action="store_true",
                   help="Only fetch/refresh 1day bars (fast)")
    p.add_argument("--status",        action="store_true",
                   help="Print progress summary and DB row counts, then exit")
    p.add_argument("--verbose",       action="store_true",
                   help="Debug logging")
    return p.parse_args()


def _print_status() -> None:
    from historical import progress, store
    prog = progress.summary()
    counts = store.row_counts()
    print("\n=== Backfill Progress ===")
    print(f"  Started      : {prog['started_at']}")
    print(f"  Chunks done  : {prog['chunks_done']:,}")
    print(f"  Tickers resampled : {prog['resampled']}")
    print(f"  Daily done   : {prog['daily_done']}")
    print("\n=== DB Row Counts ===")
    for iv, n in counts.items():
        bar = "█" * min(40, n // 50000)
        print(f"  {iv:>6}  {n:>12,}  {bar}")
    total = sum(counts.values())
    print(f"\n  TOTAL        : {total:,} rows")
    print()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger("backfill.main")

    # Late imports — after sys.path and env are set up
    from historical import fetcher, progress, store

    # ── Load Schwab Market Data token from disk ──────────────────────────────
    # The main service calls this at startup. The backfill is a separate process
    # so it must load the stored token explicitly before making any API calls.
    try:
        from agent.broker.schwab_auth import load_stored_md_tokens
        ok = load_stored_md_tokens()
        if not ok:
            log.warning(
                "Schwab MD token could not be loaded from disk — auth will likely fail.\n"
                "  Ensure nasdaq-agent service is running and has completed OAuth for Market Data."
            )
    except Exception as _auth_exc:
        log.warning("Could not load MD token: %s", _auth_exc)

    # ── Status only ──────────────────────────────────────────────────────────
    if args.status:
        progress.load()
        store.init_tables()
        _print_status()
        return

    # ── Reset ────────────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("This will delete all backfill progress. Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return
        progress.reset()
        log.info("Progress reset. Run again without --reset to start fresh.")
        return

    # ── Load checkpoint ──────────────────────────────────────────────────────
    progress.load()
    if args.reset:   # already handled above but guard
        progress.reset()

    # ── Init DB tables ───────────────────────────────────────────────────────
    store.init_tables()

    # ── Tickers ──────────────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        from config import NASDAQ_TICKERS
        tickers = list(NASDAQ_TICKERS)
    log.info("Tickers: %d", len(tickers))

    # ── Time estimate ────────────────────────────────────────────────────────
    if not args.resample_only and not args.daily_only:
        est = fetcher.estimate_time(tickers, args.years, args.rate)
        log.info("Time estimate: %s", est)

    # ── Run ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        fetcher.run(
            tickers       = tickers,
            years         = args.years,
            rate_s        = args.rate,
            resample_only = args.resample_only,
            daily_only    = args.daily_only,
        )
    except KeyboardInterrupt:
        log.info("\nInterrupted. Progress saved — resume with: python -m nasdaq_agent.historical --resume")
        sys.exit(0)

    elapsed = time.time() - t0
    h, rem  = divmod(int(elapsed), 3600)
    log.info("Done in %dh %dm %ds", h, rem // 60, rem % 60)
    _print_status()


if __name__ == "__main__":
    main()
