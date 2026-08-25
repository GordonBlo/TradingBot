"""Run a bounded, unauthenticated BTCUSDC 15m shadow runtime smoke session."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.config.settings import Settings
from src.exchange.historical_data import HistoricalDataService
from src.exchange.market_stream import MarketStream
from src.exchange.public_market_client import PublicMarketDataClient
from src.runtime.shadow import LiveShadowService, ShadowRuntime
from src.runtime.state import RuntimeStateStore, ShadowDecisionJournal
from src.strategy.base import BaseStrategy
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.v4_breakout import V4BreakoutStrategy
from src.strategy.v5_mean_reversion import V5MeanReversionStrategy
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-seconds",
        type=int,
        required=True,
        help="Bounded validation duration from 5 through 300 seconds.",
    )
    parser.add_argument(
        "--strategy",
        choices=("baseline", "v4", "v5", "v6"),
        default="baseline",
    )
    parser.add_argument("--run-id", default="live-shadow-foundation-smoke")
    parser.add_argument("--output-root", type=Path, default=Path("data/runtime/shadow"))
    parser.add_argument("--stale-seconds", type=float, default=30.0)
    return parser


def _strategy(name: str, settings: Settings) -> BaseStrategy:
    config = settings.trend_momentum_config()
    if name == "baseline":
        return TrendMomentumBaselineStrategy(config)
    if name == "v4":
        return V4BreakoutStrategy(config)
    if name == "v5":
        return V5MeanReversionStrategy(config)
    return V6MTFContinuationStrategy(config)


async def _run(args: argparse.Namespace) -> tuple[dict, Path]:
    if not 5 <= args.smoke_seconds <= 300:
        raise ValueError("smoke duration must be between 5 and 300 seconds")
    if not 1 <= args.stale_seconds <= 900:
        raise ValueError("stale threshold must be between 1 and 900 seconds")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("run ID may contain only letters, numbers, dot, underscore, and dash")

    settings = Settings(binance_api_key="", binance_secret_key="")
    client = PublicMarketDataClient(settings.trading_symbol)
    if client.uses_authentication:
        raise RuntimeError("shadow runtime refuses authenticated market data")
    historical = HistoricalDataService(client)
    stream = MarketStream(settings.trading_symbol, settings.candle_interval)
    strategy = _strategy(args.strategy, settings)
    run_root = args.output_root / args.run_id
    state_store = RuntimeStateStore(run_root / "runtime_state.json")
    journal = ShadowDecisionJournal(run_root / "shadow_decisions.jsonl")
    decisions_before = journal.sequence
    runtime = ShadowRuntime(
        strategy=strategy,
        strategy_config=settings.trend_momentum_config(),
        run_identity=args.run_id,
        state_store=state_store,
        journal=journal,
        history_limit=settings.historical_candle_limit,
        stale_after=timedelta(seconds=args.stale_seconds),
        equity=settings.backtest_initial_capital_usdc,
        cash_usdc=settings.backtest_initial_capital_usdc,
        fee_bps=settings.backtest_fee_bps,
    )

    async def load_history():
        return await asyncio.to_thread(
            historical.load_recent_candles,
            settings.trading_symbol,
            settings.candle_interval,
            limit=settings.historical_candle_limit,
            closed_only=True,
        )

    service = LiveShadowService(
        runtime=runtime,
        stream=stream,
        load_closed_history=load_history,
    )
    started = datetime.now(UTC)
    try:
        await service.run(duration_seconds=args.smoke_seconds)
    except Exception:
        runtime.stop(at=datetime.now(UTC))
        raise
    ended = datetime.now(UTC)
    new_records = tuple(journal.records())[decisions_before:]
    summary = {
        "version": "LIVE_SHADOW_FOUNDATION_1",
        "run_identity": args.run_id,
        "strategy": {
            "name": runtime.strategy_name,
            "version": runtime.strategy_version,
        },
        "symbol": runtime.SYMBOL,
        "interval": runtime.INTERVAL,
        "mode": "SHADOW",
        "source": "BINANCE_PUBLIC_SPOT",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "duration_seconds": args.smoke_seconds,
        "final_state": runtime.state.value,
        "orders_enabled": False,
        "authentication_used": False,
        "new_shadow_decisions": len(new_records),
        "new_shadow_actions": [record["decision"]["action"] for record in new_records],
        "counters": asdict(runtime.snapshot.counters),
        "state_path": str(state_store.path),
        "journal_path": str(journal.path),
    }
    summary_path = run_root / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary, summary_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, summary_path = asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"LIVE SHADOW SMOKE FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SUMMARY: {summary_path}")
    print("LIVE SHADOW SMOKE COMPLETE — ZERO ORDERS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
