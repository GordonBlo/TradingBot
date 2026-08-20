"""Application startup and lifecycle orchestration."""

from __future__ import annotations

import asyncio
import signal

from src.config.settings import SettingsError, load_settings
from src.exchange.historical_data import HistoricalDataError, HistoricalDataService
from src.exchange.market_stream import MarketStream
from src.exchange.public_market_client import (
    PublicMarketDataClient,
    PublicMarketDataError,
)
from src.market.candle_history import CandleHistory
from src.models.candle import Candle
from src.analysis.market_analyzer import MarketAnalyzer
from src.risk.risk_manager import RiskManager
from src.utils.logger import configure_logging, get_logger


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for signal_name in ("SIGINT", "SIGTERM"):
        received_signal = getattr(signal, signal_name, None)
        if received_signal is None:
            continue
        try:
            loop.add_signal_handler(received_signal, request_stop)
        except (NotImplementedError, RuntimeError):
            try:
                signal.signal(
                    received_signal,
                    lambda _signum, _frame: loop.call_soon_threadsafe(request_stop),
                )
            except (ValueError, OSError):
                pass


async def main() -> int:
    """Start the V1.1 read-only analytics services and run until interrupted."""

    try:
        settings = load_settings()
    except SettingsError as exc:
        configure_logging()
        get_logger(__name__).error("Configuration error: %s", exc)
        return 2

    configure_logging(
        sensitive_values=(settings.binance_api_key, settings.binance_secret_key)
    )
    logger = get_logger(__name__)
    logger.info("Trading Bot V1.1 starting")
    logger.info("Market data source: %s", settings.market_data_source.display_name)
    logger.info("Market data REST: REAL/PUBLIC")
    logger.info("Market data WebSocket: REAL/PUBLIC")
    logger.info(
        "Execution environment: %s", settings.execution_environment_name
    )
    logger.info("Trading capability: DISABLED")

    # Instantiate now so invalid risk limits fail before any network work begins.
    RiskManager.from_settings(settings)
    client = PublicMarketDataClient(settings.trading_symbol)
    stream = MarketStream(settings.trading_symbol, settings.candle_interval)
    if (
        client.source is not settings.market_data_source
        or stream.source is not settings.market_data_source
    ):
        logger.error("Market-data source mismatch; startup aborted")
        return 2

    try:
        await asyncio.to_thread(client.verify_connection)
        logger.info("Connected to Binance PUBLIC REST market data")
        price = await asyncio.to_thread(
            client.get_current_price, settings.trading_symbol
        )
        logger.info("%s real market price: %s", settings.trading_symbol, price)
    except PublicMarketDataError as exc:
        logger.error("Public Binance market data unavailable: %s", exc)
        return 1

    historical_service = HistoricalDataService(client)
    try:
        historical_candles = await asyncio.to_thread(
            historical_service.load_recent_candles,
            settings.trading_symbol,
            settings.candle_interval,
            limit=settings.historical_candle_limit,
        )
    except (PublicMarketDataError, HistoricalDataError, ValueError) as exc:
        logger.error("Failed to load historical market data: %s", exc)
        return 1

    history = CandleHistory(
        settings.trading_symbol,
        settings.candle_interval,
        max_size=settings.historical_candle_limit,
        candles=historical_candles,
    )
    analyzer = MarketAnalyzer()
    initial_snapshot = analyzer.analyze(history)
    logger.info(
        "Loaded %d real %s %s closed candles",
        len(history),
        settings.trading_symbol,
        settings.candle_interval,
    )
    logger.info(
        "Historical range: %s to %s",
        history.candles[0].timestamp.isoformat(),
        history.candles[-1].timestamp.isoformat(),
    )
    logger.info("Indicator engine initialized")
    if initial_snapshot is not None:
        logger.info(
            "Latest historical market snapshot\n%s",
            analyzer.format_snapshot(initial_snapshot),
        )

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    def handle_live_candle(candle: Candle) -> None:
        updated_snapshot = analyzer.update_with_closed_candle(history, candle)
        if updated_snapshot is None:
            return
        logger.info(
            "%s %s candle closed; market snapshot updated\n%s",
            candle.symbol,
            candle.interval,
            analyzer.format_snapshot(updated_snapshot),
        )

    stream.subscribe(handle_live_candle)

    logger.info("Starting read-only market stream; press Ctrl+C to stop")
    try:
        await stream.run(stop_event)
    finally:
        stop_event.set()
        logger.info("Trading Bot V1.1 shut down cleanly")
    return 0
