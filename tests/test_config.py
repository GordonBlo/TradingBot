from collections.abc import Iterator
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from src.config.settings import ExecutionEnvironment, Settings, SettingsError
from src.exchange.market_stream import MarketStream
from src.exchange.public_market_client import PublicMarketDataClient
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.risk.risk_manager import RiskManager


@pytest.fixture
def project_tmp_path() -> Iterator[Path]:
    """Provide a self-cleaning temp directory inside the writable project."""

    temp_root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pytest-", dir=temp_root) as directory:
        yield Path(directory)


def test_configuration_loading(project_tmp_path: Path) -> None:
    env_file = project_tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "BINANCE_API_KEY=test-key",
                "BINANCE_SECRET_KEY=test-secret",
                "MARKET_DATA_SOURCE=binance_public",
                "EXECUTION_ENV=testnet",
                "TRADING_SYMBOL=btcusdc",
                "CANDLE_INTERVAL=15m",
                "HISTORICAL_CANDLE_LIMIT=250",
                "BACKTEST_INITIAL_CAPITAL_USDC=1500",
                "BACKTEST_FEE_BPS=8",
                "BACKTEST_SLIPPAGE_BPS=3",
                "BACKTEST_AMBIGUOUS_BAR_POLICY=stop_first",
                "MAX_CAPITAL_USDC=75.50",
                "MAX_RISK_PER_TRADE_PERCENT=0.75",
                "MAX_DAILY_LOSS_PERCENT=3",
            )
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(env_file, environ={})

    assert settings.binance_api_key == "test-key"
    assert settings.binance_secret_key == "test-secret"
    assert settings.market_data_source is MarketDataSource.BINANCE_PUBLIC
    assert settings.execution_environment is ExecutionEnvironment.TESTNET
    assert settings.trading_symbol == "BTCUSDC"
    assert settings.historical_candle_limit == 250
    assert settings.backtest_initial_capital_usdc == Decimal("1500")
    assert settings.backtest_fee_bps == Decimal("8")
    assert settings.backtest_slippage_bps == Decimal("3")
    assert settings.backtest_ambiguous_bar_policy == "STOP_FIRST"
    assert settings.max_capital_usdc == Decimal("75.50")
    assert settings.max_risk_per_trade_percent == Decimal("0.75")
    assert settings.max_daily_loss_percent == Decimal("3")


def test_process_environment_overrides_dotenv(project_tmp_path: Path) -> None:
    env_file = project_tmp_path / ".env"
    env_file.write_text("HISTORICAL_CANDLE_LIMIT=100\n", encoding="utf-8")

    settings = Settings.from_env(
        env_file, environ={"HISTORICAL_CANDLE_LIMIT": "250"}
    )

    assert settings.historical_candle_limit == 250


@pytest.mark.parametrize(
    ("variable", "invalid_value"),
    (
        ("MAX_CAPITAL_USDC", "0"),
        ("MAX_RISK_PER_TRADE_PERCENT", "-0.1"),
        ("MAX_RISK_PER_TRADE_PERCENT", "101"),
        ("MAX_DAILY_LOSS_PERCENT", "0"),
        ("MAX_DAILY_LOSS_PERCENT", "not-a-number"),
    ),
)
def test_invalid_risk_configuration(variable: str, invalid_value: str) -> None:
    environment = {variable: invalid_value}

    with pytest.raises(SettingsError):
        Settings.from_env(env_file=None, environ=environment)


def test_default_configuration_uses_testnet() -> None:
    settings = Settings.from_env(env_file=None, environ={})

    assert settings.market_data_source is MarketDataSource.BINANCE_PUBLIC
    assert settings.execution_environment is ExecutionEnvironment.TESTNET
    assert settings.execution_environment_name == "TESTNET"
    assert settings.trading_symbol == "BTCUSDC"
    assert settings.candle_interval == "15m"
    assert settings.historical_candle_limit == 300
    assert settings.has_credentials is False
    baseline = settings.trend_momentum_config()
    assert baseline.fast_ema_period == 20
    assert baseline.slow_ema_period == 50
    assert baseline.rsi_min == Decimal("52")
    assert baseline.rsi_max == Decimal("68")
    assert baseline.minimum_volume_ratio == Decimal("0.80")
    assert baseline.atr_stop_multiplier == Decimal("2.0")
    assert baseline.reward_risk_ratio == Decimal("2.0")
    assert baseline.risk_per_trade_percent == Decimal("0.50")
    assert baseline.maximum_bars_in_position == 96
    assert baseline.cooldown_bars == 4
    assert settings.research_min_trades_warning == 30


def test_v3_strategy_environment_values_are_typed() -> None:
    settings = Settings.from_env(
        env_file=None,
        environ={
            "BASELINE_FAST_EMA_PERIOD": "10",
            "BASELINE_SLOW_EMA_PERIOD": "30",
            "BASELINE_RSI_MIN": "50.5",
            "BASELINE_RSI_MAX": "70.5",
            "BASELINE_MIN_VOLUME_RATIO": "0.9",
            "BASELINE_MAX_BARS_IN_POSITION": "48",
            "BASELINE_COOLDOWN_BARS": "3",
            "RESEARCH_MIN_TRADES_WARNING": "20",
        },
    )

    baseline = settings.trend_momentum_config()
    assert (baseline.fast_ema_period, baseline.slow_ema_period) == (10, 30)
    assert (baseline.rsi_min, baseline.rsi_max) == (
        Decimal("50.5"),
        Decimal("70.5"),
    )
    assert baseline.minimum_volume_ratio == Decimal("0.9")
    assert baseline.maximum_bars_in_position == 48
    assert baseline.cooldown_bars == 3
    assert settings.research_min_trades_warning == 20


@pytest.mark.parametrize("invalid_value", ("0", "1001", "not-an-integer"))
def test_invalid_historical_candle_limit(invalid_value: str) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env(
            env_file=None,
            environ={"HISTORICAL_CANDLE_LIMIT": invalid_value},
        )


@pytest.mark.parametrize(
    ("variable", "invalid_value"),
    (
        ("BACKTEST_INITIAL_CAPITAL_USDC", "0"),
        ("BACKTEST_FEE_BPS", "-1"),
        ("BACKTEST_FEE_BPS", "10001"),
        ("BACKTEST_SLIPPAGE_BPS", "10000"),
        ("BACKTEST_AMBIGUOUS_BAR_POLICY", "TARGET_FIRST"),
    ),
)
def test_invalid_backtest_configuration(variable: str, invalid_value: str) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env(env_file=None, environ={variable: invalid_value})


def test_candle_model_creation_uses_decimal_values() -> None:
    timestamp = datetime(2026, 8, 17, 20, 30, tzinfo=timezone.utc)

    candle = Candle(
        timestamp=timestamp,
        symbol="btcusdc",
        interval="15m",
        open=Decimal("64000.10"),
        high=Decimal("64100.25"),
        low=Decimal("63950.00"),
        close=Decimal("64050.25"),
        volume=Decimal("1.23456789"),
        is_closed=True,
    )

    assert candle.timestamp == timestamp
    assert candle.symbol == "BTCUSDC"
    assert candle.close == Decimal("64050.25")
    assert candle.volume == Decimal("1.23456789")
    assert candle.is_closed is True


def test_risk_manager_calculations() -> None:
    manager = RiskManager(
        max_capital_usdc=Decimal("50"),
        max_risk_per_trade_percent=Decimal("0.5"),
        max_daily_loss_percent=Decimal("2"),
    )

    assert manager.calculate_max_risk_amount(Decimal("50")) == Decimal("0.25")
    assert manager.calculate_max_daily_loss(Decimal("50")) == Decimal("1")
    assert manager.remaining_daily_loss_allowance(
        Decimal("50"), Decimal("-0.4")
    ) == Decimal("0.6")
    assert manager.is_potential_loss_allowed(Decimal("0.25"), Decimal("50"))
    assert not manager.is_potential_loss_allowed(Decimal("0.26"), Decimal("50"))


def test_public_binance_wrapped_ticker_response_is_read_as_decimal() -> None:
    class FakeResponse:
        def data(self) -> SimpleNamespace:
            ticker = SimpleNamespace(symbol="BTCUSDC", price="64050.25000000")
            return SimpleNamespace(actual_instance=ticker)

    class FakeRestApi:
        def ticker_price(self, *, symbol: str) -> FakeResponse:
            assert symbol == "BTCUSDC"
            return FakeResponse()

    client = PublicMarketDataClient(rest_api=FakeRestApi())

    assert client.get_current_price() == Decimal("64050.25000000")


def test_market_stream_parses_binance_kline_into_candle() -> None:
    message = """{
        "e": "kline",
        "k": {
            "t": 1786998600000,
            "s": "BTCUSDC",
            "i": "15m",
            "o": "64000.10",
            "h": "64100.25",
            "l": "63950.00",
            "c": "64050.25",
            "v": "1.23456789",
            "x": true
        }
    }"""

    candle = MarketStream.parse_message(message)

    assert candle is not None
    assert candle.symbol == "BTCUSDC"
    assert candle.interval == "15m"
    assert candle.close == Decimal("64050.25")
    assert candle.volume == Decimal("1.23456789")
    assert candle.is_closed is True
