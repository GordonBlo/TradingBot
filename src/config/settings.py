"""Centralized, environment-backed application settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from dotenv import dotenv_values

from src.market.intervals import SUPPORTED_INTERVALS
from src.models.market_data_source import MarketDataSource
from src.hypotheses.models import HypothesisSuiteConfig
from src.strategy.models import TrendMomentumConfig


class SettingsError(ValueError):
    """Raised when application configuration is missing or invalid."""


class ExecutionEnvironment(str, Enum):
    """Supported environment for future authenticated execution functionality."""

    TESTNET = "testnet"


def _parse_market_data_source(value: str) -> MarketDataSource:
    try:
        return MarketDataSource(value.strip().lower())
    except ValueError as exc:
        raise SettingsError(
            "MARKET_DATA_SOURCE must be binance_public in V1.1."
        ) from exc


def _parse_execution_environment(value: str) -> ExecutionEnvironment:
    try:
        return ExecutionEnvironment(value.strip().lower())
    except ValueError as exc:
        raise SettingsError(
            "EXECUTION_ENV must be testnet in V1.1; live execution is disabled."
        ) from exc


def _parse_decimal(name: str, value: str) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise SettingsError(f"{name} must be a valid decimal number.") from exc
    if not parsed.is_finite():
        raise SettingsError(f"{name} must be finite.")
    return parsed


def _parse_int(name: str, value: str) -> int:
    try:
        parsed = int(value.strip())
    except (ValueError, AttributeError) as exc:
        raise SettingsError(f"{name} must be a valid integer.") from exc
    return parsed


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings used by every application component."""

    binance_api_key: str = field(default="", repr=False)
    binance_secret_key: str = field(default="", repr=False)
    market_data_source: MarketDataSource = MarketDataSource.BINANCE_PUBLIC
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.TESTNET
    trading_symbol: str = "BTCUSDC"
    candle_interval: str = "15m"
    historical_candle_limit: int = 300
    backtest_initial_capital_usdc: Decimal = Decimal("1000")
    backtest_fee_bps: Decimal = Decimal("10")
    backtest_slippage_bps: Decimal = Decimal("2")
    backtest_ambiguous_bar_policy: str = "STOP_FIRST"
    max_capital_usdc: Decimal = Decimal("50")
    max_risk_per_trade_percent: Decimal = Decimal("0.5")
    max_daily_loss_percent: Decimal = Decimal("2.0")
    baseline_fast_ema_period: int = 20
    baseline_slow_ema_period: int = 50
    baseline_rsi_period: int = 14
    baseline_rsi_min: Decimal = Decimal("52")
    baseline_rsi_max: Decimal = Decimal("68")
    baseline_volume_sma_period: int = 20
    baseline_min_volume_ratio: Decimal = Decimal("0.80")
    baseline_atr_period: int = 14
    baseline_atr_stop_multiplier: Decimal = Decimal("2.0")
    baseline_reward_risk_ratio: Decimal = Decimal("2.0")
    baseline_risk_per_trade_percent: Decimal = Decimal("0.50")
    baseline_max_bars_in_position: int = 96
    baseline_cooldown_bars: int = 4
    research_min_trades_warning: int = 30
    h1_atr_percentile_lookback: int = 100
    h1_max_percentile: Decimal = Decimal("75")
    h2_ema_spread_lookback: int = 100
    h2_max_percentile: Decimal = Decimal("75")
    h3_confirmation_window_bars: int = 8
    h4_reward_risk_ratio: Decimal = Decimal("1.0")
    hypothesis_trade_count_ratio_warning: Decimal = Decimal("0.40")
    multiregime_window_days: int = 90
    multiregime_min_window_days: int = 60
    multiregime_vol_lookback: int = 200
    multiregime_vol_low_percentile: Decimal = Decimal("33")
    multiregime_vol_high_percentile: Decimal = Decimal("67")
    h5_lookback: int = 100
    h5_min_percentile: Decimal = Decimal("25")
    h5_max_percentile: Decimal = Decimal("75")
    h6_confirmation_bars: int = 1
    h7_cost_opportunity_multiplier: Decimal = Decimal("5")

    def __post_init__(self) -> None:
        object.__setattr__(self, "binance_api_key", self.binance_api_key.strip())
        object.__setattr__(self, "binance_secret_key", self.binance_secret_key.strip())
        object.__setattr__(self, "trading_symbol", self.trading_symbol.strip().upper())
        object.__setattr__(self, "candle_interval", self.candle_interval.strip())
        object.__setattr__(
            self,
            "backtest_ambiguous_bar_policy",
            self.backtest_ambiguous_bar_policy.strip().upper(),
        )

        if not isinstance(self.market_data_source, MarketDataSource):
            object.__setattr__(
                self,
                "market_data_source",
                _parse_market_data_source(str(self.market_data_source)),
            )
        if not isinstance(self.execution_environment, ExecutionEnvironment):
            object.__setattr__(
                self,
                "execution_environment",
                _parse_execution_environment(str(self.execution_environment)),
            )

        if bool(self.binance_api_key) != bool(self.binance_secret_key):
            raise SettingsError(
                "BINANCE_API_KEY and BINANCE_SECRET_KEY must either both be set or both be empty."
            )
        if self.trading_symbol != "BTCUSDC":
            raise SettingsError("V1 supports Spot BTC/USDC only (TRADING_SYMBOL=BTCUSDC).")
        if self.candle_interval not in SUPPORTED_INTERVALS:
            raise SettingsError("CANDLE_INTERVAL is not a supported Binance kline interval.")
        if not 1 <= self.historical_candle_limit <= 1_000:
            raise SettingsError("HISTORICAL_CANDLE_LIMIT must be between 1 and 1000.")
        if self.backtest_initial_capital_usdc <= 0:
            raise SettingsError(
                "BACKTEST_INITIAL_CAPITAL_USDC must be greater than zero."
            )
        if not Decimal("0") <= self.backtest_fee_bps <= Decimal("10000"):
            raise SettingsError("BACKTEST_FEE_BPS must be between 0 and 10000.")
        if not Decimal("0") <= self.backtest_slippage_bps < Decimal("10000"):
            raise SettingsError(
                "BACKTEST_SLIPPAGE_BPS must be at least 0 and below 10000."
            )
        if self.backtest_ambiguous_bar_policy != "STOP_FIRST":
            raise SettingsError(
                "BACKTEST_AMBIGUOUS_BAR_POLICY must be STOP_FIRST in V2."
            )
        if self.max_capital_usdc <= 0:
            raise SettingsError("MAX_CAPITAL_USDC must be greater than zero.")
        if not Decimal("0") < self.max_risk_per_trade_percent <= Decimal("100"):
            raise SettingsError(
                "MAX_RISK_PER_TRADE_PERCENT must be greater than zero and at most 100."
            )
        if not Decimal("0") < self.max_daily_loss_percent <= Decimal("100"):
            raise SettingsError(
                "MAX_DAILY_LOSS_PERCENT must be greater than zero and at most 100."
            )
        try:
            self.trend_momentum_config()
        except ValueError as exc:
            raise SettingsError(f"Invalid baseline strategy configuration: {exc}") from exc
        if self.research_min_trades_warning < 1:
            raise SettingsError("RESEARCH_MIN_TRADES_WARNING must be at least 1.")
        try:
            self.hypothesis_suite_config()
        except ValueError as exc:
            raise SettingsError(f"Invalid pre-registered V3.2 configuration: {exc}") from exc
        try:
            self.multiregime_config()
        except ValueError as exc:
            raise SettingsError(f"Invalid frozen V3.2.1 configuration: {exc}") from exc
        try:
            self.mechanism_suite_config()
        except ValueError as exc:
            raise SettingsError(f"Invalid frozen V3.2.2 configuration: {exc}") from exc

    @property
    def has_credentials(self) -> bool:
        """Return whether a complete API credential pair is configured."""

        return bool(self.binance_api_key and self.binance_secret_key)

    @property
    def execution_environment_name(self) -> str:
        """Return a log-safe name for the future execution environment."""

        return self.execution_environment.value.upper()

    def trend_momentum_config(self) -> TrendMomentumConfig:
        """Build the single typed V3 baseline configuration."""

        return TrendMomentumConfig(
            fast_ema_period=self.baseline_fast_ema_period,
            slow_ema_period=self.baseline_slow_ema_period,
            rsi_period=self.baseline_rsi_period,
            rsi_min=self.baseline_rsi_min,
            rsi_max=self.baseline_rsi_max,
            volume_sma_period=self.baseline_volume_sma_period,
            minimum_volume_ratio=self.baseline_min_volume_ratio,
            atr_period=self.baseline_atr_period,
            atr_stop_multiplier=self.baseline_atr_stop_multiplier,
            reward_risk_ratio=self.baseline_reward_risk_ratio,
            risk_per_trade_percent=self.baseline_risk_per_trade_percent,
            maximum_bars_in_position=self.baseline_max_bars_in_position,
            cooldown_bars=self.baseline_cooldown_bars,
            maximum_position_notional_usdc=self.max_capital_usdc,
        )

    def hypothesis_suite_config(self) -> HypothesisSuiteConfig:
        """Build the fixed V3.2 registry; parameter-search overrides are refused."""

        return HypothesisSuiteConfig(
            h1_atr_percentile_lookback=self.h1_atr_percentile_lookback,
            h1_max_percentile=self.h1_max_percentile,
            h2_ema_spread_lookback=self.h2_ema_spread_lookback,
            h2_max_percentile=self.h2_max_percentile,
            h3_confirmation_window_bars=self.h3_confirmation_window_bars,
            h4_reward_risk_ratio=self.h4_reward_risk_ratio,
            trade_count_ratio_warning=self.hypothesis_trade_count_ratio_warning,
        )

    def multiregime_config(self) -> "MultiRegimeConfig":
        """Build the fixed V3.2.1 window and diagnostic-regime configuration."""

        from src.research.multiregime.models import MultiRegimeConfig

        return MultiRegimeConfig(
            window_days=self.multiregime_window_days,
            minimum_window_days=self.multiregime_min_window_days,
            volatility_lookback=self.multiregime_vol_lookback,
            volatility_low_percentile=self.multiregime_vol_low_percentile,
            volatility_high_percentile=self.multiregime_vol_high_percentile,
            trade_count_ratio_required=self.hypothesis_trade_count_ratio_warning,
        )

    def mechanism_suite_config(self) -> "MechanismSuiteConfig":
        """Build the fixed H5-H7 mechanism registry; tuning is refused."""

        from src.hypotheses.mechanisms import MechanismSuiteConfig

        return MechanismSuiteConfig(
            h5_lookback=self.h5_lookback,
            h5_min_percentile=self.h5_min_percentile,
            h5_max_percentile=self.h5_max_percentile,
            h6_confirmation_bars=self.h6_confirmation_bars,
            h7_cost_opportunity_multiplier=self.h7_cost_opportunity_multiplier,
            trade_count_ratio_warning=self.hypothesis_trade_count_ratio_warning,
        )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Load a .env file and overlay it with process environment variables."""

        values: dict[str, str | None] = {}
        if env_file is not None:
            values.update(dotenv_values(env_file, encoding="utf-8-sig"))
        values.update(os.environ if environ is None else environ)

        def value(name: str, default: str) -> str:
            configured = values.get(name, default)
            return default if configured is None else str(configured)

        return cls(
            binance_api_key=value("BINANCE_API_KEY", ""),
            binance_secret_key=value("BINANCE_SECRET_KEY", ""),
            market_data_source=_parse_market_data_source(
                value("MARKET_DATA_SOURCE", "binance_public")
            ),
            execution_environment=_parse_execution_environment(
                value("EXECUTION_ENV", "testnet")
            ),
            trading_symbol=value("TRADING_SYMBOL", "BTCUSDC"),
            candle_interval=value("CANDLE_INTERVAL", "15m"),
            historical_candle_limit=_parse_int(
                "HISTORICAL_CANDLE_LIMIT",
                value("HISTORICAL_CANDLE_LIMIT", "300"),
            ),
            backtest_initial_capital_usdc=_parse_decimal(
                "BACKTEST_INITIAL_CAPITAL_USDC",
                value("BACKTEST_INITIAL_CAPITAL_USDC", "1000"),
            ),
            backtest_fee_bps=_parse_decimal(
                "BACKTEST_FEE_BPS", value("BACKTEST_FEE_BPS", "10")
            ),
            backtest_slippage_bps=_parse_decimal(
                "BACKTEST_SLIPPAGE_BPS",
                value("BACKTEST_SLIPPAGE_BPS", "2"),
            ),
            backtest_ambiguous_bar_policy=value(
                "BACKTEST_AMBIGUOUS_BAR_POLICY", "STOP_FIRST"
            ),
            max_capital_usdc=_parse_decimal(
                "MAX_CAPITAL_USDC", value("MAX_CAPITAL_USDC", "50")
            ),
            max_risk_per_trade_percent=_parse_decimal(
                "MAX_RISK_PER_TRADE_PERCENT",
                value("MAX_RISK_PER_TRADE_PERCENT", "0.5"),
            ),
            max_daily_loss_percent=_parse_decimal(
                "MAX_DAILY_LOSS_PERCENT", value("MAX_DAILY_LOSS_PERCENT", "2.0")
            ),
            baseline_fast_ema_period=_parse_int(
                "BASELINE_FAST_EMA_PERIOD", value("BASELINE_FAST_EMA_PERIOD", "20")
            ),
            baseline_slow_ema_period=_parse_int(
                "BASELINE_SLOW_EMA_PERIOD", value("BASELINE_SLOW_EMA_PERIOD", "50")
            ),
            baseline_rsi_period=_parse_int(
                "BASELINE_RSI_PERIOD", value("BASELINE_RSI_PERIOD", "14")
            ),
            baseline_rsi_min=_parse_decimal(
                "BASELINE_RSI_MIN", value("BASELINE_RSI_MIN", "52")
            ),
            baseline_rsi_max=_parse_decimal(
                "BASELINE_RSI_MAX", value("BASELINE_RSI_MAX", "68")
            ),
            baseline_volume_sma_period=_parse_int(
                "BASELINE_VOLUME_SMA_PERIOD",
                value("BASELINE_VOLUME_SMA_PERIOD", "20"),
            ),
            baseline_min_volume_ratio=_parse_decimal(
                "BASELINE_MIN_VOLUME_RATIO",
                value("BASELINE_MIN_VOLUME_RATIO", "0.80"),
            ),
            baseline_atr_period=_parse_int(
                "BASELINE_ATR_PERIOD", value("BASELINE_ATR_PERIOD", "14")
            ),
            baseline_atr_stop_multiplier=_parse_decimal(
                "BASELINE_ATR_STOP_MULTIPLIER",
                value("BASELINE_ATR_STOP_MULTIPLIER", "2.0"),
            ),
            baseline_reward_risk_ratio=_parse_decimal(
                "BASELINE_REWARD_RISK_RATIO",
                value("BASELINE_REWARD_RISK_RATIO", "2.0"),
            ),
            baseline_risk_per_trade_percent=_parse_decimal(
                "BASELINE_RISK_PER_TRADE_PERCENT",
                value("BASELINE_RISK_PER_TRADE_PERCENT", "0.50"),
            ),
            baseline_max_bars_in_position=_parse_int(
                "BASELINE_MAX_BARS_IN_POSITION",
                value("BASELINE_MAX_BARS_IN_POSITION", "96"),
            ),
            baseline_cooldown_bars=_parse_int(
                "BASELINE_COOLDOWN_BARS", value("BASELINE_COOLDOWN_BARS", "4")
            ),
            research_min_trades_warning=_parse_int(
                "RESEARCH_MIN_TRADES_WARNING",
                value("RESEARCH_MIN_TRADES_WARNING", "30"),
            ),
            h1_atr_percentile_lookback=_parse_int(
                "H1_ATR_PERCENTILE_LOOKBACK",
                value("H1_ATR_PERCENTILE_LOOKBACK", "100"),
            ),
            h1_max_percentile=_parse_decimal(
                "H1_MAX_PERCENTILE", value("H1_MAX_PERCENTILE", "75")
            ),
            h2_ema_spread_lookback=_parse_int(
                "H2_EMA_SPREAD_LOOKBACK",
                value("H2_EMA_SPREAD_LOOKBACK", "100"),
            ),
            h2_max_percentile=_parse_decimal(
                "H2_MAX_PERCENTILE", value("H2_MAX_PERCENTILE", "75")
            ),
            h3_confirmation_window_bars=_parse_int(
                "H3_CONFIRMATION_WINDOW_BARS",
                value("H3_CONFIRMATION_WINDOW_BARS", "8"),
            ),
            h4_reward_risk_ratio=_parse_decimal(
                "H4_REWARD_RISK_RATIO", value("H4_REWARD_RISK_RATIO", "1.0")
            ),
            hypothesis_trade_count_ratio_warning=_parse_decimal(
                "HYPOTHESIS_TRADE_COUNT_RATIO_WARNING",
                value("HYPOTHESIS_TRADE_COUNT_RATIO_WARNING", "0.40"),
            ),
            multiregime_window_days=_parse_int(
                "MULTIREGIME_WINDOW_DAYS",
                value("MULTIREGIME_WINDOW_DAYS", "90"),
            ),
            multiregime_min_window_days=_parse_int(
                "MULTIREGIME_MIN_WINDOW_DAYS",
                value("MULTIREGIME_MIN_WINDOW_DAYS", "60"),
            ),
            multiregime_vol_lookback=_parse_int(
                "MULTIREGIME_VOL_LOOKBACK",
                value("MULTIREGIME_VOL_LOOKBACK", "200"),
            ),
            multiregime_vol_low_percentile=_parse_decimal(
                "MULTIREGIME_VOL_LOW_PERCENTILE",
                value("MULTIREGIME_VOL_LOW_PERCENTILE", "33"),
            ),
            multiregime_vol_high_percentile=_parse_decimal(
                "MULTIREGIME_VOL_HIGH_PERCENTILE",
                value("MULTIREGIME_VOL_HIGH_PERCENTILE", "67"),
            ),
            h5_lookback=_parse_int(
                "H5_LOOKBACK", value("H5_LOOKBACK", "100")
            ),
            h5_min_percentile=_parse_decimal(
                "H5_MIN_PERCENTILE", value("H5_MIN_PERCENTILE", "25")
            ),
            h5_max_percentile=_parse_decimal(
                "H5_MAX_PERCENTILE", value("H5_MAX_PERCENTILE", "75")
            ),
            h6_confirmation_bars=_parse_int(
                "H6_CONFIRMATION_BARS", value("H6_CONFIRMATION_BARS", "1")
            ),
            h7_cost_opportunity_multiplier=_parse_decimal(
                "H7_COST_OPPORTUNITY_MULTIPLIER",
                value("H7_COST_OPPORTUNITY_MULTIPLIER", "5"),
            ),
        )


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Load and validate application settings."""

    return Settings.from_env(env_file=env_file)
