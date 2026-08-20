"""Post-backtest audit of requested risk, actual risk, capital caps, and friction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.execution import basis_points_rate
from src.backtest.models import BacktestConfig, EquityPoint, Trade
from src.diagnostics.cost_analysis import decompose_trade_costs
from src.research.evaluation import SignalRecord
from src.strategy.models import StrategyAction, TrendMomentumConfig


@dataclass(frozen=True, slots=True)
class RiskCapitalAuditRecord:
    trade_id: str

    equity_at_signal: Decimal
    cash_at_signal: Decimal

    configured_risk_percent: Decimal
    requested_risk_budget: Decimal

    stop_distance: Decimal

    risk_sized_notional: Decimal
    configured_position_cap: Decimal
    affordable_notional: Decimal
    effective_max_notional: Decimal

    actual_entry_notional: Decimal
    actual_stop_risk: Decimal

    actual_risk_percent_of_equity: Decimal
    risk_budget_utilization_percent: Decimal
    capital_utilization_percent: Decimal

    limited_by_risk_budget: bool
    limited_by_position_cap: bool
    limited_by_available_cash: bool

    total_fee: Decimal
    slippage_drag: Decimal
    total_friction: Decimal
    friction_to_actual_risk_percent: Decimal


@dataclass(frozen=True, slots=True)
class RiskCapitalAuditSummary:
    trade_count: int

    risk_budget_limited_count: int
    position_cap_limited_count: int
    available_cash_limited_count: int

    position_cap_limited_percent: Decimal

    average_requested_risk_percent: Decimal
    average_actual_risk_percent: Decimal
    maximum_actual_risk_percent: Decimal

    average_risk_budget_utilization_percent: Decimal
    average_capital_utilization_percent: Decimal

    average_friction_to_actual_risk_percent: Decimal


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, start=Decimal("0")) / Decimal(len(values))


def build_risk_capital_audit(
    *,
    trades: tuple[Trade, ...],
    signals: tuple[SignalRecord, ...],
    equity_curve: tuple[EquityPoint, ...],
    backtest_config: BacktestConfig,
    strategy_config: TrendMomentumConfig,
) -> tuple[RiskCapitalAuditRecord, ...]:
    """Build post-trade risk/capital diagnostics without altering execution."""

    entry_signals = {
        signal.timestamp: signal
        for signal in signals
        if signal.action is StrategyAction.ENTER_LONG
    }

    equity_by_timestamp = {
        point.timestamp: point
        for point in equity_curve
    }

    fee_rate = basis_points_rate(backtest_config.fee_bps)

    records: list[RiskCapitalAuditRecord] = []

    for trade in trades:
        signal = entry_signals.get(trade.entry_signal_time)
        if signal is None:
            raise ValueError(
                f"Entry signal not found for trade {trade.trade_id}."
            )

        if signal.atr is None:
            raise ValueError(
                f"ATR unavailable at entry for trade {trade.trade_id}."
            )

        equity_point = equity_by_timestamp.get(trade.entry_signal_time)
        if equity_point is None:
            raise ValueError(
                f"Equity snapshot not found for trade {trade.trade_id} "
                f"at {trade.entry_signal_time.isoformat()}."
            )

        equity = equity_point.total_equity
        cash = equity_point.cash

        if equity <= 0 or cash <= 0:
            raise ValueError(
                f"Invalid signal-time account state for {trade.trade_id}."
            )

        requested_risk_budget = (
            equity
            * strategy_config.risk_per_trade_percent
            / Decimal("100")
        )

        stop_distance = (
            signal.atr
            * strategy_config.atr_stop_multiplier
        )

        if stop_distance <= 0:
            raise ValueError(
                f"Invalid stop distance for {trade.trade_id}."
            )

        # This mirrors the V2 risk-sized BUY calculation using the
        # actual simulated fill price.
        risk_sized_notional = (
            requested_risk_budget
            / stop_distance
            * trade.entry_price
        )

        affordable_notional = cash / (
            Decimal("1") + fee_rate
        )

        configured_cap = (
            strategy_config.maximum_position_notional_usdc
        )

        effective_max_notional = min(
            affordable_notional,
            configured_cap,
        )

        actual_stop_risk = (
            trade.quantity
            * stop_distance
        )

        actual_risk_percent = (
            actual_stop_risk
            / equity
            * Decimal("100")
        )

        risk_budget_utilization = (
            actual_stop_risk
            / requested_risk_budget
            * Decimal("100")
        )

        capital_utilization = (
            trade.entry_notional
            / equity
            * Decimal("100")
        )

        limited_by_risk_budget = (
            risk_sized_notional <= configured_cap
            and risk_sized_notional <= affordable_notional
        )

        limited_by_position_cap = (
            configured_cap <= risk_sized_notional
            and configured_cap <= affordable_notional
        )

        limited_by_available_cash = (
            affordable_notional <= risk_sized_notional
            and affordable_notional <= configured_cap
        )

        costs = decompose_trade_costs(
            trade,
            backtest_config.slippage_bps,
        )

        total_friction = (
            trade.total_fee
            + abs(costs.slippage_drag)
        )

        friction_to_actual_risk = (
            total_friction
            / actual_stop_risk
            * Decimal("100")
        )

        records.append(
            RiskCapitalAuditRecord(
                trade_id=trade.trade_id,
                equity_at_signal=equity,
                cash_at_signal=cash,
                configured_risk_percent=(
                    strategy_config.risk_per_trade_percent
                ),
                requested_risk_budget=requested_risk_budget,
                stop_distance=stop_distance,
                risk_sized_notional=risk_sized_notional,
                configured_position_cap=configured_cap,
                affordable_notional=affordable_notional,
                effective_max_notional=effective_max_notional,
                actual_entry_notional=trade.entry_notional,
                actual_stop_risk=actual_stop_risk,
                actual_risk_percent_of_equity=actual_risk_percent,
                risk_budget_utilization_percent=(
                    risk_budget_utilization
                ),
                capital_utilization_percent=capital_utilization,
                limited_by_risk_budget=limited_by_risk_budget,
                limited_by_position_cap=limited_by_position_cap,
                limited_by_available_cash=limited_by_available_cash,
                total_fee=trade.total_fee,
                slippage_drag=costs.slippage_drag,
                total_friction=total_friction,
                friction_to_actual_risk_percent=(
                    friction_to_actual_risk
                ),
            )
        )

    return tuple(records)


def summarize_risk_capital_audit(
    records: tuple[RiskCapitalAuditRecord, ...],
) -> RiskCapitalAuditSummary:
    """Summarize capital/risk behavior across completed trades."""

    if not records:
        return RiskCapitalAuditSummary(
            trade_count=0,
            risk_budget_limited_count=0,
            position_cap_limited_count=0,
            available_cash_limited_count=0,
            position_cap_limited_percent=Decimal("0"),
            average_requested_risk_percent=Decimal("0"),
            average_actual_risk_percent=Decimal("0"),
            maximum_actual_risk_percent=Decimal("0"),
            average_risk_budget_utilization_percent=Decimal("0"),
            average_capital_utilization_percent=Decimal("0"),
            average_friction_to_actual_risk_percent=Decimal("0"),
        )

    trade_count = len(records)

    risk_limited = sum(
        record.limited_by_risk_budget
        for record in records
    )

    cap_limited = sum(
        record.limited_by_position_cap
        for record in records
    )

    cash_limited = sum(
        record.limited_by_available_cash
        for record in records
    )

    return RiskCapitalAuditSummary(
        trade_count=trade_count,
        risk_budget_limited_count=risk_limited,
        position_cap_limited_count=cap_limited,
        available_cash_limited_count=cash_limited,
        position_cap_limited_percent=(
            Decimal(cap_limited)
            / Decimal(trade_count)
            * Decimal("100")
        ),
        average_requested_risk_percent=_mean(
            [
                record.configured_risk_percent
                for record in records
            ]
        ),
        average_actual_risk_percent=_mean(
            [
                record.actual_risk_percent_of_equity
                for record in records
            ]
        ),
        maximum_actual_risk_percent=max(
            record.actual_risk_percent_of_equity
            for record in records
        ),
        average_risk_budget_utilization_percent=_mean(
            [
                record.risk_budget_utilization_percent
                for record in records
            ]
        ),
        average_capital_utilization_percent=_mean(
            [
                record.capital_utilization_percent
                for record in records
            ]
        ),
        average_friction_to_actual_risk_percent=_mean(
            [
                record.friction_to_actual_risk_percent
                for record in records
            ]
        ),
    )