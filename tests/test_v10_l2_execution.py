from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from src.backtest.v10_l2 import (
    Decision, EventCompressor, EventRules, ExecutionRules, L2IntegrityError,
    Quote, SignalState, replay,
)


D = Decimal
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def at(seconds):
    return BASE + timedelta(seconds=seconds)


def quote(t, bid="99", ask="101", *, source=None):
    return Quote(at(t if source is None else source), at(t), D(bid), D(ask))


def decision(t, action, quantity="1"):
    return Decision(at(t), at(t), action, D(quantity) if action == "ENTER" else None)


def rules(fee="0", slip="0", latency=0, age=30):
    return ExecutionRules(D(fee), D(slip), timedelta(seconds=latency), timedelta(seconds=age))


def run(quotes=None, decisions=None, config=None, cash="1000"):
    return replay(
        session_id="synthetic", initial_cash=D(cash), rules=config or rules(),
        quotes=quotes if quotes is not None else [quote(0), quote(10, "109", "111")],
        decisions=decisions if decisions is not None else [decision(0, "ENTER"), decision(10, "EXIT")],
    )


def state(t, value):
    return SignalState(at(t), at(t), at(t), D(str(value)))


def compressor(persistence=2, cooldown=3):
    return EventCompressor(EventRules(D(2), D(1), persistence, timedelta(seconds=cooldown)))


def test_bid_ask_execution_and_additive_spread_accounting():
    result = run()
    trade = result.trades[0]
    assert trade.entry.price == D(101)
    assert trade.exit.price == D(109)
    assert trade.mid_price_move == D(10)
    assert trade.executable_gross_pnl == D(8)
    assert trade.spread_cost == D(2)
    assert trade.net_pnl == D(8)
    assert trade.holding_time == timedelta(seconds=10)
    assert result.final_cash == D(1008)


def test_unchanged_mid_loses_both_half_spreads():
    trade = run(quotes=[quote(0), quote(10)]).trades[0]
    assert trade.mid_price_move == 0
    assert trade.net_pnl == -2
    assert trade.spread_cost == 2


@pytest.mark.parametrize("fee,slip", [("10", "2"), ("20", "4")])
def test_fee_slippage_are_charged_on_both_actual_notionals(fee, slip):
    result = run(config=rules(fee, slip))
    trade = result.trades[0]
    entry = D(101) * (1 + D(slip) / 10000)
    exit_price = D(109) * (1 - D(slip) / 10000)
    fees = (entry + exit_price) * D(fee) / 10000
    assert trade.entry.price == entry
    assert trade.exit.price == exit_price
    assert trade.fees == fees
    assert trade.slippage_cost == (entry - 101) + (109 - exit_price)
    assert trade.net_pnl == exit_price - entry - fees
    assert trade.net_pnl == trade.mid_price_move - trade.spread_cost - trade.slippage_cost - trade.fees
    assert result.final_cash - result.initial_cash == trade.net_pnl
    assert trade.entry.fee == entry * D(fee) / 10000
    assert trade.exit.fee == exit_price * D(fee) / 10000


def test_latency_uses_latest_available_quote_at_exact_execution_time():
    result = run(
        quotes=[quote(0), quote(2, "102", "104"), quote(12, "110", "112"), quote(13, "999", "1001")],
        config=rules(latency=2),
    )
    trade = result.trades[0]
    assert trade.entry.executed_at == at(2)
    assert trade.entry.price == 104
    assert trade.exit.executed_at == at(12)
    assert trade.exit.price == 110
    assert trade.exit.quote.available_at == at(12)


def test_late_available_quote_cannot_be_backdated_to_source_time():
    late = Quote(at(1), at(3), D(500), D(501))
    result = run(quotes=[quote(0), late, quote(12)], config=rules(latency=2))
    assert result.trades[0].entry.price == 101


def test_appended_future_quotes_do_not_change_fills_or_accounting():
    base = [quote(0), quote(10)]
    first = run(quotes=base)
    second = run(quotes=base + [quote(11, "1000", "2000")])
    assert first.trades == second.trades
    assert first.summary() == second.summary()
    assert first.input_sha256 != second.input_sha256


@pytest.mark.parametrize("quotes,config,message", [
    ([quote(1)], rules(), "no available"),
    ([quote(0)], rules(age=1), "stale"),
    ([quote(0), quote(0)], rules(), "strictly increasing"),
    ([quote(10), quote(0)], rules(), "strictly increasing"),
    ([quote(0), quote(2, source=-1)], rules(), "regressed"),
    ([quote(0, source=-10)], rules(age=1), "stale"),
])
def test_quote_integrity_rejections(quotes, config, message):
    with pytest.raises(L2IntegrityError, match=message):
        run(quotes=quotes, config=config)


@pytest.mark.parametrize("bid,ask", [(None, D(101)), (D(99), None), (D(102), D(101)), (D(0), D(1)), (D('NaN'), D(1)), (1.0, D(2))])
def test_missing_crossed_or_malformed_prices_rejected(bid, ask):
    with pytest.raises(L2IntegrityError):
        Quote(at(0), at(0), bid, ask)


def test_future_source_or_signal_availability_rejected():
    with pytest.raises(L2IntegrityError):
        Quote(at(1), at(0), D(99), D(101))
    with pytest.raises(L2IntegrityError):
        Decision(at(0), at(1), "ENTER", D(1))
    with pytest.raises(L2IntegrityError):
        SignalState(at(0), at(0), at(1), D(1))
    with pytest.raises(L2IntegrityError):
        Quote(BASE.replace(tzinfo=None), at(0), D(99), D(101))


@pytest.mark.parametrize("decisions,message", [
    ([decision(0, "ENTER"), decision(1, "ENTER")], "overlapping"),
    ([decision(0, "EXIT")], "without"),
    ([decision(0, "ENTER")], "unclosed"),
    ([decision(1, "ENTER"), decision(0, "EXIT")], "strictly increasing"),
    ([decision(0, "ENTER"), decision(0, "EXIT")], "strictly increasing"),
])
def test_position_and_decision_integrity(decisions, message):
    with pytest.raises(L2IntegrityError, match=message):
        run(decisions=decisions)


def test_pending_decisions_and_insufficient_cash_rejected():
    with pytest.raises(L2IntegrityError, match="pending"):
        run(decisions=[decision(0, "ENTER"), decision(1, "EXIT")], config=rules(latency=2))
    with pytest.raises(L2IntegrityError, match="cash"):
        run(cash="101", config=rules(fee="10"))


def test_aggregate_accounting_and_profit_factor():
    result = run(
        quotes=[quote(0), quote(10, "109", "111"), quote(20), quote(30)],
        decisions=[decision(0, "ENTER"), decision(10, "EXIT"), decision(20, "ENTER"), decision(30, "EXIT")],
    )
    summary = result.summary()
    assert summary["trade_count"] == 2
    assert summary["profit_factor"] == 4
    assert summary["win_rate"] == D("0.5")
    assert summary["total_spread_cost"] == 4
    assert summary["average_cost_per_trade"] == 2
    assert summary["turnover_notional"] == 410
    assert summary["turnover_over_initial_cash"] == D("0.410")
    assert result.final_cash == 1006
    empty = run(quotes=[], decisions=[])
    assert empty.summary()["trade_count"] == 0
    assert empty.summary()["net_expectancy"] is None
    assert empty.summary()["profit_factor"] is None
    assert run().summary()["profit_factor"] is None


def test_replay_is_deterministic_even_when_ambient_decimal_precision_changes():
    first = run(config=rules("10", "2"))
    with localcontext() as context:
        context.prec = 9
        second = run(config=rules("10", "2"))
        assert first.to_json() == second.to_json()
        assert first.sha256() == second.sha256()
    assert first.sha256() != run(config=rules("20", "4")).sha256()


def test_persistence_crossing_and_busy_reservation():
    engine = compressor()
    assert engine.observe(state(0, 3)) is None  # no observed rearm yet
    assert engine.observe(state(1, 1)) is None
    assert engine.observe(state(2, 2)) is None  # strict > threshold
    assert engine.observe(state(3, 3)) is None
    event = engine.observe(state(4, 3))
    assert event.crossing_at == at(3)
    assert event.decision_at == at(4)
    assert engine.observe(state(5, 1)) is None
    assert engine.observe(state(6, 4)) is None  # cannot queue another event


def test_persistence_resets_on_nonqualifying_values_and_gaps():
    engine = compressor()
    assert engine.observe(state(0, 1)) is None
    assert engine.observe(state(1, 3)) is None
    assert engine.observe(state(2, 2)) is None
    assert engine.observe(state(3, 3)) is None
    assert engine.observe(state(5, 3)) is None  # gap requires a fresh rearm
    assert engine.observe(state(6, 3)) is None
    assert engine.observe(state(7, 1)) is None
    assert engine.observe(state(8, 3)) is None
    assert engine.observe(state(9, 3)) is not None


def test_cooldown_starts_at_release_and_rearm_must_follow_cooldown():
    engine = compressor(persistence=1)
    engine.observe(state(0, 1))
    assert engine.observe(state(1, 3)) is not None
    engine.release(at(2))
    assert engine.observe(state(3, 1)) is None
    assert engine.observe(state(4, 3)) is None
    assert engine.observe(state(5, 3)) is None  # cooldown expired but unarmed
    assert engine.observe(state(6, 1)) is None
    assert engine.observe(state(7, 3)) is not None


def test_event_stream_is_deterministic_and_rejects_time_reversal():
    def events():
        engine = compressor()
        return [engine.observe(state(i, v)) for i, v in enumerate([1, 3, 3, 1, 4])]
    assert events() == events()
    engine = compressor()
    with pytest.raises(L2IntegrityError):
        engine.release(at(0))
    engine.observe(state(0, 1))
    with pytest.raises(L2IntegrityError):
        engine.observe(state(0, 3))


def test_compressed_event_can_drive_an_explicit_nonoverlapping_schedule():
    engine = compressor(persistence=2, cooldown=2)
    engine.observe(state(0, 1))
    engine.observe(state(1, 3))
    first = engine.observe(state(2, 3))
    engine.release(at(6))  # caller's explicit exit at decision 5 + latency 1
    assert engine.observe(state(7, 1)) is None  # cooldown
    engine.observe(state(8, 1))
    engine.observe(state(9, 3))
    second = engine.observe(state(10, 3))
    entries = [
        Decision(event.decision_at, event.signal_available_at, "ENTER", D(1))
        for event in (first, second)
    ]
    result = run(
        quotes=[quote(0), quote(6), quote(11), quote(14)],
        decisions=[entries[0], decision(5, "EXIT"), entries[1], decision(13, "EXIT")],
        config=rules(latency=1),
    )
    assert len(result.trades) == 2
    assert result.trades[0].entry.executed_at == at(3)
    assert result.trades[0].exit.executed_at == at(6)
    assert result.trades[1].entry.executed_at == at(11)
    assert result.final_cash == D(996)


def test_execution_quote_age_boundary_is_inclusive():
    result = run(quotes=[quote(0)], config=rules(age=10))
    assert result.trades[0].exit.quote.source_at == at(0)
    with pytest.raises(L2IntegrityError, match="stale"):
        run(quotes=[quote(0)], config=rules(age=9))


@pytest.mark.parametrize("create", [
    lambda: rules(fee="-1"), lambda: rules(slip="10000"),
    lambda: rules(latency=-1), lambda: rules(age=-1),
    lambda: EventRules(D(1), D(2), 1, timedelta(0)),
    lambda: EventRules(D(2), D(1), 0, timedelta(0)),
    lambda: EventRules(D(2), D(1), True, timedelta(0)),
    lambda: Decision(at(0), at(0), "SHORT", D(1)),
])
def test_invalid_caller_rules_rejected(create):
    with pytest.raises(L2IntegrityError):
        create()
