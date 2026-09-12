from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from src.backtest.v10_l2 import (
    ALLOW_PARTIAL_FILL,
    FULL_FILL_OR_REJECT,
    Decision,
    DepthLevel,
    DepthSnapshot,
    EventCompressor,
    EventRules,
    ExchangeConstraints,
    ExecutionRules,
    L2IntegrityError,
    NotionalFilter,
    QuantityFilter,
    SignalState,
    exchange_constraints_from_exchange_info,
    load_public_exchange_constraints,
    replay,
)

D = Decimal
BASE = datetime(2024, 1, 1, tzinfo=UTC)
TWO_THIRDS_AT_ENGINE_PRECISION = D("0." + "6" * 49 + "7")


def at(seconds: int) -> datetime:
    return BASE + timedelta(seconds=seconds)


def levels(*values: tuple[str, str]) -> tuple[DepthLevel, ...]:
    return tuple(DepthLevel(D(price), D(quantity)) for price, quantity in values)


def book(
    time: int,
    *,
    bids: tuple[tuple[str, str], ...] = (("99", "100"),),
    asks: tuple[tuple[str, str], ...] = (("101", "100"),),
    source: int | None = None,
    sequence: int | None = None,
) -> DepthSnapshot:
    return DepthSnapshot(
        "BTCUSDC",
        at(time if source is None else source),
        at(time),
        time if sequence is None else sequence,
        levels(*bids),
        levels(*asks),
    )


def decision(time: int, action: str, quantity: str = "1") -> Decision:
    return Decision(at(time), at(time), action, D(quantity) if action == "ENTER" else None)


def rules(fee: str = "0", slip: str = "0", latency: int = 0, age: int = 30) -> ExecutionRules:
    return ExecutionRules(D(fee), D(slip), timedelta(seconds=latency), timedelta(seconds=age))


def constraints(
    *,
    minimum: str = "0.1",
    maximum: str = "100",
    step: str = "0.1",
    min_notional: str | None = None,
    max_notional: str | None = None,
    market: QuantityFilter | None = None,
) -> ExchangeConstraints:
    notional = ()
    if min_notional is not None or max_notional is not None:
        notional = (
            NotionalFilter(
                "NOTIONAL",
                D(min_notional) if min_notional is not None else None,
                D(max_notional) if max_notional is not None else None,
                min_notional is not None,
                max_notional is not None,
                0,
            ),
        )
    return ExchangeConstraints(
        "BTCUSDC",
        QuantityFilter(D(minimum), D(maximum), D(step)),
        market,
        notional,
    )


def run(
    *,
    snapshots=None,
    decisions=None,
    config=None,
    limits=None,
    mode=FULL_FILL_OR_REJECT,
    cash="100000",
):
    return replay(
        session_id="synthetic",
        snapshots=snapshots if snapshots is not None else [book(0), book(10, bids=(("109", "100"),), asks=(("111", "100"),))],
        decisions=decisions if decisions is not None else [decision(0, "ENTER"), decision(10, "EXIT")],
        rules=config or rules(),
        constraints=limits or constraints(),
        fill_mode=mode,
        initial_cash=D(cash),
    )


def state(time: int, value: int) -> SignalState:
    return SignalState(at(time), at(time), at(time), D(value))


def compressor(persistence: int = 2, cooldown: int = 3) -> EventCompressor:
    return EventCompressor(EventRules(D(2), D(1), persistence, timedelta(seconds=cooldown)))


def test_single_level_full_fill_uses_actual_ask_and_bid():
    trade = run().trades[0]
    assert trade.entry.requested_quantity == D(1)
    assert trade.entry.filled_quantity == D(1)
    assert trade.entry.fill_ratio == D(1)
    assert trade.entry.levels_consumed == 1
    assert trade.entry.best_quote == D(101)
    assert trade.entry.book_vwap == D(101)
    assert trade.entry.vwap == D(101)
    assert trade.exit.best_quote == D(109)
    assert trade.exit.vwap == D(109)


def test_multi_level_ask_sweep_has_exact_vwap_worst_fill_and_depth_slippage():
    trade = run(
        snapshots=[
            book(0, asks=(("101", "1"), ("102", "2"))),
            book(10, bids=(("110", "5"),), asks=(("112", "5"),)),
        ],
        decisions=[decision(0, "ENTER", "2"), decision(10, "EXIT")],
    ).trades[0]
    fill = trade.entry
    assert fill.levels_consumed == 2
    assert fill.book_vwap == D("101.5")
    assert fill.vwap == D("101.5")
    assert fill.worst_book_price == D(102)
    assert fill.worst_fill_price == D(102)
    assert fill.depth_slippage_per_unit == D("0.5")
    assert fill.depth_slippage_cost == D(1)


def test_multi_level_bid_sweep_has_exact_vwap_and_depth_slippage():
    trade = run(
        snapshots=[
            book(0, asks=(("101", "5"),)),
            book(10, bids=(("109", "1"), ("108", "2")), asks=(("111", "5"),)),
        ],
        decisions=[decision(0, "ENTER", "2"), decision(10, "EXIT")],
    ).trades[0]
    fill = trade.exit
    assert fill.levels_consumed == 2
    assert fill.book_vwap == D("108.5")
    assert fill.worst_fill_price == D(108)
    assert fill.depth_slippage_per_unit == D("0.5")
    assert fill.depth_slippage_cost == D(1)


def test_full_fill_rejects_insufficient_causal_liquidity():
    with pytest.raises(L2IntegrityError, match="insufficient causal depth"):
        run(
            snapshots=[book(0, asks=(("101", "1"),)), book(10)],
            decisions=[decision(0, "ENTER", "2"), decision(10, "EXIT")],
        )


def test_partial_entry_retains_exact_filled_position_quantity():
    result = run(
        snapshots=[book(0, asks=(("101", "2"),)), book(10, bids=(("109", "2"),), asks=(("111", "2"),))],
        decisions=[decision(0, "ENTER", "3"), decision(10, "EXIT")],
        limits=constraints(minimum="1", step="1"),
        mode=ALLOW_PARTIAL_FILL,
    )
    trade = result.trades[0]
    assert trade.entry.requested_quantity == D(3)
    assert trade.entry.filled_quantity == D(2)
    assert trade.entry.fill_ratio == TWO_THIRDS_AT_ENGINE_PRECISION
    assert trade.exit.requested_quantity == D(2)
    assert trade.exit.filled_quantity == D(2)


def test_partial_exit_requires_explicit_later_exit_and_reconciles_trade():
    trade = run(
        snapshots=[
            book(0, asks=(("101", "3"),)),
            book(10, bids=(("109", "2"),), asks=(("111", "3"),)),
            book(20, bids=(("108", "1"),), asks=(("110", "3"),)),
        ],
        decisions=[decision(0, "ENTER", "3"), decision(10, "EXIT"), decision(20, "EXIT")],
        limits=constraints(minimum="1", step="1"),
        mode=ALLOW_PARTIAL_FILL,
    ).trades[0]
    assert len(trade.exit_fills) == 2
    assert trade.exit_fills[0].requested_quantity == D(3)
    assert trade.exit_fills[0].filled_quantity == D(2)
    assert trade.exit_fills[0].fill_ratio == TWO_THIRDS_AT_ENGINE_PRECISION
    assert trade.exit_fills[1].requested_quantity == D(1)
    assert sum(fill.filled_quantity for fill in trade.exit_fills) == trade.entry.filled_quantity
    assert trade.exit_timestamp == at(20)


def test_partial_exit_rejects_an_untradable_residual_position():
    with pytest.raises(L2IntegrityError, match="residual position quantity is below minimum"):
        run(
            snapshots=[book(0, asks=(("101", "2"),)), book(10, bids=(("109", "1.5"),), asks=(("111", "2"),))],
            decisions=[decision(0, "ENTER", "2"), decision(10, "EXIT")],
            limits=constraints(minimum="1", step="0.5"),
            mode=ALLOW_PARTIAL_FILL,
        )


def test_partial_position_must_be_explicitly_closed():
    with pytest.raises(L2IntegrityError, match="partially closed"):
        run(
            snapshots=[book(0, asks=(("101", "2"),)), book(10, bids=(("109", "1"),), asks=(("111", "2"),))],
            decisions=[decision(0, "ENTER", "2"), decision(10, "EXIT")],
            limits=constraints(minimum="1", step="1"),
            mode=ALLOW_PARTIAL_FILL,
        )


@pytest.mark.parametrize(
    "quantity,message",
    [("0.5", "below minimum"), ("11", "above maximum"), ("1.25", "step size")],
)
def test_requested_quantity_constraints_are_exact(quantity, message):
    with pytest.raises(L2IntegrityError, match=message):
        run(
            decisions=[decision(0, "ENTER", quantity), decision(10, "EXIT")],
            limits=constraints(minimum="1", maximum="10", step="0.5"),
        )


@pytest.mark.parametrize("quantity", [D(0), D(-1), D("NaN"), D("Infinity"), 1.0])
def test_nonpositive_nonfinite_or_non_decimal_quantity_rejected(quantity):
    with pytest.raises(L2IntegrityError):
        Decision(at(0), at(0), "ENTER", quantity)


def test_market_lot_size_is_applied_in_addition_to_lot_size():
    limits = constraints(market=QuantityFilter(D(1), D(10), D(1)))
    with pytest.raises(L2IntegrityError, match="step size"):
        run(decisions=[decision(0, "ENTER", "1.5"), decision(10, "EXIT")], limits=limits)


def test_minimum_and_maximum_market_notional_are_enforced_on_fills():
    with pytest.raises(L2IntegrityError, match="below NOTIONAL minimum"):
        run(limits=constraints(min_notional="102"))
    with pytest.raises(L2IntegrityError, match="above NOTIONAL maximum"):
        run(limits=constraints(max_notional="100"))


def test_min_notional_market_applicability_is_enforced_exactly():
    quantity = QuantityFilter(D("0.1"), D(100), D("0.1"))
    active = ExchangeConstraints(
        "BTCUSDC",
        quantity,
        None,
        (NotionalFilter("MIN_NOTIONAL", D(102), None, True, False, 5),),
    )
    with pytest.raises(L2IntegrityError, match="MIN_NOTIONAL minimum"):
        run(limits=active)
    inactive = ExchangeConstraints(
        "BTCUSDC",
        quantity,
        None,
        (NotionalFilter("MIN_NOTIONAL", D(102), None, False, False, 5),),
    )
    assert run(limits=inactive).trades[0].entry.notional == D(101)


def exchange_info():
    return {
        "symbols": [{
            "symbol": "BTCUSDC",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "100", "stepSize": "0.001"},
                {"filterType": "MARKET_LOT_SIZE", "minQty": "0", "maxQty": "10", "stepSize": "0"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10", "applyToMarket": True, "avgPriceMins": 5},
                {"filterType": "NOTIONAL", "minNotional": "5", "maxNotional": "100000", "applyMinToMarket": False, "applyMaxToMarket": True, "avgPriceMins": 5},
            ],
        }]
    }


def test_public_exchange_info_adapter_preserves_relevant_spot_semantics():
    result = exchange_constraints_from_exchange_info(exchange_info())
    assert result.symbol == "BTCUSDC"
    assert result.lot_size == QuantityFilter(D("0.001"), D(100), D("0.001"))
    assert result.market_lot_size == QuantityFilter(None, D(10), None)
    assert result.notional_filters[0] == NotionalFilter("MIN_NOTIONAL", D(10), None, True, False, 5)
    assert result.notional_filters[1] == NotionalFilter("NOTIONAL", D(5), D(100000), False, True, 5)
    # PRICE_FILTER is deliberately absent; observed book prices are never tick-rounded.
    assert "tick" not in result.__dataclass_fields__


def test_public_client_adapter_only_calls_unauthenticated_exchange_info():
    class Client:
        calls = []

        def get_exchange_info(self, symbol=None):
            self.calls.append(symbol)
            return exchange_info()

    client = Client()
    result = load_public_exchange_constraints(client)
    assert result.symbol == "BTCUSDC"
    assert client.calls == ["BTCUSDC"]


@pytest.mark.parametrize("mutation,message", [
    (lambda p: p["symbols"][0].update(filters=[]), "LOT_SIZE"),
    (lambda p: p["symbols"].append(p["symbols"][0].copy()), "exactly one"),
    (lambda p: p["symbols"][0]["filters"].append(p["symbols"][0]["filters"][1].copy()), "duplicate"),
    (lambda p: p["symbols"][0]["filters"][1].update(stepSize=0.001), "exact decimal"),
    (lambda p: p["symbols"][0]["filters"][3].update(applyToMarket="true"), "boolean"),
])
def test_malformed_exchange_info_is_rejected(mutation, message):
    payload = exchange_info()
    mutation(payload)
    with pytest.raises(L2IntegrityError, match=message):
        exchange_constraints_from_exchange_info(payload)


def test_observed_book_prices_are_not_rounded_to_price_filter_tick():
    payload = exchange_info()
    limits = exchange_constraints_from_exchange_info(payload)
    trade = run(
        snapshots=[
            book(0, asks=(("101.003", "1"),)),
            book(10, bids=(("109.007", "1"),), asks=(("111.001", "1"),)),
        ],
        limits=limits,
    ).trades[0]
    assert trade.entry.vwap == D("101.003")
    assert trade.exit.vwap == D("109.007")


def test_cost_decomposition_is_additive_with_depth_slippage_fees_and_adverse_slippage():
    trade = run(
        snapshots=[
            book(0, bids=(("99", "5"),), asks=(("101", "1"), ("102", "2"))),
            book(10, bids=(("109", "1"), ("108", "2")), asks=(("111", "5"),)),
        ],
        decisions=[decision(0, "ENTER", "2"), decision(10, "EXIT")],
        config=rules(fee="10", slip="2"),
    ).trades[0]
    assert trade.entry.book_vwap == D("101.5")
    assert trade.exit.book_vwap == D("108.5")
    assert trade.entry.vwap == D("101.52030")
    assert trade.exit.vwap == D("108.47830")
    assert trade.spread_cost == D(4)
    assert trade.depth_slippage_cost == D(2)
    assert trade.additional_slippage_cost == D("0.08400")
    assert trade.fees == D("0.41999720")
    assert trade.mid_price_move - trade.total_cost == trade.net_pnl
    assert trade.filled_gross_pnl - trade.fees == trade.net_pnl
    assert run().summary()["total_depth_slippage_cost"] == D(0)


def test_latency_selects_latest_available_causal_depth_at_execution_time():
    trade = run(
        snapshots=[
            book(0),
            book(2, asks=(("104", "10"),), bids=(("102", "10"),)),
            book(12, asks=(("112", "10"),), bids=(("110", "10"),)),
            book(13, asks=(("1001", "10"),), bids=(("999", "10"),)),
        ],
        config=rules(latency=2),
    ).trades[0]
    assert trade.entry.executed_at == at(2)
    assert trade.entry.vwap == D(104)
    assert trade.exit.executed_at == at(12)
    assert trade.exit.vwap == D(110)


def test_future_depth_is_never_used_and_future_append_does_not_change_result():
    base = [book(0), book(10)]
    first = run(snapshots=base)
    second = run(snapshots=base + [book(11, bids=(("999", "1"),), asks=(("1001", "1"),))])
    assert first.trades == second.trades
    assert first.summary() == second.summary()
    assert first.input_sha256 != second.input_sha256
    with pytest.raises(L2IntegrityError, match="no causally available"):
        run(snapshots=[book(1)])


def test_stale_depth_and_insufficient_causal_provenance_are_rejected():
    with pytest.raises(L2IntegrityError, match="stale"):
        run(snapshots=[book(0)], config=rules(age=9))
    with pytest.raises(L2IntegrityError, match="source is in the future"):
        book(0, source=1)
    with pytest.raises(L2IntegrityError, match="sequence provenance"):
        book(0, sequence=-1)
    wrong_symbol = DepthSnapshot("ETHUSDC", at(0), at(0), 0, levels(("99", "1")), levels(("101", "1")))
    with pytest.raises(L2IntegrityError, match="symbol"):
        run(snapshots=[wrong_symbol])


def test_depth_order_crossing_duplicates_and_regressions_are_rejected():
    with pytest.raises(L2IntegrityError, match="nonempty levels"):
        DepthSnapshot("BTCUSDC", at(0), at(0), 0, (), levels(("101", "1")))
    with pytest.raises(L2IntegrityError, match="finite Decimal"):
        DepthLevel(D("NaN"), D(1))
    with pytest.raises(L2IntegrityError, match="crossed or locked"):
        book(0, bids=(("101", "1"),), asks=(("101", "1"),))
    with pytest.raises(L2IntegrityError, match="price-monotonic"):
        book(0, bids=(("98", "1"), ("99", "1")))
    with pytest.raises(L2IntegrityError, match="conflicting"):
        book(0, asks=(("101", "1"), ("101", "2")))
    with pytest.raises(L2IntegrityError, match="availability"):
        run(snapshots=[book(0), book(0)])
    with pytest.raises(L2IntegrityError, match="source timestamps regressed"):
        run(snapshots=[book(0), book(2, source=-1)])
    with pytest.raises(L2IntegrityError, match="sequence provenance regressed"):
        run(snapshots=[book(0, sequence=2), book(2, sequence=1)])


def test_replay_json_and_hash_are_deterministic_and_bind_all_inputs():
    first = run(config=rules("10", "2"))
    with localcontext() as context:
        context.prec = 8
        second = run(config=rules("10", "2"))
    assert first.to_json() == second.to_json()
    assert first.sha256() == second.sha256()
    variants = [
        run(config=rules("11", "2")),
        run(limits=constraints(maximum="101"), config=rules("10", "2")),
        run(mode=ALLOW_PARTIAL_FILL, config=rules("10", "2")),
        run(snapshots=[book(0, sequence=1), book(10, sequence=10)], config=rules("10", "2")),
        run(decisions=[Decision(at(0), at(-1), "ENTER", D(1)), decision(10, "EXIT")], config=rules("10", "2")),
    ]
    assert len({first.input_sha256, *(item.input_sha256 for item in variants)}) == 6
    assert len({first.sha256(), *(item.sha256() for item in variants)}) == 6
    assert '"constraints"' in first.to_json()
    assert '"snapshot"' in first.to_json()


def test_exact_decimal_accounting_is_independent_of_ambient_context():
    def calculate():
        return run(
            snapshots=[
                book(0, asks=(("101.123456789", "3"),)),
                book(10, bids=(("109.987654321", "3"),), asks=(("111.000000001", "3"),)),
            ],
            decisions=[decision(0, "ENTER", "1.3"), decision(10, "EXIT")],
            config=rules("10", "2"),
        )

    expected = calculate()
    with localcontext() as context:
        context.prec = 7
        actual = calculate()
    assert actual.to_json() == expected.to_json()
    assert actual.final_cash - actual.initial_cash == actual.trades[0].net_pnl


def test_position_and_decision_integrity_still_rejects_overlap_and_lookahead():
    with pytest.raises(L2IntegrityError, match="overlapping"):
        run(decisions=[decision(0, "ENTER"), decision(1, "ENTER")])
    with pytest.raises(L2IntegrityError, match="precedes signal"):
        Decision(at(0), at(1), "ENTER", D(1))
    with pytest.raises(L2IntegrityError, match="pending"):
        run(decisions=[decision(0, "ENTER"), decision(1, "EXIT")], config=rules(latency=2))
    with pytest.raises(L2IntegrityError, match="choose"):
        run(mode="UNSPECIFIED")


def test_signal_persistence_cooldown_and_nonoverlap_contract_remains_deterministic():
    def emitted():
        engine = compressor()
        assert engine.observe(state(0, 3)) is None
        assert engine.observe(state(1, 1)) is None
        assert engine.observe(state(2, 3)) is None
        event = engine.observe(state(3, 3))
        assert engine.observe(state(4, 1)) is None
        engine.release(at(5))
        assert engine.observe(state(6, 1)) is None
        assert engine.observe(state(8, 1)) is None
        assert engine.observe(state(9, 3)) is None
        return event, engine.observe(state(10, 3))

    assert emitted() == emitted()


@pytest.mark.parametrize("create", [
    lambda: rules(fee="-1"),
    lambda: rules(slip="10000"),
    lambda: rules(latency=-1),
    lambda: QuantityFilter(D(2), D(1), D(1)),
    lambda: NotionalFilter("BAD", D(1), None, True, False, 0),
    lambda: EventRules(D(1), D(2), 1, timedelta(0)),
    lambda: EventRules(D(2), D(1), 0, timedelta(0)),
])
def test_invalid_frozen_caller_rules_are_rejected(create):
    with pytest.raises(L2IntegrityError):
        create()
