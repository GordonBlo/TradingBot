from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.cli.run_v7_h0 import _load_json, validate_v7_dataset_manifest
from src.cli.run_v7_orderflow_signal_diagnostic import (
    DIAGNOSTIC_ID,
    load_frozen_v6_ledger,
    v7_zero_boundary_groups,
    write_reports,
)
from src.diagnostics.v7_orderflow_signal import (
    alignment_state,
    build_causal_features,
    cliffs_delta,
    compare_feature,
    feature_window_consistency,
    qimb_slope_4,
)
from src.models.candle import Candle
from src.orderflow.aggregation import OrderFlowBucket


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def bucket(index: int, qimb: str, *, quote: str = "100", count: int = 10) -> OrderFlowBucket:
    total = Decimal(quote)
    imbalance = Decimal(qimb)
    buy = total * (Decimal("1") + imbalance) / Decimal("2")
    sell = total - buy
    return OrderFlowBucket(
        bucket_open_time=BASE + timedelta(minutes=15 * index),
        bucket_close_time=BASE + timedelta(minutes=15 * (index + 1)),
        aggregate_trade_count=count,
        underlying_trade_count=count * 2,
        total_base_volume=Decimal("10"),
        total_quote_volume=total,
        taker_buy_base_volume=Decimal("5"),
        taker_buy_quote_volume=buy,
        taker_buy_aggtrade_count=count // 2,
        taker_sell_base_volume=Decimal("5"),
        taker_sell_quote_volume=sell,
        taker_sell_aggtrade_count=count - count // 2,
    )


def candle(index: int, open_: str, close: str) -> Candle:
    low = min(Decimal(open_), Decimal(close))
    high = max(Decimal(open_), Decimal(close))
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal(open_),
        high=high,
        low=low,
        close=Decimal(close),
        volume=Decimal("1"),
        is_closed=True,
    )


def features():
    return build_causal_features(
        buckets=(
            bucket(1, "-0.4", quote="60", count=6),
            bucket(2, "-0.2", quote="90", count=9),
            bucket(3, "0.0", quote="150", count=15),
            bucket(4, "0.2", quote="200", count=20),
        ),
        signal_candle=candle(4, "100", "102"),
        t_minus_4_candle=candle(0, "98", "100"),
    )


def test_fixed_imbalance_dynamics_formulas() -> None:
    row = features()
    assert row["qimb_delta_1"] == Decimal("0.2")
    assert row["qimb_mean_4"] == Decimal("-0.1")
    assert row["qimb_slope_4"] == Decimal("0.2")
    assert qimb_slope_4((Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"))) == Decimal("1")


def test_cumulative_activity_and_interaction_formulas() -> None:
    row = features()
    assert row["cumulative_quote_imbalance_4"] == Decimal("-2") / Decimal("500")
    assert row["cumulative_signed_quote_4"] == Decimal("-2")
    assert row["quote_volume_ratio_vs_prev3"] == Decimal("2")
    assert row["aggtrade_size_ratio_vs_prev3"] == Decimal("1")
    assert row["underlying_trade_count_ratio_vs_prev3"] == Decimal("2")
    assert row["signal_return"] == Decimal("0.02")
    assert row["four_bar_return"] == Decimal("0.02")
    assert row["price_flow_interaction_1"] == Decimal("0.004")
    assert row["price_flow_interaction_4"] == Decimal("-0.00008")


def test_alignment_and_absorption_zero_boundaries() -> None:
    row = features()
    assert alignment_state(Decimal("1"), Decimal("-1")) == "PRICE_UP_FLOW_SELL"
    assert alignment_state(Decimal("0"), Decimal("1")) == "NEUTRAL"
    assert row["current_price_flow_alignment"] == "PRICE_UP_FLOW_BUY"
    assert row["four_bar_price_flow_alignment"] == "PRICE_UP_FLOW_SELL"
    assert row["sell_absorption_proxy"] is True
    assert row["buy_absorption_proxy"] is False


def test_feature_construction_rejects_noncausal_or_inexact_inputs() -> None:
    with pytest.raises(ValueError, match="Exactly four"):
        build_causal_features(
            buckets=(bucket(2, "0"), bucket(3, "0"), bucket(4, "0")),
            signal_candle=candle(4, "100", "101"),
            t_minus_4_candle=candle(0, "100", "100"),
        )
    with pytest.raises(ValueError, match="timestamp join"):
        build_causal_features(
            buckets=tuple(bucket(index, "0") for index in range(1, 5)),
            signal_candle=candle(5, "100", "101"),
            t_minus_4_candle=candle(1, "100", "100"),
        )


def test_cliffs_delta_known_examples() -> None:
    assert cliffs_delta((Decimal("3"), Decimal("4")), (Decimal("1"), Decimal("2"))) == 1
    assert cliffs_delta((Decimal("1"), Decimal("2")), (Decimal("3"), Decimal("4"))) == -1
    assert cliffs_delta((Decimal("1"), Decimal("2")), (Decimal("1"), Decimal("2"))) == 0


def test_effect_summary_math() -> None:
    rows = (
        {"net_r": Decimal("1"), "x": Decimal("4")},
        {"net_r": Decimal("2"), "x": Decimal("6")},
        {"net_r": Decimal("-1"), "x": Decimal("1")},
        {"net_r": Decimal("-2"), "x": Decimal("3")},
    )
    result = compare_feature(rows, "x")
    assert result["winner_mean"] == Decimal("5")
    assert result["loser_mean"] == Decimal("2")
    assert result["median_difference"] == Decimal("3")
    assert abs(
        result["standardized_mean_difference"]
        - Decimal("2.121320343559642573202533087")
    ) < Decimal("1e-27")
    assert result["cliffs_delta"] == Decimal("1")


def test_window_direction_consistency() -> None:
    rows = (
        {"window_id": "W1", "net_r": Decimal("1"), "x": Decimal("3")},
        {"window_id": "W1", "net_r": Decimal("-1"), "x": Decimal("1")},
        {"window_id": "W2", "net_r": Decimal("1"), "x": Decimal("0")},
        {"window_id": "W2", "net_r": Decimal("-1"), "x": Decimal("2")},
        {"window_id": "W3", "net_r": Decimal("1"), "x": Decimal("4")},
        {"window_id": "W3", "net_r": Decimal("-1"), "x": Decimal("1")},
    )
    aggregate = compare_feature(rows, "x")
    result = feature_window_consistency(rows, (aggregate,))[0]
    assert result["eligible_comparison_windows"] == 3
    assert result["winner_median_greater_windows"] == 2
    assert result["same_direction_windows"] == 2
    assert result["consistency_ratio"] == Decimal("2") / Decimal("3")
    assert result["directionally_unstable"] is False


def _group_row(qimb: str, net: str, window: str) -> dict:
    return {
        "window_id": window,
        "trade_id": f"{qimb}:{net}",
        "qimb_0": Decimal(qimb),
        "frictionless_r": Decimal(net) + Decimal("0.1"),
        "net_r": Decimal(net),
        "fee_r": Decimal("0.08"),
        "slippage_r": Decimal("0.02"),
        "total_friction_r": Decimal("0.10"),
    }


def test_v7_retained_filtered_zero_boundary_grouping(monkeypatch) -> None:
    monkeypatch.setattr("src.cli.run_v7_orderflow_signal_diagnostic.EXPECTED_CANDIDATES", 4)
    monkeypatch.setattr("src.cli.run_v7_orderflow_signal_diagnostic.EXPECTED_RETAINED", 2)
    result = v7_zero_boundary_groups(
        (
            _group_row("0.1", "1", "W1"),
            _group_row("0.2", "-1", "W2"),
            _group_row("0", "1", "W1"),
            _group_row("-0.1", "-1", "W2"),
        )
    )
    assert result["retained_qimb_0_gt_0"]["trades"] == 2
    assert result["filtered_qimb_0_lte_0"]["trades"] == 2


def test_frozen_v6_reproduction_guard_uses_exact_443_ledger() -> None:
    rows = load_frozen_v6_ledger(
        "reports/diagnostics/v6_signal_quality/e1eef7bdd0c37ad4/trade_ledger.csv"
    )
    assert len(rows) == 443


def test_holdout_integrity_rejects_contamination() -> None:
    manifest = _load_json("data/orderflow/aggregated/15m/BTCUSDC/dataset_manifest.json")
    contaminated = deepcopy(manifest)
    contaminated["blind_holdout"]["loaded"] = True
    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v7_dataset_manifest(contaminated)


def test_deterministic_report_serialization(tmp_path) -> None:
    assert DIAGNOSTIC_ID == "8107260358f28fca"
    output = tmp_path / DIAGNOSTIC_ID
    rows = ({"name": "x", "value": Decimal("1.25")},)
    paths = write_reports(
        output=output,
        summary={"diagnostic_id": DIAGNOSTIC_ID},
        ledger=rows,
        comparisons=rows,
        consistency=rows,
        categories=rows,
    )
    assert all(path.is_file() for path in paths)
    assert '"diagnostic_id"' in paths[0].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="exists"):
        write_reports(
            output=output,
            summary={},
            ledger=rows,
            comparisons=rows,
            consistency=rows,
            categories=rows,
        )
