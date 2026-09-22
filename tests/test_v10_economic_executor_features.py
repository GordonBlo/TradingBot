from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from src.diagnostics.v10_economic_features import (
    FEATURES,
    HORIZONS,
    Quote,
    SamplingMachine,
    quote_economics,
    sample_session,
)
from src.research.v10_collection_preregistration import timestamp, utc
from tests.test_v10_microstructure_collector import aggtrade, diff, snapshot

START = datetime(2026, 10, 1, tzinfo=UTC)
D = Decimal


def record(kind, index, at, payload):
    return {
        "session_id": "SYNTHETIC",
        "record_type": kind,
        "record_index": index,
        "received_at_utc": timestamp(at),
        "payload": payload,
    }


def records(seconds=1540):
    yield "DEPTH", record("REST_SNAPSHOT", 0, START, snapshot())
    for i in range(seconds + 1):
        at = START + timedelta(seconds=i)
        yield (
            "DEPTH",
            record(
                "DIFF_DEPTH",
                i + 1,
                at,
                diff(101 + i, 101 + i, event_at=at, bids=[["100", str(2 + i % 4)]]),
            ),
        )
        yield (
            "AGGTRADE",
            record(
                "AGG_TRADE",
                i,
                at,
                aggtrade(i + 1, event_at=at, buyer_is_maker=bool(i % 2)),
            ),
        )


@pytest.fixture(scope="module")
def sampled():
    return sample_session(
        records(),
        session_id="SYNTHETIC",
        started_at=START,
        ended_at=START + timedelta(seconds=1540),
        max_levels=5000,
    )


def test_exact_features_common_mask_and_terminal_exclusion(sampled):
    assert len(FEATURES) == len(set(FEATURES)) == 40
    assert len(sampled.rows[0].features[:13]) == 13
    assert set(sampled.rows[0].targets) == set(HORIZONS)
    assert len(sampled.rows) == 1210
    assert sampled.rows[0].at == START + timedelta(seconds=30)
    assert sampled.rows[-1].at == START + timedelta(seconds=1239)
    assert sampled.exclusions == {"terminal_target_exclusion": 301, "warmup": 30}
    for row in sampled.rows:
        assert len(row.provenance) == 10
        assert all(
            utc(value["available_at"]) <= utc(value["query_at"])
            and utc(value["source_at"]) <= utc(value["query_at"])
            for value in row.provenance.values()
        )
        assert row.targets[300] == 0


def machine():
    value = SamplingMachine(5000)
    for kind, item in list(records(0)):
        value.accept(kind, item)
    return value


def test_source_availability_and_freshness():
    value = machine()
    assert value.quote(START - timedelta(microseconds=1)) is None
    assert value.quote(START + timedelta(seconds=1)) is not None
    assert value.quote(START + timedelta(seconds=1, microseconds=1)) is None
    value.source_at = START + timedelta(milliseconds=1)
    assert value.quote(START) is None
    value.source_at = START - timedelta(seconds=2)
    assert value.quote(START) is None


def test_snapshot_without_diff_is_not_synchronized():
    value = SamplingMachine(5000)
    value.accept("DEPTH", record("REST_SNAPSHOT", 0, START, snapshot()))
    assert value.quote(START) is None


def test_buffered_effects_only_become_available_at_bridge():
    value = SamplingMachine(5000)
    event = diff(101, 101, event_at=START, bids=[["100", "5"]])
    value.accept("DEPTH", record("DIFF_DEPTH", 0, START, event))
    assert value.quote(START) is None and not value.flows
    received = START + timedelta(milliseconds=100)
    value.accept("DEPTH", record("REST_SNAPSHOT", 1, received, snapshot()))
    assert value.quote(START) is None
    quote = value.quote(received)
    assert quote.available_at == received and quote.source_at == START
    assert value.flows[0][0] == received
    assert value.flows[0][2][0] == 3


def test_trade_sign_exact_windows_duplicates_and_interactions():
    value = machine()
    at = START + timedelta(milliseconds=500)
    payload = aggtrade(2, event_at=at, quantity="0.25", buyer_is_maker=True)
    value.accept("AGGTRADE", record("AGG_TRADE", 1, at, payload))
    value.accept("AGGTRADE", record("AGG_TRADE", 2, at, payload))
    features = dict(zip(FEATURES, value.features(at)))
    assert features["buy_count_1s"] == features["sell_count_1s"] == 1
    assert features["signed_quantity_1s"] == D("-0.125")
    assert features["signed_notional_1s"] == D("-12.56250")
    assert features["quantity_flow_imbalance_1s"] == D(-1) / 3
    assert (
        features["depth_imbalance_20_x_flow_imbalance_1s"]
        == features["depth_imbalance_20"] * features["quantity_flow_imbalance_1s"]
    )
    later = dict(zip(FEATURES, value.features(START + timedelta(seconds=1))))
    assert later["buy_count_1s"] == 0  # left-open boundary
    assert later["sell_count_1s"] == 1
    empty = dict(zip(FEATURES, value.features(START + timedelta(seconds=31))))
    assert all(empty[name] == 0 for name in FEATURES[13:34])


def test_stale_diffs_do_not_add_flow_and_resync_restarts_windows():
    value = machine()
    value.accept(
        "DEPTH",
        record(
            "DIFF_DEPTH",
            2,
            START,
            diff(101, 101, event_at=START, bids=[["100", "999"]]),
        ),
    )
    assert len(value.flows) == 1
    value.accept("DEPTH", record("RESYNC_BOUNDARY", 3, START, {}))
    assert (
        not value.flows
        and value.first_trade_at is None
        and value.synchronized_at is None
    )
    assert not value.trades[30] and value.quote(START) is None


def test_gap_and_invalid_trade_fail_closed():
    value = machine()
    with pytest.raises(ValueError, match="integrity"):
        value.accept(
            "DEPTH", record("DIFF_DEPTH", 2, START, diff(104, 104, event_at=START))
        )
    value = machine()
    with pytest.raises(ValueError, match="integrity"):
        value.accept(
            "AGGTRADE",
            record("AGG_TRADE", 1, START, aggtrade(2, event_at=START, quantity="0")),
        )


def test_noncanonical_merge_is_rejected():
    items = list(records(1))
    items[1], items[2] = items[2], items[1]
    with pytest.raises(ValueError, match="causal merge"):
        sample_session(
            items,
            session_id="SYNTHETIC",
            started_at=START,
            ended_at=START + timedelta(seconds=1540),
            max_levels=5000,
        )


def test_insufficient_rows_and_no_cross_session_fill():
    with pytest.raises(ValueError, match="1202"):
        sample_session(
            records(30),
            session_id="SYNTHETIC",
            started_at=START,
            ended_at=START + timedelta(seconds=30),
            max_levels=5000,
        )


def test_future_changes_do_not_change_past_features(sampled):
    def modified():
        for kind, item in records():
            if kind == "DEPTH" and item["record_index"] > 1000:
                item["payload"]["b"] = [["100", "900"]]
            yield kind, item

    other = sample_session(
        modified(),
        session_id="SYNTHETIC",
        started_at=START,
        ended_at=START + timedelta(seconds=1540),
        max_levels=5000,
    )
    assert [r.features for r in sampled.rows[:500]] == [
        r.features for r in other.rows[:500]
    ]
    assert sampled.rows[-1].features != other.rows[-1].features


def test_decimal_cost_decomposition_and_context_isolation():
    entry = Quote(START, START, START, 1, D("99.999"), D("100.001"))
    end = START + timedelta(seconds=300)
    exit_quote = Quote(end, end, end, 2, D("101.003"), D("101.005"))
    with localcontext() as context:
        context.prec = 8
        first = quote_economics(entry, exit_quote)
    with localcontext() as context:
        context.prec = 50
        second = quote_economics(entry, exit_quote)
        assert first == second
        for name, fee, slip in (
            ("base", D(".001"), D(".0002")),
            ("stress", D(".002"), D(".0004")),
        ):
            values = first[name]
            paid, received = entry.ask * (1 + slip), exit_quote.bid * (1 - slip)
            assert (
                values["net"]
                == 10000 * (received - paid - fee * (paid + received)) / paid
            )
            assert abs(
                values["net"]
                - (
                    values["mid_move"]
                    - values["spread"]
                    - values["adverse"]
                    - values["fees"]
                )
            ) < D("1e-45")
        assert first["stress"]["net"] < first["base"]["net"]
    with pytest.raises(ValueError, match="future"):
        quote_economics(replace(entry, available_at=end), exit_quote)
