"""Immutable preregistration for outcome-unseen V8-H2 validation only."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


VERSION = "8.2"
BASELINE_ID = "V8_H2"
VALIDATION_DATASET_ID = "52302ca4385f5240"
VALIDATION_START = "2021-09-29T09:00:00+00:00"
VALIDATION_END = "2022-04-27T00:00:00+00:00"
FROZEN_V6_RUN_ID = "e1eef7bdd0c37ad4"
FROZEN_V6_CANDIDATE_COUNT = 291
CHRONOLOGICAL_BLOCK_COUNT = 7
MINIMUM_RETAINED_TRADES = 50
PERMUTATION_COUNT = 10_000
PERMUTATION_SEED = 20260825
SIGNIFICANCE_ALPHA = Decimal("0.05")
HOLDOUT_START = "2025-08-01T00:00:00+00:00"
HOLDOUT_END = "2026-02-01T00:00:00+00:00"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc)


def derived_mark_index_premium(
    *,
    mark_price: Decimal | None,
    mark_timestamp: datetime | None,
    index_price: Decimal | None,
    index_timestamp: datetime | None,
    signal_close: datetime,
) -> Decimal | None:
    """Return only the exact causal H2 premium; missing/future input is rejected."""

    if None in (mark_price, mark_timestamp, index_price, index_timestamp):
        return None
    close = _utc(signal_close)
    if _utc(mark_timestamp) > close or _utc(index_timestamp) > close:
        return None
    if index_price <= 0:
        return None
    return (mark_price - index_price) / index_price


def h2_premium_filter_accepts(**kwargs) -> bool:
    premium = derived_mark_index_premium(**kwargs)
    return premium is not None and premium < Decimal("0")


def require_h2_subset_of_frozen_v6(
    frozen_v6_candidate_ids: Iterable[str], h2_candidate_ids: Iterable[str]
) -> None:
    if set(h2_candidate_ids) - set(frozen_v6_candidate_ids):
        raise ValueError("V8-H2 may not create non-V6 entries.")


def chronological_candidate_blocks(
    ordered_signal_closes: Sequence[datetime],
) -> tuple[tuple[datetime, ...], ...]:
    """Partition the exact frozen candidate order into seven balanced blocks."""

    if len(ordered_signal_closes) != FROZEN_V6_CANDIDATE_COUNT:
        raise ValueError("V8-H2 requires exactly 291 frozen V6 candidates.")
    timestamps = tuple(_utc(value) for value in ordered_signal_closes)
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError("Frozen V6 candidates must be strictly chronological.")
    base, remainder = divmod(len(timestamps), CHRONOLOGICAL_BLOCK_COUNT)
    blocks = []
    cursor = 0
    for number in range(CHRONOLOGICAL_BLOCK_COUNT):
        size = base + int(number < remainder)
        blocks.append(timestamps[cursor : cursor + size])
        cursor += size
    return tuple(blocks)


def within_block_permutations(
    blocks: Sequence[Sequence[object]], *, count: int = PERMUTATION_COUNT,
    seed: int = PERMUTATION_SEED,
) -> Iterator[tuple[tuple[object, ...], ...]]:
    """Frozen null: permute gross-R labels inside, never across, time blocks."""

    if len(blocks) != CHRONOLOGICAL_BLOCK_COUNT or count < 1:
        raise ValueError("V8-H2 requires seven blocks and a positive permutation count.")
    rng = random.Random(seed)
    frozen = tuple(tuple(block) for block in blocks)
    for _ in range(count):
        yield tuple(tuple(rng.sample(block, len(block))) for block in frozen)


@dataclass(frozen=True, slots=True)
class SupportMetrics:
    subset_integrity: bool
    retained_trades: int
    h2_gross_expectancy_r: Decimal
    v6_gross_expectancy_r: Decimal
    h2_net_expectancy_r: Decimal
    v6_net_expectancy_r: Decimal
    h2_profit_factor_r: Decimal
    v6_profit_factor_r: Decimal
    gross_improved_blocks: int
    selection_advantage_permutation_p_value: Decimal


def support_gate_passes(metrics: SupportMetrics) -> bool:
    return (
        metrics.subset_integrity
        and metrics.retained_trades >= MINIMUM_RETAINED_TRADES
        and metrics.h2_gross_expectancy_r > metrics.v6_gross_expectancy_r
        and metrics.h2_net_expectancy_r > metrics.v6_net_expectancy_r
        and metrics.h2_profit_factor_r > metrics.v6_profit_factor_r
        and metrics.gross_improved_blocks >= 5
        and metrics.selection_advantage_permutation_p_value < SIGNIFICANCE_ALPHA
    )


def progression_gate_passes(metrics: SupportMetrics) -> bool:
    return (
        support_gate_passes(metrics)
        and metrics.h2_net_expectancy_r > Decimal("0")
        and metrics.h2_profit_factor_r > Decimal("1")
    )


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "strategy_family": "DERIVED_MARK_INDEX_PREMIUM_FILTERED_FROZEN_V6_H0",
        "hypothesis": "Frozen V6-H0 candidates improve when causal BTCUSDT derived mark/index premium is negative.",
        "validation_dataset": {
            "dataset_id": VALIDATION_DATASET_ID,
            "classification_required": "VALIDATION_DATA_READY",
            "window_start": VALIDATION_START,
            "window_end": VALIDATION_END,
            "market": "BTCUSDC_BINANCE_PUBLIC_SPOT_15M_LONG_ONLY",
            "candidate_universe": "FROZEN_OUTCOME_UNSEEN_V6_H0_CANDIDATES",
            "candidate_count": FROZEN_V6_CANDIDATE_COUNT,
        },
        "frozen_v6_reference": {
            "run_id": FROZEN_V6_RUN_ID,
            "candidate_schedule": "FROZEN_V6_H0_SIGNAL_CANDIDATE_SCHEDULE_DIRECT",
            "must_not_reevaluate_v6_against_h2_account_state": True,
            "h2_may_only_remove_v6_candidates": True,
            "required_non_v6_entry_count": 0,
        },
        "h2_rule": {
            "formula": "(MARK_PRICE_MINUS_INDEX_PRICE)_DIVIDED_BY_INDEX_PRICE",
            "input_sources": ["BTCUSDT_MARK_PRICE_KLINES", "BTCUSDT_INDEX_PRICE_KLINES"],
            "causal_timestamp_rule": "MARK_AND_INDEX_SOURCE_TIMESTAMPS_LESS_THAN_OR_EQUAL_TO_SPOT_SIGNAL_CLOSE",
            "acceptance_rule": "DERIVED_MARK_INDEX_PREMIUM_STRICTLY_LESS_THAN_ZERO",
            "zero_qualifies": False,
            "positive_qualifies": False,
            "missing_premium_rule": "REJECT_CANDIDATE",
            "missing_or_future_mark_or_index_rule": "REJECT_CANDIDATE",
            "interpolation": False,
            "future_fill": False,
        },
        "execution_risk_and_costs": {
            "entry_execution": "FROZEN_V6_H0_NEXT_BAR_OPEN",
            "initial_stop": "FROZEN_V6_H0_ACTUAL_FILL_MINUS_MAX_OF_ATR14_4H_AND_FILL_TIMES_0.0096",
            "take_profit": "FROZEN_V6_H0_ACTUAL_FILL_PLUS_2R",
            "maximum_hold_15m_bars": 96,
            "cooldown_bars": 4,
            "ambiguous_bar_policy": "STOP_FIRST",
            "base_costs": {"fee_bps_per_side": 10, "slippage_bps_per_side": 2},
            "doubled_cost_stress": {"fee_bps_per_side": 20, "slippage_bps_per_side": 4, "descriptive_only": True},
        },
        "chronological_blocks": {
            "candidate_count": FROZEN_V6_CANDIDATE_COUNT,
            "block_count": CHRONOLOGICAL_BLOCK_COUNT,
            "assignment": "ORDERED_SIGNAL_CLOSE_TIMESTAMP_ONLY_BALANCED_CONTIGUOUS_BLOCKS",
            "block_sizes": [42, 42, 42, 42, 41, 41, 41],
            "outcomes_used_for_assignment": False,
        },
        "permutation_test": {
            "statistic": "H2_GROSS_R_SELECTION_ADVANTAGE_OVER_VALIDATION_V6",
            "null": "PERMUTE_GROSS_R_LABELS_WITHIN_EACH_CHRONOLOGICAL_BLOCK_KEEPING_SELECTION_FIXED",
            "alternative": "ONE_SIDED_H2_SELECTION_ADVANTAGE_GREATER_THAN_ZERO",
            "permutation_count": PERMUTATION_COUNT,
            "seed": PERMUTATION_SEED,
            "p_value_requirement": "STRICTLY_LESS_THAN_0.05",
        },
        "support_gate": {
            "all_conditions_required": True,
            "conditions": [
                "SUBSET_INTEGRITY_PASSES",
                "RETAINED_H2_TRADES_AT_LEAST_50",
                "H2_GROSS_EXPECTANCY_R_STRICTLY_GREATER_THAN_VALIDATION_V6_GROSS_EXPECTANCY_R",
                "H2_NET_EXPECTANCY_R_STRICTLY_GREATER_THAN_VALIDATION_V6_NET_EXPECTANCY_R",
                "H2_PROFIT_FACTOR_R_STRICTLY_GREATER_THAN_VALIDATION_V6_PROFIT_FACTOR_R",
                "H2_GROSS_EXPECTANCY_IMPROVES_IN_AT_LEAST_5_OF_7_CHRONOLOGICAL_BLOCKS",
                "ONE_SIDED_WITHIN_BLOCK_PERMUTATION_P_VALUE_STRICTLY_LESS_THAN_0.05",
            ],
        },
        "progression_gate": {
            "all_conditions_required": True,
            "requires_support_pass": True,
            "conditions": [
                "H2_NET_EXPECTANCY_R_STRICTLY_GREATER_THAN_0",
                "H2_PROFIT_FACTOR_R_STRICTLY_GREATER_THAN_1",
            ],
        },
        "data_policy": {
            "only_validation_dataset_id": VALIDATION_DATASET_ID,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT", "start": HOLDOUT_START, "end": HOLDOUT_END,
                "loaded": False, "revealed": False, "consumed": False, "evaluated": False,
            },
        },
        "prohibitions": {
            "validation_replay_or_trade_execution": False,
            "outcome_inspection_before_preregistration": False,
            "retained_vs_filtered_performance_inspection": False,
            "alternate_premium_thresholds": False,
            "parameter_or_block_tuning": False,
        },
        "research_rule": "PREREGISTRATION_ONLY_NO_VALIDATION_OUTCOME_INSPECTION",
    }


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_manifest() -> dict:
    definition = _definition()
    digest = hashlib.sha256(_canonical_json(definition).encode("utf-8")).hexdigest()
    return {"run_id": digest[:16], "definition_sha256": digest, **definition}


def write_manifest(root: str | Path = "research/v8_h2_validation_preregistration") -> Path:
    manifest = build_manifest()
    path = Path(root) / manifest["run_id"] / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("Existing V8-H2 preregistration manifest differs from frozen definition.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()
    print(f"V8-H2 VALIDATION PREREGISTRATION | Run ID: {manifest['run_id']}")
    print("PREREGISTRATION COMPLETE — NO VALIDATION OUTCOMES INSPECTED")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
