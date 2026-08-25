"""Execute the one-time preregistered independent V8-H2 validation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.backtest.models import BacktestConfig
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v6_h0_replay import classify_stop_source
from src.cli.run_v8_h0_replay import doubled_cost_stress_records
from src.derivatives.integrity import CoverageClassification, analyze_coverage
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    RNormalizedTrade,
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.historical.dataset import HistoricalDataset
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.market.intervals import next_open_time
from src.models.market_data_source import MarketDataSource
from src.research.evaluation import evaluate_strategy_period
from src.research.v6_mtf_continuation_preregistration import (
    build_manifest as build_v6_manifest,
)
from src.research.v8_h2_validation_dataset import (
    TimeRange,
    _in_range,
    _parse_spot_archive,
    archive_plan,
)
from src.research.v8_h2_validation_preregistration import (
    CHRONOLOGICAL_BLOCK_COUNT,
    FROZEN_V6_CANDIDATE_COUNT,
    MINIMUM_RETAINED_TRADES,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    SIGNIFICANCE_ALPHA,
    SupportMetrics,
    VALIDATION_DATASET_ID,
    VALIDATION_END,
    VALIDATION_START,
    build_manifest,
    chronological_candidate_blocks,
    h2_premium_filter_accepts,
    progression_gate_passes,
    require_h2_subset_of_frozen_v6,
    support_gate_passes,
    within_block_permutations,
)
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v6_mtf_continuation import (
    V6MTFContinuationStrategy,
    prepare_v6_context,
)
from src.strategy.v8_funding_schedule import V8FundingScheduleStrategy
from src.orderflow.integrity import verify_sha256
from src.utils.logger import configure_logging, get_logger


EXPECTED_RUN_ID = "b334dac73e746982"
UTC = timezone.utc
INTERVAL = timedelta(minutes=15)


@dataclass(frozen=True, slots=True)
class FrozenValidationCandidate:
    candidate_id: str
    signal_close: datetime
    signal_candle_timestamp: datetime
    atr14_4h: Decimal
    block_number: int


@dataclass(frozen=True, slots=True)
class PremiumObservation:
    bucket_close: datetime
    mark_price: Decimal | None
    mark_timestamp: datetime | None
    index_price: Decimal | None
    index_timestamp: datetime | None
    derived_premium: Decimal | None


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    candidate_id: str
    signal_close_utc: str
    block_number: int
    status: str
    derived_mark_index_premium: Decimal | None


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate_id: str
    signal_close_utc: str
    block_number: int
    h2_retained: bool
    frictionless_r: Decimal
    net_r: Decimal
    fee_cost_r: Decimal
    slippage_cost_r: Decimal
    total_friction_r: Decimal
    exit_reason: str
    bars_held: int


@dataclass(frozen=True, slots=True)
class BlockResult:
    block_number: int
    v6_candidates: int
    h2_retained: int
    v6_gross_expectancy_r: Decimal
    h2_gross_expectancy_r: Decimal
    gross_better: bool
    h2_net_result_r: Decimal
    positive_net: bool


@dataclass(frozen=True, slots=True)
class PermutationAudit:
    observed_statistic: Decimal
    exact_conditional_null_mean: Decimal
    centered_observed_statistic: Decimal
    simulated_null_mean: Decimal
    simulated_null_minimum: Decimal
    simulated_null_p05: Decimal
    simulated_null_median: Decimal
    simulated_null_p95: Decimal
    simulated_null_maximum: Decimal
    upper_tail_extreme_count: int
    p_value: Decimal


def _utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V8-H2 timestamps must be timezone-aware.")
    return parsed.astimezone(UTC)


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required V8-H2 file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_preregistration(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID or payload != build_manifest():
        raise ValueError("V8-H2 preregistration differs from the frozen manifest.")


def validate_dataset_manifest(payload: dict) -> None:
    expected_range = {"start": VALIDATION_START, "end": VALIDATION_END}
    coverage = payload.get("coverage", {})
    if (
        payload.get("dataset_id") != VALIDATION_DATASET_ID
        or payload.get("classification") != "VALIDATION_DATA_READY"
        or payload.get("selected_interval") != expected_range
        or payload.get("candidate_count") != FROZEN_V6_CANDIDATE_COUNT
        or payload.get("future_data_violations") != 0
        or payload.get("discovery_overlap") != 0
        or payload.get("blind_holdout_overlap") != 0
        or payload.get("derived_mark_index_premium_available") != 20124
    ):
        raise ValueError("V8-H2 validation dataset identity or integrity changed.")
    for source in ("spot", "mark", "index"):
        item = coverage.get(source, {})
        if (
            item.get("classification") != CoverageClassification.COMPLETE.value
            or item.get("rows") != 20124
            or item.get("missing_intervals") != 0
            or item.get("duplicate_timestamps") != 0
            or item.get("duplicate_exact_records") != 0
        ):
            raise ValueError(f"V8-H2 {source} coverage integrity changed.")
    if payload.get("blind_holdout_exclusion") != {
        "start": "2025-08-01T00:00:00+00:00",
        "end": "2026-02-01T00:00:00+00:00",
    }:
        raise ValueError("V8-H2 blind holdout exclusion changed.")


def validate_execution_config(
    *, preregistration: dict, strategy_config: TrendMomentumConfig,
    backtest_config: BacktestConfig,
) -> None:
    frozen_v6 = build_v6_manifest()
    risk = frozen_v6["volatility_and_initial_risk"]
    exits = frozen_v6["risk_and_exit"]
    costs = preregistration["execution_risk_and_costs"]
    if (
        backtest_config.fee_bps != Decimal(costs["base_costs"]["fee_bps_per_side"])
        or backtest_config.slippage_bps != Decimal(costs["base_costs"]["slippage_bps_per_side"])
        or V8FundingScheduleStrategy.REWARD_RISK_RATIO != Decimal(exits["reward_risk_ratio"])
        or V8FundingScheduleStrategy.MIN_STOP_DISTANCE_FRACTION
        != Decimal(risk["minimum_stop_distance_entry_fraction"])
        or V8FundingScheduleStrategy.MAXIMUM_HOLD_BARS != exits["maximum_hold_bars"]
        or V8FundingScheduleStrategy.COOLDOWN_BARS != exits["cooldown_bars"]
        or costs["entry_execution"] != "FROZEN_V6_H0_NEXT_BAR_OPEN"
        or costs["ambiguous_bar_policy"] != "STOP_FIRST"
        or strategy_config != TrendMomentumConfig()
    ):
        raise ValueError("Frozen V6 execution, risk, exit, or costs changed.")


def load_spot_dataset(*, dataset_manifest: dict, raw_root: str | Path) -> HistoricalDataset:
    selected = TimeRange(_utc(VALIDATION_START), _utc(VALIDATION_END))
    candles = []
    for item in archive_plan(selected, raw_root=raw_root):
        if item.source != "spot_klines":
            continue
        verify_sha256(item.location.destination, item.location.checksum_destination)
        candles.extend(_parse_spot_archive(item.location.destination))
    selected_candles = _in_range(candles, selected, "timestamp")
    coverage = analyze_coverage(
        selected_candles, source="BTCUSDC_SPOT_15M", timestamp_field="timestamp",
        expected_step=INTERVAL, expected_count=20124,
    )
    if coverage.classification is not CoverageClassification.COMPLETE:
        raise ValueError("V8-H2 local Spot source is not exact and continuous.")
    if len(selected_candles) != dataset_manifest["coverage"]["spot"]["rows"]:
        raise ValueError("V8-H2 local Spot rows differ from the dataset manifest.")
    return HistoricalDataset(
        symbol="BTCUSDC", interval="15m", source=MarketDataSource.BINANCE_PUBLIC,
        candles=selected_candles,
    )


def load_premium_context(path: str | Path, *, spot: HistoricalDataset) -> dict[datetime, PremiumObservation]:
    rows: dict[datetime, PremiumObservation] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            bucket_open = _utc(raw["bucket_open_time"])
            bucket_close = _utc(raw["bucket_close_time"])
            mark_timestamp = _utc(raw["mark_price_timestamp"]) if raw["mark_price_timestamp"] else None
            index_timestamp = _utc(raw["index_price_timestamp"]) if raw["index_price_timestamp"] else None
            mark = Decimal(raw["mark_price"]) if raw["mark_price"] else None
            index = Decimal(raw["index_price"]) if raw["index_price"] else None
            premium = Decimal(raw["derived_mark_index_premium"]) if raw["derived_mark_index_premium"] else None
            if (
                bucket_close != bucket_open + INTERVAL
                or bucket_close in rows
                or mark is None or index is None or premium != (mark - index) / index
                or mark_timestamp is None or index_timestamp is None
                or mark_timestamp > bucket_close or index_timestamp > bucket_close
            ):
                raise ValueError("V8-H2 context formula, uniqueness, or causality changed.")
            rows[bucket_close] = PremiumObservation(
                bucket_close, mark, mark_timestamp, index, index_timestamp, premium
            )
    expected_closes = tuple(next_open_time(candle.timestamp, "15m") for candle in spot.candles)
    if tuple(rows) != expected_closes:
        raise ValueError("V8-H2 context does not align exactly to Spot bucket closes.")
    return rows


def build_frozen_candidate_schedule(spot: HistoricalDataset) -> tuple[FrozenValidationCandidate, ...]:
    prepared = prepare_v6_context(spot.candles)
    strategy = V6MTFContinuationStrategy(TrendMomentumConfig(), prepared_context=prepared)
    raw: list[tuple[datetime, datetime, Decimal]] = []
    for index, candle in enumerate(spot.candles):
        if index < strategy.required_history_bars:
            continue
        current = prepared.indicators_15m_by_timestamp[candle.timestamp]
        previous = prepared.indicators_15m_by_timestamp.get(spot.candles[index - 1].timestamp)
        signal_close = next_open_time(candle.timestamp, spot.interval)
        decision = strategy.evaluate(StrategyContext(
            timestamp=signal_close, current_candle=candle,
            recent_history=spot.candles[max(0, index - 20) : index + 1],
            indicators=current, previous_indicators=previous,
            has_position=False, bars_in_position=0, equity=Decimal("1000"),
            cash_usdc=Decimal("1000"), completed_trade_count=0,
            bars_since_exit=None, entry_fee_rate=Decimal("0.001"),
        ))
        if decision.action is StrategyAction.ENTER_LONG:
            assert decision.stop_distance is not None
            raw.append((signal_close, candle.timestamp, decision.stop_distance))
    if len(raw) != FROZEN_V6_CANDIDATE_COUNT:
        raise ValueError(
            f"Frozen V6 validation schedule changed: expected 291, got {len(raw)}."
        )
    timestamp_blocks = chronological_candidate_blocks(tuple(item[0] for item in raw))
    block_by_timestamp = {
        timestamp: block_number
        for block_number, block in enumerate(timestamp_blocks, start=1)
        for timestamp in block
    }
    return tuple(
        FrozenValidationCandidate(
            candidate_id=f"C{number:04d}", signal_close=signal_close,
            signal_candle_timestamp=signal_candle, atr14_4h=atr,
            block_number=block_by_timestamp[signal_close],
        )
        for number, (signal_close, signal_candle, atr) in enumerate(raw, start=1)
    )


def filter_candidates(
    candidates: tuple[FrozenValidationCandidate, ...],
    context: dict[datetime, PremiumObservation],
) -> tuple[tuple[FrozenValidationCandidate, ...], tuple[CandidateDecision, ...]]:
    retained = []
    decisions = []
    for candidate in candidates:
        observation = context.get(candidate.signal_close)
        accepted = bool(
            observation
            and h2_premium_filter_accepts(
                mark_price=observation.mark_price,
                mark_timestamp=observation.mark_timestamp,
                index_price=observation.index_price,
                index_timestamp=observation.index_timestamp,
                signal_close=candidate.signal_close,
            )
        )
        if accepted:
            retained.append(candidate)
        decisions.append(CandidateDecision(
            candidate_id=candidate.candidate_id,
            signal_close_utc=candidate.signal_close.isoformat(),
            block_number=candidate.block_number,
            status="RETAINED_NEGATIVE_PREMIUM" if accepted else "FILTERED_NONNEGATIVE_OR_MISSING_PREMIUM",
            derived_mark_index_premium=(observation.derived_premium if observation else None),
        ))
    require_h2_subset_of_frozen_v6(
        (item.candidate_id for item in candidates),
        (item.candidate_id for item in retained),
    )
    return tuple(retained), tuple(decisions)


def evaluate_frozen_candidate(
    *, candidate: FrozenValidationCandidate, spot: HistoricalDataset,
    strategy_config: TrendMomentumConfig, backtest_config: BacktestConfig,
) -> tuple[RNormalizedTrade, str, int]:
    by_timestamp = {candle.timestamp: index for index, candle in enumerate(spot.candles)}
    try:
        signal_index = by_timestamp[candidate.signal_candle_timestamp]
    except KeyError as exc:
        raise ValueError("Frozen V6 signal candle is absent from validation Spot data.") from exc
    start = max(0, signal_index - 60)
    end = min(len(spot.candles), signal_index + 98)
    local = HistoricalDataset(
        symbol=spot.symbol, interval=spot.interval, source=spot.source,
        candles=spot.candles[start:end],
    )
    strategy = V8FundingScheduleStrategy(
        strategy_config,
        retained_atr_by_signal_close={candidate.signal_close: candidate.atr14_4h},
    )
    evaluation = evaluate_strategy_period(
        local, strategy=strategy, strategy_config=strategy_config,
        backtest_config=backtest_config, evaluation_start_index=signal_index - start,
    )
    if (
        strategy.scheduled_candidate_conflict_timestamps
        or strategy.emitted_candidate_timestamps != frozenset((candidate.signal_close,))
        or len(evaluation.backtest.trades) != 1
    ):
        raise ValueError(f"Frozen candidate {candidate.candidate_id} did not execute exactly once.")
    trade = evaluation.backtest.trades[0]
    if trade.entry_signal_time.astimezone(UTC) != candidate.signal_close:
        raise ValueError("Frozen candidate entry signal time changed.")
    stop = classify_stop_source(
        trade_id=trade.trade_id, entry_signal_time=trade.entry_signal_time,
        entry_price=trade.entry_price, atr_distance=candidate.atr14_4h,
        minimum_stop_distance_fraction=V8FundingScheduleStrategy.MIN_STOP_DISTANCE_FRACTION,
    )
    record = build_r_normalized_trades(
        trades=(trade,),
        actual_stop_risk_by_trade_id={trade.trade_id: trade.quantity * stop.initial_stop_distance},
        slippage_bps=backtest_config.slippage_bps,
    )[0]
    return replace(record, trade_id=candidate.candidate_id), trade.exit_reason.value, trade.bars_held


def build_block_results(
    candidates: tuple[FrozenValidationCandidate, ...],
    retained_ids: frozenset[str], records: dict[str, RNormalizedTrade],
) -> tuple[BlockResult, ...]:
    output = []
    for number in range(1, CHRONOLOGICAL_BLOCK_COUNT + 1):
        block = tuple(item for item in candidates if item.block_number == number)
        v6 = tuple(records[item.candidate_id] for item in block)
        h2 = tuple(records[item.candidate_id] for item in block if item.candidate_id in retained_ids)
        v6_gross = summarize_r_normalized_trades(v6).frictionless_expectancy_r
        h2_gross = summarize_r_normalized_trades(h2).frictionless_expectancy_r
        net_result = sum((item.net_r for item in h2), start=Decimal("0"))
        output.append(BlockResult(
            block_number=number, v6_candidates=len(v6), h2_retained=len(h2),
            v6_gross_expectancy_r=v6_gross, h2_gross_expectancy_r=h2_gross,
            gross_better=bool(h2 and h2_gross > v6_gross),
            h2_net_result_r=net_result, positive_net=bool(h2 and net_result > 0),
        ))
    return tuple(output)


def permutation_test_audit(
    *, candidates: tuple[FrozenValidationCandidate, ...],
    retained_ids: frozenset[str], records: dict[str, RNormalizedTrade],
) -> PermutationAudit:
    all_values = tuple(records[item.candidate_id].frictionless_r for item in candidates)
    selected_values = tuple(
        records[item.candidate_id].frictionless_r
        for item in candidates if item.candidate_id in retained_ids
    )
    if not selected_values:
        raise ValueError("V8-H2 permutation test requires retained candidates.")
    baseline = sum(all_values, start=Decimal("0")) / Decimal(len(all_values))
    observed = sum(selected_values, start=Decimal("0")) / Decimal(len(selected_values)) - baseline
    value_blocks = tuple(
        tuple(records[item.candidate_id].frictionless_r for item in candidates if item.block_number == number)
        for number in range(1, CHRONOLOGICAL_BLOCK_COUNT + 1)
    )
    mask_blocks = tuple(
        tuple(item.candidate_id in retained_ids for item in candidates if item.block_number == number)
        for number in range(1, CHRONOLOGICAL_BLOCK_COUNT + 1)
    )
    expected_selected_sum = sum(
        Decimal(sum(block_mask))
        * (sum(block_values, start=Decimal("0")) / Decimal(len(block_values)))
        for block_values, block_mask in zip(value_blocks, mask_blocks, strict=True)
    )
    exact_null_mean = expected_selected_sum / Decimal(len(selected_values)) - baseline
    null_statistics = []
    for permuted in within_block_permutations(
        value_blocks, count=PERMUTATION_COUNT, seed=PERMUTATION_SEED
    ):
        selected = tuple(
            value
            for block_values, block_mask in zip(permuted, mask_blocks, strict=True)
            for value, keep in zip(block_values, block_mask, strict=True)
            if keep
        )
        statistic = sum(selected, start=Decimal("0")) / Decimal(len(selected)) - baseline
        null_statistics.append(statistic)
    ordered = sorted(null_statistics)
    extreme = sum(statistic >= observed for statistic in null_statistics)
    return PermutationAudit(
        observed_statistic=observed,
        exact_conditional_null_mean=exact_null_mean,
        centered_observed_statistic=observed - exact_null_mean,
        simulated_null_mean=(
            sum(null_statistics, start=Decimal("0")) / Decimal(PERMUTATION_COUNT)
        ),
        simulated_null_minimum=ordered[0],
        simulated_null_p05=ordered[int(Decimal("0.05") * (PERMUTATION_COUNT - 1))],
        simulated_null_median=ordered[(PERMUTATION_COUNT - 1) // 2],
        simulated_null_p95=ordered[int(Decimal("0.95") * (PERMUTATION_COUNT - 1))],
        simulated_null_maximum=ordered[-1],
        upper_tail_extreme_count=extreme,
        p_value=Decimal(1 + extreme) / Decimal(1 + PERMUTATION_COUNT),
    )


def permutation_p_value(
    *, candidates: tuple[FrozenValidationCandidate, ...],
    retained_ids: frozenset[str], records: dict[str, RNormalizedTrade],
) -> Decimal:
    return permutation_test_audit(
        candidates=candidates, retained_ids=retained_ids, records=records
    ).p_value


def _gate_details(metrics: SupportMetrics) -> tuple[dict, dict, str]:
    support_conditions = {
        "subset_integrity": metrics.subset_integrity,
        "retained_trades_at_least_50": metrics.retained_trades >= MINIMUM_RETAINED_TRADES,
        "gross_expectancy_beats_v6": metrics.h2_gross_expectancy_r > metrics.v6_gross_expectancy_r,
        "net_expectancy_beats_v6": metrics.h2_net_expectancy_r > metrics.v6_net_expectancy_r,
        "profit_factor_beats_v6": metrics.h2_profit_factor_r > metrics.v6_profit_factor_r,
        "gross_better_blocks_at_least_5": metrics.gross_improved_blocks >= 5,
        "permutation_p_value_below_0_05": metrics.selection_advantage_permutation_p_value < SIGNIFICANCE_ALPHA,
    }
    support = support_gate_passes(metrics)
    progression_conditions = {
        "support_pass": support,
        "net_expectancy_positive": metrics.h2_net_expectancy_r > 0,
        "profit_factor_above_1": metrics.h2_profit_factor_r > 1,
    }
    progression = progression_gate_passes(metrics)
    classification = "PROGRESSION_ELIGIBLE" if progression else (
        "SUPPORTED_NOT_PROFITABLE" if support else "NOT_SUPPORTED"
    )
    return (
        {"conditions": support_conditions, "all_conditions_met": support},
        {"conditions": progression_conditions, "all_conditions_met": progression},
        classification,
    )


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError(f"V8-H2 report rows are empty: {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(asdict(row)))


def write_reports(
    *, output: Path, summary: dict, decisions: tuple[CandidateDecision, ...],
    results: tuple[CandidateResult, ...], blocks: tuple[BlockResult, ...],
) -> Path:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("V8-H2 validation result already exists; refusing overwrite or rerun.")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "candidate_decisions.csv", decisions)
    _write_csv(output / "candidate_results.csv", results)
    _write_csv(output / "block_results.csv", blocks)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", default="research/v8_h2_validation_preregistration/b334dac73e746982/manifest.json")
    parser.add_argument("--dataset-manifest", default="research/v8_h2_validation/52302ca4385f5240/manifest.json")
    parser.add_argument("--context", default="research/v8_h2_validation/52302ca4385f5240/context.csv")
    parser.add_argument("--raw-root", default="data/v8_h2_validation/raw")
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument("--output-root", default="reports/v8_h2_validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V8-H2 validation not executed. Explicit --execute is required.")
        return 1
    try:
        preregistration = _load_json(args.manifest)
        dataset_manifest = _load_json(args.dataset_manifest)
        validate_preregistration(preregistration)
        validate_dataset_manifest(dataset_manifest)
        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError("V8-H2 validation result already exists; refusing overwrite or rerun.")
        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        if (
            root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
            or root_manifest.blind_holdout is None
            or root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")
        strategy_config = TrendMomentumConfig()
        backtest_config = BacktestConfig()
        validate_execution_config(
            preregistration=preregistration, strategy_config=strategy_config,
            backtest_config=backtest_config,
        )
        spot = load_spot_dataset(dataset_manifest=dataset_manifest, raw_root=args.raw_root)
        context = load_premium_context(args.context, spot=spot)
        candidates = build_frozen_candidate_schedule(spot)
        retained, decisions = filter_candidates(candidates, context)
        retained_ids = frozenset(item.candidate_id for item in retained)
        records: dict[str, RNormalizedTrade] = {}
        results = []
        for candidate in candidates:
            record, exit_reason, bars_held = evaluate_frozen_candidate(
                candidate=candidate, spot=spot, strategy_config=strategy_config,
                backtest_config=backtest_config,
            )
            records[candidate.candidate_id] = record
            results.append(CandidateResult(
                candidate_id=candidate.candidate_id,
                signal_close_utc=candidate.signal_close.isoformat(),
                block_number=candidate.block_number,
                h2_retained=candidate.candidate_id in retained_ids,
                frictionless_r=record.frictionless_r, net_r=record.net_r,
                fee_cost_r=record.fee_cost_r, slippage_cost_r=record.slippage_cost_r,
                total_friction_r=record.total_friction_r,
                exit_reason=exit_reason, bars_held=bars_held,
            ))
        if len(records) != FROZEN_V6_CANDIDATE_COUNT:
            raise ValueError("V8-H2 did not produce one frozen V6 outcome per candidate.")
        v6_records = tuple(records[item.candidate_id] for item in candidates)
        h2_records = tuple(records[item.candidate_id] for item in retained)
        v6_summary = summarize_r_normalized_trades(v6_records)
        h2_summary = summarize_r_normalized_trades(h2_records)
        stress_summary = summarize_r_normalized_trades(doubled_cost_stress_records(h2_records))
        blocks = build_block_results(candidates, retained_ids, records)
        permutation = permutation_test_audit(
            candidates=candidates, retained_ids=retained_ids, records=records
        )
        p_value = permutation.p_value
        if v6_summary.profit_factor_r is None or h2_summary.profit_factor_r is None:
            raise ValueError("V8-H2 profit factor is undefined; frozen gates cannot pass.")
        metrics = SupportMetrics(
            subset_integrity=True, retained_trades=len(retained),
            h2_gross_expectancy_r=h2_summary.frictionless_expectancy_r,
            v6_gross_expectancy_r=v6_summary.frictionless_expectancy_r,
            h2_net_expectancy_r=h2_summary.net_expectancy_r,
            v6_net_expectancy_r=v6_summary.net_expectancy_r,
            h2_profit_factor_r=h2_summary.profit_factor_r,
            v6_profit_factor_r=v6_summary.profit_factor_r,
            gross_improved_blocks=sum(item.gross_better for item in blocks),
            selection_advantage_permutation_p_value=p_value,
        )
        support, progression, classification = _gate_details(metrics)
        summary = {
            "run_id": EXPECTED_RUN_ID,
            "validation_dataset_id": VALIDATION_DATASET_ID,
            "validation_window": {"start": VALIDATION_START, "end": VALIDATION_END},
            "frozen_rule": "(MARK_PRICE_MINUS_INDEX_PRICE)_DIVIDED_BY_INDEX_PRICE_STRICTLY_LESS_THAN_ZERO",
            "v6_candidate_count": len(candidates),
            "h2_retained_count": len(retained),
            "h2_filtered_count": len(candidates) - len(retained),
            "retention_ratio": Decimal(len(retained)) / Decimal(len(candidates)),
            "non_v6_entry_count": 0,
            "v6_metrics": asdict(v6_summary),
            "h2_metrics": asdict(h2_summary),
            "gross_better_chronological_blocks": sum(item.gross_better for item in blocks),
            "positive_net_blocks": sum(item.positive_net for item in blocks),
            "permutation_test": {
                "count": PERMUTATION_COUNT, "seed": PERMUTATION_SEED,
                "statistic": preregistration["permutation_test"]["statistic"],
                "operational_statistic": "MEAN_RETAINED_GROSS_R_MINUS_MEAN_ALL_V6_GROSS_R",
                "alternative": preregistration["permutation_test"]["alternative"],
                "null": preregistration["permutation_test"]["null"],
                "observed_statistic": permutation.observed_statistic,
                "observed_statistic_sign": (
                    "POSITIVE" if permutation.observed_statistic > 0 else
                    "NEGATIVE" if permutation.observed_statistic < 0 else "ZERO"
                ),
                "exact_conditional_null_mean": permutation.exact_conditional_null_mean,
                "centered_observed_statistic": permutation.centered_observed_statistic,
                "simulated_null_summary": {
                    "mean": permutation.simulated_null_mean,
                    "minimum": permutation.simulated_null_minimum,
                    "p05": permutation.simulated_null_p05,
                    "median": permutation.simulated_null_median,
                    "p95": permutation.simulated_null_p95,
                    "maximum": permutation.simulated_null_maximum,
                },
                "upper_tail_extreme_count": permutation.upper_tail_extreme_count,
                "finite_sample_correction": "(EXTREME_COUNT_PLUS_1)_DIVIDED_BY_(PERMUTATION_COUNT_PLUS_1)",
                "interpretation": "CONDITIONAL_WITHIN_BLOCK_SELECTION_ASSOCIATION_NOT_AGGREGATE_GROSS_OUTPERFORMANCE",
                "p_value": p_value,
            },
            "doubled_cost_stress": {
                "net_expectancy_r": stress_summary.net_expectancy_r,
                "profit_factor_r": stress_summary.profit_factor_r,
                "trade_count": stress_summary.trade_count,
                "descriptive_only": True,
            },
            "support_gate": support,
            "progression_gate": progression,
            "final_classification": classification,
            "execution_integrity": {
                "candidate_outcomes_computed_once_from_frozen_v6_schedule": True,
                "h2_reuses_frozen_v6_candidate_outcomes": True,
                "h2_account_state_replay": False,
                "next_bar_open": True, "stop_first": True,
                "maximum_hold_bars": 96, "cooldown_frozen_in_v6_schedule": True,
                "non_v6_entries": 0,
            },
            "blind_holdout": preregistration["data_policy"]["blind_holdout"],
            "alternative_variants_inspected": False,
        }
        path = write_reports(
            output=output, summary=summary, decisions=decisions,
            results=tuple(results), blocks=blocks,
        )
        logger.info("V8-H2 classification: %s", classification)
        logger.info("Report: %s", path.resolve())
        logger.warning("Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | NOT CONSUMED | NOT EVALUATED")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V8-H2 validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
