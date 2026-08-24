"""Execute the preregistered V8 derivatives discovery diagnostic once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import _build_regions, _read_reference_windows
from src.cli.run_v6_h0_replay import (
    expand_complete_region_history,
    validate_eligible_windows,
)
from src.cli.run_v8_h0_replay import (
    _load_json,
    load_frozen_v6_reference,
    validate_derivatives_manifest,
)
from src.config.settings import load_settings
from src.derivatives.models import FuturesMetricsRecord
from src.derivatives.parser import parse_metrics_archive
from src.derivatives.sources import build_archive_location
from src.derivatives.models import DerivativesSource
from src.diagnostics.v8_derivatives_discovery import (
    PreparedFold,
    classify_discovery,
    cohens_d,
    fit_prepared_fold,
    mean_squared_error,
    prepare_fold,
    spearman,
    within_window_permutations,
)
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.research.multiregime.windows import construct_windows
from src.research.v8_derivatives_discovery_preregistration import (
    ELIGIBLE_WINDOWS,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    RIDGE_ALPHA,
    V6_CANDIDATE_COUNT,
    build_manifest,
)
from src.research.v8_funding_context_preregistration import (
    build_manifest as build_v8_h0_manifest,
)
from src.utils.logger import configure_logging, get_logger


EXPECTED_RUN_ID = "d15c2c2b5bafefd2"
HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)
METRICS_MAX_AGE = timedelta(minutes=5)
HORIZONS = (15, 60, 240)


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    window_id: str
    trade_id: str
    signal_close: datetime
    gross_r: float
    net_r: float


@dataclass(frozen=True, slots=True)
class ContextRow:
    bucket_close: datetime
    funding_rate: float | None
    funding_timestamp: datetime | None
    open_interest: float | None
    open_interest_timestamp: datetime | None
    mark_price: float | None
    mark_timestamp: datetime | None
    index_price: float | None
    index_timestamp: datetime | None
    premium_index: float | None
    premium_timestamp: datetime | None
    derived_premium: float | None


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    candidate: CandidateOutcome
    features: tuple[float | None, ...]


def _utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V8 discovery timestamps must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _optional_float(value: str) -> float | None:
    text = value.strip()
    return float(Decimal(text)) if text else None


def _optional_timestamp(value: str) -> datetime | None:
    text = value.strip()
    return _utc(text) if text else None


def validate_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V8 discovery preregistration ID.")
    if payload != build_manifest():
        raise ValueError("V8 discovery manifest differs from the frozen definition.")


def load_candidate_outcomes(
    *,
    ledger_path: str | Path,
    v6_summary_path: str | Path,
    v6_windows_path: str | Path,
) -> tuple[CandidateOutcome, ...]:
    _, _, frozen_candidates = load_frozen_v6_reference(
        summary_path=v6_summary_path,
        windows_path=v6_windows_path,
        schedule_path=ledger_path,
        manifest=build_v8_h0_manifest(),
    )
    with Path(ledger_path).open("r", encoding="utf-8", newline="") as stream:
        raw = tuple(csv.DictReader(stream))
    rows = tuple(
        CandidateOutcome(
            window_id=row["window_id"],
            trade_id=row["trade_id"],
            signal_close=_utc(row["entry_signal_time"]),
            gross_r=float(Decimal(row["frictionless_r"])),
            net_r=float(Decimal(row["net_r"])),
        )
        for row in raw
    )
    frozen_keys = tuple((row.window_id, row.signal_close) for row in frozen_candidates)
    keys = tuple((row.window_id, row.signal_close) for row in rows)
    if (
        len(rows) != V6_CANDIDATE_COUNT
        or keys != frozen_keys
        or len(set(keys)) != len(keys)
        or tuple(dict.fromkeys(row.window_id for row in rows)) != ELIGIBLE_WINDOWS
        or any(HOLDOUT_START <= row.signal_close < HOLDOUT_END for row in rows)
    ):
        raise ValueError("Diagnostic universe differs from the frozen 443 V6 candidates.")
    return rows


def _required_bucket_closes(candidates: tuple[CandidateOutcome, ...]) -> frozenset[datetime]:
    return frozenset(
        candidate.signal_close - timedelta(minutes=horizon)
        for candidate in candidates
        for horizon in (0, *HORIZONS)
    )


def load_context_rows(
    dataset_manifest: dict,
    *,
    required_bucket_closes: frozenset[datetime],
) -> dict[datetime, ContextRow]:
    selected: dict[datetime, ContextRow] = {}
    seen: set[datetime] = set()
    total = 0
    timestamp_columns = (
        "funding_timestamp",
        "open_interest_timestamp",
        "mark_price_timestamp",
        "index_price_timestamp",
        "premium_index_timestamp",
    )
    for path_text in dataset_manifest["partition_paths"]:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"V8 context partition is missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            for raw in csv.DictReader(stream):
                total += 1
                bucket = _utc(raw["bucket_close_time"])
                if bucket in seen:
                    raise ValueError("V8 context contains a duplicate bucket close.")
                seen.add(bucket)
                if HOLDOUT_START <= bucket < HOLDOUT_END:
                    raise ValueError("V8 context intersects the blind holdout.")
                timestamps = {
                    name: _optional_timestamp(raw[name]) for name in timestamp_columns
                }
                if any(
                    timestamp is not None and timestamp > bucket
                    for timestamp in timestamps.values()
                ):
                    raise ValueError("V8 context contains a future observation.")
                if bucket not in required_bucket_closes:
                    continue
                selected[bucket] = ContextRow(
                    bucket_close=bucket,
                    funding_rate=_optional_float(raw["latest_known_funding_rate"]),
                    funding_timestamp=timestamps["funding_timestamp"],
                    open_interest=_optional_float(raw["open_interest"]),
                    open_interest_timestamp=timestamps["open_interest_timestamp"],
                    mark_price=_optional_float(raw["mark_price"]),
                    mark_timestamp=timestamps["mark_price_timestamp"],
                    index_price=_optional_float(raw["index_price"]),
                    index_timestamp=timestamps["index_price_timestamp"],
                    premium_index=_optional_float(raw["premium_index"]),
                    premium_timestamp=timestamps["premium_index_timestamp"],
                    derived_premium=_optional_float(raw["premium_fraction"]),
                )
    if total != dataset_manifest["context_buckets"] or set(selected) != set(
        required_bucket_closes
    ):
        raise ValueError("V8 context coverage differs from the frozen candidate horizons.")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics_ratios(
    dataset_manifest: dict,
    *,
    required_bucket_closes: frozenset[datetime],
) -> tuple[tuple[FuturesMetricsRecord, ...], tuple[datetime, ...]]:
    checksums = {
        row["period"]: row["sha256"]
        for row in dataset_manifest["archive_checksums"]
        if row["source"] == "metrics" and row["granularity"] == "daily"
    }
    required_dates = sorted({timestamp.date() for timestamp in required_bucket_closes})
    records: list[FuturesMetricsRecord] = []
    for required_date in required_dates:
        period = required_date.isoformat()
        expected_sha = checksums.get(period)
        if expected_sha is None:
            raise ValueError(f"Metrics archive is outside the frozen dataset: {period}.")
        location = build_archive_location(
            DerivativesSource.METRICS,
            required_date,
            cadence="daily",
        )
        path = Path(location.destination)
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"Metrics archive checksum mismatch: {period}.")
        parsed = parse_metrics_archive(path)
        if any(HOLDOUT_START <= row.timestamp < HOLDOUT_END for row in parsed):
            raise ValueError("Metrics archive intersects the blind holdout.")
        records.extend(parsed)
    counts = Counter(row.timestamp for row in records)
    usable = tuple(
        sorted(
            (row for row in records if counts[row.timestamp] == 1),
            key=lambda row: row.timestamp,
        )
    )
    return usable, tuple(row.timestamp for row in usable)


def _latest_metric(
    records: tuple[FuturesMetricsRecord, ...],
    timestamps: tuple[datetime, ...],
    cutoff: datetime,
) -> FuturesMetricsRecord | None:
    index = bisect_right(timestamps, cutoff) - 1
    if index < 0:
        return None
    record = records[index]
    if record.timestamp > cutoff:
        raise ValueError("Metrics lookup selected a future observation.")
    return record if cutoff - record.timestamp <= METRICS_MAX_AGE else None


def load_spot_closes(
    *,
    root_manifest,
    required_bucket_closes: frozenset[datetime],
    expansion_data_root: str | Path,
    consumed_data_root: str | Path,
    mechanism_report: str | Path,
) -> dict[datetime, float]:
    regions = _build_regions(
        root_manifest=root_manifest,
        expansion_data_root=expansion_data_root,
        consumed_data_root=consumed_data_root,
    )
    reference_windows = _read_reference_windows(Path(mechanism_report))
    if tuple(sorted(reference_windows)) != ELIGIBLE_WINDOWS:
        raise ValueError("Discovery requires the exact 11 consumed windows.")
    windows = construct_windows(regions, load_settings().multiregime_config())
    eligible = tuple(window for window in windows if window.window_id in reference_windows)
    eligible = expand_complete_region_history(eligible, regions)
    validate_eligible_windows(eligible)
    if tuple(window.window_id for window in eligible) != ELIGIBLE_WINDOWS:
        raise ValueError("Discovery window identity or order changed.")
    closes: dict[datetime, float] = {}
    for window in eligible:
        for candle in window.replay_dataset.candles:
            bucket_close = candle.timestamp + timedelta(minutes=15)
            if HOLDOUT_START <= bucket_close < HOLDOUT_END:
                raise ValueError("Spot context intersects the blind holdout.")
            if bucket_close not in required_bucket_closes:
                continue
            value = float(candle.close)
            previous = closes.get(bucket_close)
            if previous is not None and previous != value:
                raise ValueError("Spot context contains conflicting closes.")
            closes[bucket_close] = value
    if set(closes) != set(required_bucket_closes):
        raise ValueError("Spot close coverage differs from candidate horizons.")
    return closes


def _context_value(
    context: ContextRow | None,
    value: float | None,
    *timestamps: datetime | None,
) -> float | None:
    if context is None or value is None or any(timestamp is None for timestamp in timestamps):
        return None
    if any(timestamp > context.bucket_close for timestamp in timestamps if timestamp is not None):
        raise ValueError("Feature uses a future context timestamp.")
    return value


def build_feature_rows(
    *,
    candidates: tuple[CandidateOutcome, ...],
    contexts: dict[datetime, ContextRow],
    metrics: tuple[FuturesMetricsRecord, ...],
    metric_timestamps: tuple[datetime, ...],
    spot_closes: dict[datetime, float],
    feature_names: tuple[str, ...],
) -> tuple[DiagnosticRow, ...]:
    output = []
    for candidate in candidates:
        point = candidate.signal_close
        context = contexts[point]
        histories = {
            horizon: contexts[point - timedelta(minutes=horizon)]
            for horizon in HORIZONS
        }
        oi_now = _context_value(
            context, context.open_interest, context.open_interest_timestamp
        )
        oi_history = {
            horizon: _context_value(
                histories[horizon],
                histories[horizon].open_interest,
                histories[horizon].open_interest_timestamp,
            )
            for horizon in HORIZONS
        }
        metric_now = _latest_metric(metrics, metric_timestamps, point)
        metric_history = {
            horizon: _latest_metric(
                metrics, metric_timestamps, point - timedelta(minutes=horizon)
            )
            for horizon in HORIZONS
        }

        def ratio(
            record: FuturesMetricsRecord | None,
            field: str,
            cutoff: datetime,
        ) -> float | None:
            if record is None:
                return None
            if record.timestamp > cutoff:
                raise ValueError("Feature uses a future metrics timestamp.")
            value = getattr(record, field)
            return float(value) if value is not None else None

        funding = _context_value(
            context, context.funding_rate, context.funding_timestamp
        )
        funding_age = (
            (point - context.funding_timestamp).total_seconds() / 60.0
            if funding is not None and context.funding_timestamp is not None
            else None
        )
        premium_now = _context_value(
            context, context.premium_index, context.premium_timestamp
        )
        premium_history = {
            horizon: _context_value(
                histories[horizon],
                histories[horizon].premium_index,
                histories[horizon].premium_timestamp,
            )
            for horizon in (15, 60)
        }
        derived_now = _context_value(
            context,
            context.derived_premium,
            context.mark_timestamp,
            context.index_timestamp,
        )
        derived_history = {
            horizon: _context_value(
                histories[horizon],
                histories[horizon].derived_premium,
                histories[horizon].mark_timestamp,
                histories[horizon].index_timestamp,
            )
            for horizon in (15, 60)
        }
        spot_return_60 = (
            spot_closes[point] / spot_closes[point - timedelta(minutes=60)] - 1.0
        )

        values: dict[str, float | None] = {
            "FUNDING_RATE_LEVEL": funding,
            "FUNDING_AGE_MINUTES": funding_age,
            "OI_LEVEL": oi_now,
            "PREMIUM_INDEX_CLOSE_LEVEL": premium_now,
            "DERIVED_MARK_INDEX_PREMIUM_LEVEL": derived_now,
        }
        for horizon in HORIZONS:
            old = oi_history[horizon]
            change = oi_now - old if oi_now is not None and old is not None else None
            values[f"OI_CHANGE_{horizon}_MINUTES"] = change
            values[f"OI_FRACTIONAL_CHANGE_{horizon}_MINUTES"] = (
                change / old if change is not None and old not in (None, 0.0) else None
            )
        taker_now = ratio(metric_now, "sum_taker_long_short_vol_ratio", point)
        values["TAKER_LONG_SHORT_VOLUME_RATIO_LEVEL"] = taker_now
        for horizon in (15, 60):
            old = ratio(
                metric_history[horizon],
                "sum_taker_long_short_vol_ratio",
                point - timedelta(minutes=horizon),
            )
            values[f"TAKER_LONG_SHORT_VOLUME_RATIO_CHANGE_{horizon}_MINUTES"] = (
                taker_now - old if taker_now is not None and old is not None else None
            )
        for field in (
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
        ):
            name = field.upper()
            current = ratio(metric_now, field, point)
            old = ratio(
                metric_history[60], field, point - timedelta(minutes=60)
            )
            values[f"{name}_LEVEL"] = current
            values[f"{name}_CHANGE_60_MINUTES"] = (
                current - old if current is not None and old is not None else None
            )
        for horizon in (15, 60):
            old_premium = premium_history[horizon]
            old_derived = derived_history[horizon]
            values[f"PREMIUM_INDEX_CLOSE_CHANGE_{horizon}_MINUTES"] = (
                premium_now - old_premium
                if premium_now is not None and old_premium is not None
                else None
            )
            values[f"DERIVED_MARK_INDEX_PREMIUM_CHANGE_{horizon}_MINUTES"] = (
                derived_now - old_derived
                if derived_now is not None and old_derived is not None
                else None
            )
        oi_change_60 = values["OI_CHANGE_60_MINUTES"]
        derived_change_60 = values[
            "DERIVED_MARK_INDEX_PREMIUM_CHANGE_60_MINUTES"
        ]
        values["SPOT_RETURN_60M_X_OI_CHANGE_60M"] = (
            spot_return_60 * oi_change_60 if oi_change_60 is not None else None
        )
        values["SPOT_RETURN_60M_X_DERIVED_PREMIUM_CHANGE_60M"] = (
            spot_return_60 * derived_change_60
            if derived_change_60 is not None
            else None
        )
        values["SPOT_RETURN_60M_X_TAKER_FLOW_DEVIATION_FROM_1"] = (
            spot_return_60 * (taker_now - 1.0) if taker_now is not None else None
        )
        if set(values) != set(feature_names):
            raise ValueError("Implemented features differ from the frozen 27-feature family.")
        output.append(
            DiagnosticRow(
                candidate=candidate,
                features=tuple(values[name] for name in feature_names),
            )
        )
    return tuple(output)


def _sign(value: float | None) -> int:
    if value is None or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def build_univariate(
    rows: tuple[DiagnosticRow, ...], feature_names: tuple[str, ...]
) -> tuple[dict, ...]:
    output = []
    for column, feature in enumerate(feature_names):
        valid = [row for row in rows if row.features[column] is not None]
        feature_values = [float(row.features[column]) for row in valid]
        gross = [row.candidate.gross_r for row in valid]
        net = [row.candidate.net_r for row in valid]
        gross_rho = spearman(feature_values, gross)
        per_window = []
        for window_id in ELIGIBLE_WINDOWS:
            window = [row for row in valid if row.candidate.window_id == window_id]
            per_window.append(
                spearman(
                    [float(row.features[column]) for row in window],
                    [row.candidate.gross_r for row in window],
                )
            )
        aggregate_sign = _sign(gross_rho)
        eligible_signs = [_sign(value) for value in per_window if _sign(value)]
        matching = sum(sign == aggregate_sign for sign in eligible_signs)
        output.append(
            {
                "feature": feature,
                "valid_sample_count": len(valid),
                "missing_count": len(rows) - len(valid),
                "missing_rate": (len(rows) - len(valid)) / len(rows),
                "spearman_gross_r": gross_rho,
                "spearman_net_r_descriptive": spearman(feature_values, net),
                "winner_loser_cohens_d": cohens_d(
                    [
                        float(row.features[column])
                        for row in valid
                        if row.candidate.gross_r > 0.0
                    ],
                    [
                        float(row.features[column])
                        for row in valid
                        if row.candidate.gross_r <= 0.0
                    ],
                ),
                "window_sign_matches_aggregate": matching,
                "window_sign_eligible": len(eligible_signs),
            }
        )
    return tuple(output)


def _prepare_folds(
    rows: tuple[DiagnosticRow, ...],
) -> tuple[tuple[str, PreparedFold], ...]:
    features = tuple(row.features for row in rows)
    output = []
    for held_out in ELIGIBLE_WINDOWS:
        training = tuple(
            index for index, row in enumerate(rows) if row.candidate.window_id != held_out
        )
        test = tuple(
            index for index, row in enumerate(rows) if row.candidate.window_id == held_out
        )
        output.append(
            (
                held_out,
                prepare_fold(
                    features,
                    training_indices=training,
                    held_out_indices=test,
                    alpha=float(RIDGE_ALPHA),
                ),
            )
        )
    return tuple(output)


def evaluate_model(
    rows: tuple[DiagnosticRow, ...],
    feature_names: tuple[str, ...],
) -> tuple[
    tuple[dict, ...],
    tuple[dict, ...],
    tuple[dict, ...],
    tuple[PreparedFold, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    targets = tuple(row.candidate.gross_r for row in rows)
    folds = _prepare_folds(rows)
    predictions = [0.0] * len(rows)
    baselines = [0.0] * len(rows)
    coefficient_rows = []
    window_rows = []
    prediction_rows = []
    for window_id, fold in folds:
        fit = fit_prepared_fold(fold, targets)
        held_actual = [targets[index] for index in fold.held_out_indices]
        held_net = [rows[index].candidate.net_r for index in fold.held_out_indices]
        for local, index in enumerate(fold.held_out_indices):
            predictions[index] = fit.predictions[local]
            baselines[index] = fit.baseline_predictions[local]
            candidate = rows[index].candidate
            prediction_rows.append(
                {
                    "window_id": window_id,
                    "trade_id": candidate.trade_id,
                    "signal_close": candidate.signal_close,
                    "gross_r": candidate.gross_r,
                    "net_r_descriptive": candidate.net_r,
                    "held_out_prediction": fit.predictions[local],
                    "constant_training_baseline": fit.baseline_predictions[local],
                }
            )
        window_rows.append(
            {
                "window_id": window_id,
                "held_out_candidates": len(fold.held_out_indices),
                "prediction_gross_r_spearman": spearman(fit.predictions, held_actual),
                "prediction_net_r_spearman_descriptive": spearman(
                    fit.predictions, held_net
                ),
                "model_mse": mean_squared_error(held_actual, fit.predictions),
                "baseline_mse": mean_squared_error(
                    held_actual, fit.baseline_predictions
                ),
                "predictive_direction_positive": (
                    (spearman(fit.predictions, held_actual) or 0.0) > 0.0
                ),
            }
        )
        for feature, coefficient in zip(feature_names, fit.coefficients):
            coefficient_rows.append(
                {
                    "window_id": window_id,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                    "sign": _sign(coefficient),
                }
            )
    return (
        tuple(prediction_rows),
        tuple(window_rows),
        tuple(coefficient_rows),
        tuple(fold for _, fold in folds),
        tuple(predictions),
        tuple(baselines),
    )


def coefficient_stability(
    coefficient_rows: tuple[dict, ...], feature_names: tuple[str, ...]
) -> tuple[dict, ...]:
    output = []
    for feature in feature_names:
        values = [
            row["standardized_coefficient"]
            for row in coefficient_rows
            if row["feature"] == feature
        ]
        signs = [_sign(value) for value in values]
        positive = signs.count(1)
        negative = signs.count(-1)
        modal_count = max(positive, negative)
        output.append(
            {
                "feature": feature,
                "mean_standardized_coefficient": sum(values) / len(values),
                "positive_folds": positive,
                "negative_folds": negative,
                "zero_folds": signs.count(0),
                "modal_nonzero_sign": (
                    "POSITIVE" if positive >= negative and positive else "NEGATIVE" if negative else "NONE"
                ),
                "modal_nonzero_sign_folds": modal_count,
                "stable_at_7_of_11": modal_count >= 7,
            }
        )
    return tuple(output)


def permutation_null(
    *,
    rows: tuple[DiagnosticRow, ...],
    folds: tuple[PreparedFold, ...],
    observed_spearman: float | None,
) -> tuple[float, tuple[dict, ...]]:
    targets = tuple(row.candidate.gross_r for row in rows)
    window_ids = tuple(row.candidate.window_id for row in rows)
    output = []
    greater_or_equal = 0
    observed = observed_spearman if observed_spearman is not None else math.inf
    for number, source_indices in enumerate(
        within_window_permutations(
            window_ids, seed=PERMUTATION_SEED, count=PERMUTATION_COUNT
        ),
        start=1,
    ):
        permuted = tuple(targets[index] for index in source_indices)
        predictions = [0.0] * len(rows)
        for fold in folds:
            fit = fit_prepared_fold(fold, permuted)
            for local, index in enumerate(fold.held_out_indices):
                predictions[index] = fit.predictions[local]
        statistic = spearman(predictions, permuted)
        if statistic is not None and statistic >= observed:
            greater_or_equal += 1
        output.append(
            {"permutation": number, "held_out_spearman": statistic}
        )
    return (
        (1.0 + greater_or_equal) / (1.0 + PERMUTATION_COUNT),
        tuple(output),
    )


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: tuple[dict, ...]) -> None:
    if not rows:
        raise ValueError(f"Diagnostic output is empty: {path.name}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the frozen V8 derivatives discovery diagnostic."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        default=(
            "research/v8_derivatives_discovery/d15c2c2b5bafefd2/manifest.json"
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=(
            "data/derivatives/processed/15m/BTCUSDT/eec764735d270f9d/manifest.json"
        ),
    )
    parser.add_argument(
        "--v6-ledger",
        default=(
            "reports/diagnostics/v6_signal_quality/e1eef7bdd0c37ad4/trade_ledger.csv"
        ),
    )
    parser.add_argument(
        "--v6-summary",
        default="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/summary.json",
    )
    parser.add_argument(
        "--v6-windows",
        default="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/window_results.csv",
    )
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--mechanism-report", default="reports/mechanisms/bc2496aed05555b5"
    )
    parser.add_argument(
        "--expansion-data-root", default="data/historical/multiregime_expansion"
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--output-root", default="reports/v8_derivatives_discovery")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V8 discovery not executed. Explicit --execute is required.")
        return 1
    try:
        manifest = _load_json(args.manifest)
        dataset_manifest = _load_json(args.dataset_manifest)
        validate_manifest(manifest)
        validate_derivatives_manifest(dataset_manifest)
        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError("V8 discovery result exists; refusing overwrite or rerun.")

        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        if (
            root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
            or root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")

        candidates = load_candidate_outcomes(
            ledger_path=args.v6_ledger,
            v6_summary_path=args.v6_summary,
            v6_windows_path=args.v6_windows,
        )
        required = _required_bucket_closes(candidates)
        contexts = load_context_rows(
            dataset_manifest, required_bucket_closes=required
        )
        metrics, metric_timestamps = load_metrics_ratios(
            dataset_manifest, required_bucket_closes=required
        )
        spot_closes = load_spot_closes(
            root_manifest=root_manifest,
            required_bucket_closes=required,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
            mechanism_report=args.mechanism_report,
        )
        feature_names = tuple(row["name"] for row in manifest["fixed_features"])
        if len(feature_names) != 27 or len(set(feature_names)) != 27:
            raise ValueError("Manifest does not contain the frozen 27 unique features.")
        rows = build_feature_rows(
            candidates=candidates,
            contexts=contexts,
            metrics=metrics,
            metric_timestamps=metric_timestamps,
            spot_closes=spot_closes,
            feature_names=feature_names,
        )
        univariate = build_univariate(rows, feature_names)
        (
            predictions,
            windows,
            fold_coefficients,
            folds,
            prediction_values,
            baseline_values,
        ) = evaluate_model(rows, feature_names)
        stability = coefficient_stability(fold_coefficients, feature_names)
        gross_targets = tuple(row.candidate.gross_r for row in rows)
        net_targets = tuple(row.candidate.net_r for row in rows)
        aggregate_spearman = spearman(prediction_values, gross_targets)
        model_mse = mean_squared_error(gross_targets, prediction_values)
        baseline_mse = mean_squared_error(gross_targets, baseline_values)
        p_value, permutations = permutation_null(
            rows=rows, folds=folds, observed_spearman=aggregate_spearman
        )
        positive_windows = sum(row["predictive_direction_positive"] for row in windows)
        has_stable_feature = any(row["stable_at_7_of_11"] for row in stability)
        classification = classify_discovery(
            held_out_spearman=aggregate_spearman,
            permutation_p_value=p_value,
            model_mse=model_mse,
            baseline_mse=baseline_mse,
            positive_direction_windows=positive_windows,
            has_stable_economic_feature=has_stable_feature,
        )
        univariate_by_feature = {row["feature"]: row for row in univariate}
        stable_ranked = sorted(
            (row for row in stability if row["stable_at_7_of_11"]),
            key=lambda row: (
                -abs(univariate_by_feature[row["feature"]]["spearman_gross_r"] or 0.0),
                row["feature"],
            ),
        )
        unstable_ranked = sorted(
            (row for row in stability if not row["stable_at_7_of_11"]),
            key=lambda row: (
                -abs(univariate_by_feature[row["feature"]]["spearman_gross_r"] or 0.0),
                row["feature"],
            ),
        )
        missingness = tuple(
            {
                "feature": row["feature"],
                "valid_sample_count": row["valid_sample_count"],
                "missing_count": row["missing_count"],
                "missing_rate": row["missing_rate"],
            }
            for row in univariate
        )
        feature_matrix = tuple(
            {
                "window_id": row.candidate.window_id,
                "trade_id": row.candidate.trade_id,
                "signal_close": row.candidate.signal_close,
                **dict(zip(feature_names, row.features)),
            }
            for row in rows
        )
        summary = {
            "version": "8.2",
            "run_id": EXPECTED_RUN_ID,
            "candidate_count": len(rows),
            "feature_count": len(feature_names),
            "eligible_windows": list(ELIGIBLE_WINDOWS),
            "primary_target": "FRICTIONLESS_GROSS_R",
            "secondary_target": "NET_R_DESCRIPTIVE",
            "model": {
                "family": "RIDGE_LINEAR_REGRESSION",
                "alpha": float(RIDGE_ALPHA),
                "folds": 11,
                "preprocessing_fit_on_training_only": True,
            },
            "aggregate_out_of_window": {
                "prediction_gross_r_spearman": aggregate_spearman,
                "prediction_net_r_spearman_descriptive": spearman(
                    prediction_values, net_targets
                ),
                "model_mse": model_mse,
                "constant_training_baseline_mse": baseline_mse,
                "model_beats_baseline": model_mse < baseline_mse,
                "positive_direction_windows": positive_windows,
            },
            "permutation_null": {
                "seed": PERMUTATION_SEED,
                "count": PERMUTATION_COUNT,
                "one_sided_p_value": p_value,
                "alpha": 0.05,
                "significant": p_value <= 0.05,
            },
            "coefficient_stability": {
                "stable_feature_count": len(stable_ranked),
                "strongest_stable_features": [row["feature"] for row in stable_ranked[:5]],
                "strongest_unstable_features": [row["feature"] for row in unstable_ranked[:5]],
            },
            "missingness": {
                "features_with_missing_values": sum(
                    row["missing_count"] > 0 for row in missingness
                ),
                "total_missing_values": sum(row["missing_count"] for row in missingness),
                "maximum_feature_missing_rate": max(
                    row["missing_rate"] for row in missingness
                ),
            },
            "classification": classification,
            "integrity": {
                "frozen_manifest_verified": True,
                "dataset_identity_verified": True,
                "metrics_archive_checksums_verified": True,
                "future_feature_timestamp_violations": 0,
                "held_out_preprocessing_or_target_leakage": False,
                "feature_threshold_mining": False,
                "trading_rule_or_replay_optimization": False,
                "blind_holdout": manifest["dataset_policy"]["blind_holdout"],
            },
        }
        output.mkdir(parents=True, exist_ok=False)
        _write_csv(output / "univariate_features.csv", univariate)
        _write_csv(output / "heldout_predictions.csv", predictions)
        _write_csv(output / "window_results.csv", windows)
        _write_csv(output / "fold_coefficients.csv", fold_coefficients)
        _write_csv(output / "coefficient_stability.csv", stability)
        _write_csv(output / "feature_missingness.csv", missingness)
        _write_csv(output / "permutation_null.csv", permutations)
        _write_csv(output / "candidate_features.csv", feature_matrix)
        temporary = output / "summary.json.tmp"
        temporary.write_text(
            json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output / "summary.json")
        logger.info("V8 discovery classification: %s", classification)
        logger.info("Report: %s", (output / "summary.json").resolve())
        logger.warning(
            "Blind holdout: LOCKED | NOT DOWNLOADED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V8 discovery failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
