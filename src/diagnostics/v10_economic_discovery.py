"""Frozen paired Ridge/statistical engine; receives samples, never opens research files."""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from src.diagnostics.v8_derivatives_discovery import (
    PreparedFold,
    fit_prepared_fold,
    mean_squared_error,
    prepare_fold,
    spearman,
)
from src.diagnostics.v10_economic_features import FEATURES, HORIZONS, SessionSamples
from src.research.v10_collection_readiness import require

QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.975, 0.99)
THRESHOLDS = (12, 24, 36, 48)
MODELS = ("l2", "combined")


def finite(value: float | None, label: str) -> float:
    if value is None or not math.isfinite(value):
        raise ValueError(f"undefined/nonfinite required statistic: {label}")
    return float(value)


def quantiles(values: Sequence[float]) -> tuple[float, ...]:
    require(
        bool(values) and all(math.isfinite(x) for x in values), "invalid quantile input"
    )
    ordered = sorted(values)
    result = []
    for q in QUANTILES:
        position = (len(ordered) - 1) * q
        low = int(position)
        fraction = position - low
        result.append(
            ordered[low]
            + fraction * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])
        )
    return tuple(result)


def strict_prepare(features, training, test) -> PreparedFold:
    require(
        bool(features) and all(len(row) == len(features[0]) for row in features),
        "invalid feature width",
    )
    require(
        all(value is None or math.isfinite(value) for row in features for value in row),
        "nonfinite feature input",
    )
    require(
        all(
            any(features[i][j] is not None for i in training)
            for j in range(len(features[0]))
        ),
        "all-missing training feature",
    )
    fold = prepare_fold(
        features, training_indices=training, held_out_indices=test, alpha=1.0
    )
    require(
        all(
            math.isfinite(x)
            for matrix in (fold.x_training, fold.x_held_out, fold.ridge_inverse)
            for row in matrix
            for x in row
        ),
        "nonfinite preprocessing",
    )
    return fold


@dataclass(frozen=True)
class PreparedDataset:
    sessions: tuple[SessionSamples, ...]
    groups: tuple[tuple[int, ...], ...]
    features: tuple[tuple[float | None, ...], ...]
    targets: dict[int, tuple[float, ...]]
    folds: dict[str, tuple[PreparedFold, ...]]


def prepare_dataset(sessions: Sequence[SessionSamples]) -> PreparedDataset:
    require(
        len(sessions) == 8 and len({s.session_id for s in sessions}) == 8,
        "exactly eight sessions required",
    )
    require(
        all(sessions[i].ended_at <= sessions[i + 1].started_at for i in range(7)),
        "nonchronological/overlapping sessions",
    )
    groups: list[tuple[int, ...]] = []
    vectors: list[tuple[float | None, ...]] = []
    targets: dict[int, list[float]] = {h: [] for h in HORIZONS}
    for session in sessions:
        require(len(session.rows) >= 1202, "fewer than 1202 common rows")
        require(
            all(
                session.rows[i].at < session.rows[i + 1].at
                for i in range(len(session.rows) - 1)
            ),
            "nonchronological rows",
        )
        require(
            all(
                session.started_at <= row.at < session.ended_at for row in session.rows
            ),
            "row outside session",
        )
        indices = tuple(range(len(vectors), len(vectors) + len(session.rows)))
        groups.append(indices)
        for row in session.rows:
            require(
                len(row.features) == len(FEATURES)
                and set(row.targets) == set(HORIZONS)
                and set(row.economics) == set(HORIZONS),
                "unpaired features/targets",
            )
            vectors.append(
                tuple(
                    float(value) if value is not None else None
                    for value in row.features
                )
            )
            for h in HORIZONS:
                targets[h].append(finite(float(row.targets[h]), "target"))
    folds: dict[str, tuple[PreparedFold, ...]] = {}
    for model, width in (("l2", 13), ("combined", 40)):
        features = [row[:width] for row in vectors]
        folds[model] = tuple(
            strict_prepare(
                features, tuple(i for group in groups[:k] for i in group), groups[k]
            )
            for k in range(1, 8)
        )
    require(
        all(
            a.training_indices == b.training_indices
            and a.held_out_indices == b.held_out_indices
            for a, b in zip(folds["l2"], folds["combined"])
        ),
        "unpaired folds",
    )
    return PreparedDataset(
        tuple(sessions),
        tuple(groups),
        tuple(vectors),
        {h: tuple(y) for h, y in targets.items()},
        folds,
    )


def checked_fit(fold: PreparedFold, targets: Sequence[float]):
    fit = fit_prepared_fold(fold, targets)
    require(
        all(
            math.isfinite(x)
            for x in (
                *fit.coefficients,
                fit.intercept,
                *fit.predictions,
                *fit.baseline_predictions,
            )
        ),
        "nonfinite fit",
    )
    return fit


def fit_models(
    folds: dict[str, tuple[PreparedFold, ...]],
    targets: Sequence[float],
    *,
    observed: bool,
) -> dict:
    result = {}
    for model in MODELS:
        predictions, actual, baselines, details = [], [], [], []
        for number, fold in enumerate(folds[model], 2):
            fit = checked_fit(fold, targets)
            y = [targets[i] for i in fold.held_out_indices]
            predictions.extend(fit.predictions)
            baselines.extend(fit.baseline_predictions)
            actual.extend(y)
            if observed:
                training_predictions = tuple(
                    fit.intercept
                    + sum(x * beta for x, beta in zip(row, fit.coefficients))
                    for row in fold.x_training
                )
                details.append(
                    {
                        "test_session_index": number,
                        "training_session_indices": list(range(1, number)),
                        "sample_count": len(y),
                        "training_count": len(fold.training_indices),
                        "spearman": spearman(y, fit.predictions),
                        "mse": mean_squared_error(y, fit.predictions),
                        "baseline_mse": mean_squared_error(y, fit.baseline_predictions),
                        "training_prediction_quantiles": quantiles(
                            training_predictions
                        ),
                        "intercept": fit.intercept,
                        "coefficients": fit.coefficients,
                        "medians": fold.medians,
                        "means": fold.means,
                        "scales": fold.scales,
                    }
                )
        metrics: dict = {"spearman": spearman(actual, predictions)}
        if observed:
            mse = finite(mean_squared_error(actual, predictions), "MSE")
            baseline_mse = finite(mean_squared_error(actual, baselines), "baseline MSE")
            rhos = [item["spearman"] for item in details]
            metrics.update(
                mse=mse,
                baseline_mse=baseline_mse,
                relative_mse_improvement=1 - mse / baseline_mse
                if baseline_mse
                else None,
                equal_session_spearman=sum(rhos) / 7
                if all(r is not None for r in rhos)
                else None,
                positive_session_fraction=sum(r is not None and r > 0 for r in rhos)
                / 7,
                mean_prediction=sum(predictions) / len(predictions),
                prediction_quantiles=quantiles(predictions),
                sample_count=len(actual),
                fold_count=7,
            )
        result[model] = {
            "metrics": metrics,
            "folds": details,
            "predictions": tuple(predictions),
            "baseline_predictions": tuple(baselines),
        }
    require(
        result["l2"]["baseline_predictions"]
        == result["combined"]["baseline_predictions"],
        "unpaired baseline",
    )
    return result


def primary_statistics(result: dict) -> tuple[float, float]:
    l2 = finite(result["l2"]["metrics"]["spearman"], "primary L2 Spearman")
    combined = finite(
        result["combined"]["metrics"]["spearman"], "primary combined Spearman"
    )
    return combined, combined - l2


def shift_offsets(groups: Sequence[Sequence[int]]):
    require(sys.version_info[:2] == (3, 12), "frozen null requires Python 3.12")
    require(
        len(groups) == 8 and all(len(group) >= 1202 for group in groups),
        "invalid circular shift groups",
    )
    generator = random.Random(20260916)
    for _ in range(1000):
        yield tuple(generator.randrange(601, len(group) - 600) for group in groups)


def rotate_targets(targets: Sequence[float], groups, offsets) -> tuple[float, ...]:
    require(len(groups) == len(offsets) == 8, "invalid shift dimensions")
    require(
        [i for group in groups for i in group] == list(range(len(targets))),
        "noncanonical shift groups",
    )
    shifted = list(targets)
    for group, offset in zip(groups, offsets):
        require(
            min(offset, len(group) - offset) > 600, "excluded circular shift offset"
        )
        for j, destination in enumerate(group):
            shifted[destination] = targets[group[(j + offset) % len(group)]]
    return tuple(shifted)


def paired_null(prepared: PreparedDataset, observed: dict) -> dict:
    observed_combined, observed_increment = primary_statistics(observed)
    statistics = []
    offsets_used = []
    for offsets in shift_offsets(prepared.groups):
        shifted = rotate_targets(prepared.targets[300], prepared.groups, offsets)
        # Coefficients/intercepts are always refit. Only feature preprocessing/Gram inverses are cached.
        result = fit_models(prepared.folds, shifted, observed=False)
        statistics.append(primary_statistics(result))
        offsets_used.append(offsets)
    require(len(statistics) == 1000, "exactly 1000 valid null statistics required")
    return {
        "seed": 20260916,
        "valid_permutations": 1000,
        "offsets": offsets_used,
        "statistics": statistics,
        "combined_p": (1 + sum(s[0] >= observed_combined for s in statistics)) / 1001,
        "incremental_p": (1 + sum(s[1] >= observed_increment for s in statistics))
        / 1001,
    }


def classify_primary(primary: dict, null: dict, economic: dict) -> dict:
    """Deliberately accepts only the primary result, never a horizon search/dictionary."""
    require(primary.get("horizon_seconds") == 300, "only 300s may classify")
    combined, increment = primary_statistics(primary)
    l2, both = primary["l2"], primary["combined"]
    require(
        len(l2["folds"]) == len(both["folds"]) == 7, "seven held-out sessions required"
    )
    l2_rhos = [finite(f["spearman"], "session L2 Spearman") for f in l2["folds"]]
    rhos = [finite(f["spearman"], "session combined Spearman") for f in both["folds"]]
    l2_mse = finite(l2["metrics"]["mse"], "L2 MSE")
    combined_mse = finite(both["metrics"]["mse"], "combined MSE")
    baseline = finite(both["metrics"]["baseline_mse"], "baseline MSE")
    require(min(l2_mse, combined_mse, baseline) >= 0, "negative MSE")
    require(null["valid_permutations"] == 1000, "invalid null count")
    p, delta_p = (
        finite(null["combined_p"], "combined p"),
        finite(null["incremental_p"], "incremental p"),
    )
    require(1 / 1001 <= p <= 1 and 1 / 1001 <= delta_p <= 1, "invalid add-one p-value")
    information = {
        "combined_pooled_spearman": combined > 0,
        "combined_minus_l2_pooled_spearman": increment > 0,
        "combined_MSE": combined_mse < l2_mse and combined_mse < baseline,
        "positive_combined_session_spearman": sum(r > 0 for r in rhos) >= 5,
        "positive_combined_minus_l2_session_spearman": sum(
            c > l for c, l in zip(rhos, l2_rhos)
        )
        >= 5,
        "combined_shift_p": p < 0.05,
        "incremental_shift_p": delta_p < 0.05,
    }
    require(
        set(economic)
        == {"tail_coverage", "tail_base_net", "anchor_coverage", "anchor_base_net"}
        and all(type(value) is bool for value in economic.values()),
        "invalid economic gate inputs",
    )
    label = (
        "NO_STABLE_COMBINED_SIGNAL"
        if not all(information.values())
        else "INFORMATION_PRESENT_BUT_TOO_SMALL"
        if not all(economic.values())
        else "ECONOMIC_SIGNAL_PRESENT"
    )
    return {
        "horizon_seconds": 300,
        "information_gates": information,
        "economic_gates": economic,
        "classification": label,
        "evidence_role": "DISCOVERY ONLY",
        "warning": "Positive classification does not establish executable profitability.",
    }
