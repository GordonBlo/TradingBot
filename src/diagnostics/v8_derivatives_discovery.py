"""Pure statistical primitives for the frozen V8 derivatives diagnostic."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


def average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for index in ordered[position:end]:
            ranks[index] = average
        position = end
    return tuple(ranks)


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(average_ranks(left), average_ranks(right))


def mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("MSE requires equal non-empty vectors.")
    return sum((observed - estimate) ** 2 for observed, estimate in zip(actual, predicted)) / len(actual)


def cohens_d(winners: Sequence[float], losers: Sequence[float]) -> float | None:
    if len(winners) < 2 or len(losers) < 2:
        return None
    winner_mean = sum(winners) / len(winners)
    loser_mean = sum(losers) / len(losers)
    winner_var = sum((value - winner_mean) ** 2 for value in winners) / (len(winners) - 1)
    loser_var = sum((value - loser_mean) ** 2 for value in losers) / (len(losers) - 1)
    pooled = (
        ((len(winners) - 1) * winner_var + (len(losers) - 1) * loser_var)
        / (len(winners) + len(losers) - 2)
    )
    if pooled <= 0.0:
        return None
    return (winner_mean - loser_mean) / math.sqrt(pooled)


def _inverse(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    size = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("Ridge matrix is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_component
                    for value, pivot_component in zip(augmented[row], augmented[column])
                ]
    return tuple(tuple(row[size:]) for row in augmented)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


@dataclass(frozen=True, slots=True)
class PreparedFold:
    training_indices: tuple[int, ...]
    held_out_indices: tuple[int, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    x_training: tuple[tuple[float, ...], ...]
    x_held_out: tuple[tuple[float, ...], ...]
    ridge_inverse: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class FoldFit:
    intercept: float
    coefficients: tuple[float, ...]
    predictions: tuple[float, ...]
    baseline_predictions: tuple[float, ...]


def prepare_fold(
    features: Sequence[Sequence[float | None]],
    *,
    training_indices: Iterable[int],
    held_out_indices: Iterable[int],
    alpha: float,
) -> PreparedFold:
    training = tuple(training_indices)
    held_out = tuple(held_out_indices)
    if not training or not held_out or set(training) & set(held_out):
        raise ValueError("Fold training and held-out rows must be non-empty and isolated.")
    width = len(features[0])
    if alpha <= 0 or any(len(row) != width for row in features):
        raise ValueError("Ridge features or alpha are invalid.")

    medians = []
    for column in range(width):
        observed = [
            float(features[index][column])
            for index in training
            if features[index][column] is not None
        ]
        medians.append(_median(observed) if observed else 0.0)

    imputed_training = tuple(
        tuple(
            float(features[index][column])
            if features[index][column] is not None
            else medians[column]
            for column in range(width)
        )
        for index in training
    )
    means = tuple(
        sum(row[column] for row in imputed_training) / len(imputed_training)
        for column in range(width)
    )
    scales = tuple(
        math.sqrt(
            sum((row[column] - means[column]) ** 2 for row in imputed_training)
            / len(imputed_training)
        )
        for column in range(width)
    )

    def transform(index: int) -> tuple[float, ...]:
        output = []
        for column in range(width):
            value = features[index][column]
            imputed = float(value) if value is not None else medians[column]
            output.append(
                (imputed - means[column]) / scales[column]
                if scales[column] > 0.0
                else 0.0
            )
        return tuple(output)

    x_training = tuple(transform(index) for index in training)
    x_held_out = tuple(transform(index) for index in held_out)
    gram = [
        [
            sum(row[left] * row[right] for row in x_training)
            + (alpha if left == right else 0.0)
            for right in range(width)
        ]
        for left in range(width)
    ]
    return PreparedFold(
        training_indices=training,
        held_out_indices=held_out,
        medians=tuple(medians),
        means=means,
        scales=scales,
        x_training=x_training,
        x_held_out=x_held_out,
        ridge_inverse=_inverse(gram),
    )


def fit_prepared_fold(fold: PreparedFold, targets: Sequence[float]) -> FoldFit:
    training_targets = tuple(float(targets[index]) for index in fold.training_indices)
    intercept = sum(training_targets) / len(training_targets)
    width = len(fold.medians)
    rhs = tuple(
        sum(
            row[column] * (target - intercept)
            for row, target in zip(fold.x_training, training_targets)
        )
        for column in range(width)
    )
    coefficients = tuple(
        sum(fold.ridge_inverse[row][column] * rhs[column] for column in range(width))
        for row in range(width)
    )
    predictions = tuple(
        intercept + sum(value * coefficient for value, coefficient in zip(row, coefficients))
        for row in fold.x_held_out
    )
    return FoldFit(
        intercept=intercept,
        coefficients=coefficients,
        predictions=predictions,
        baseline_predictions=(intercept,) * len(fold.held_out_indices),
    )


def within_window_permutations(
    window_ids: Sequence[str],
    *,
    seed: int,
    count: int,
) -> Iterator[tuple[int, ...]]:
    if count < 1:
        raise ValueError("Permutation count must be positive.")
    groups: dict[str, list[int]] = {}
    for index, window_id in enumerate(window_ids):
        groups.setdefault(window_id, []).append(index)
    randomizer = random.Random(seed)
    for _ in range(count):
        source_indices = list(range(len(window_ids)))
        for group in groups.values():
            shuffled = list(group)
            randomizer.shuffle(shuffled)
            for destination, source in zip(group, shuffled):
                source_indices[destination] = source
        yield tuple(source_indices)


def classify_discovery(
    *,
    held_out_spearman: float | None,
    permutation_p_value: float,
    model_mse: float,
    baseline_mse: float,
    positive_direction_windows: int,
    has_stable_economic_feature: bool,
) -> str:
    correlation_positive = held_out_spearman is not None and held_out_spearman > 0.0
    beats_baseline = model_mse < baseline_mse
    stable = all(
        (
            correlation_positive,
            permutation_p_value <= 0.05,
            beats_baseline,
            positive_direction_windows >= 7,
            has_stable_economic_feature,
        )
    )
    if stable:
        return "STABLE_DERIVATIVES_SIGNAL"
    if correlation_positive or beats_baseline:
        return "WEAK_OR_UNSTABLE_SIGNAL"
    return "NO_STABLE_DERIVATIVES_SIGNAL"
