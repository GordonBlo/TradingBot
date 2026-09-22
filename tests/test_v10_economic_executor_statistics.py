from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from src.diagnostics import v10_economic_discovery as engine
from src.diagnostics import v10_economic_reporting as reporting
from src.diagnostics.v10_economic_features import HORIZONS, Sample, SessionSamples
from src.research.v10_economic_execution import encoded

D = Decimal
START = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def sessions():
    result = []
    for s in range(8):
        start = START + timedelta(days=s)
        rows = []
        for i in range(1202):
            features = tuple(
                D(str(math.sin((i + 3 * s) / (j + 3)) + (i % (j + 2)) / 50))
                for j in range(40)
            )
            targets = {
                h: D(str(math.sin((i + 3 * s) / 5) * h / 30 + i % 13 / 10))
                for h in HORIZONS
            }
            economics = {
                h: {
                    "unadjusted_quote_return_bps": targets[h] - 1,
                    **{
                        scenario: {
                            "mid_move": targets[h],
                            "executable_gross": targets[h] - 1,
                            "spread": D(1),
                            "adverse": D(4 * multiplier),
                            "fees": D(20 * multiplier),
                            "net": targets[h] - 1 - 24 * multiplier,
                        }
                        for scenario, multiplier in (("base", 1), ("stress", 2))
                    },
                }
                for h in HORIZONS
            }
            rows.append(
                Sample(
                    start + timedelta(seconds=i + 30), features, targets, economics, {}
                )
            )
        result.append(
            SessionSamples(
                f"SYNTHETIC_{s}", start, start + timedelta(hours=3), tuple(rows), {}
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def prepared(sessions):
    return engine.prepare_dataset(sessions)


def test_paired_rows_features_and_seven_expanding_folds(prepared):
    assert len(prepared.groups) == 8
    assert set(prepared.targets) == set(HORIZONS)
    for k, (l2, combined) in enumerate(
        zip(prepared.folds["l2"], prepared.folds["combined"]), 1
    ):
        assert len(l2.medians) == 13 and len(combined.medians) == 40
        assert (
            l2.training_indices == combined.training_indices == tuple(range(1202 * k))
        )
        assert l2.held_out_indices == combined.held_out_indices == prepared.groups[k]
        assert all(a == b[:13] for a, b in zip(l2.x_training, combined.x_training))


def test_training_only_preprocessing_constant_and_missing_columns():
    rows = [(1.0, None, 7.0), (3.0, 4.0, 7.0), (1000000.0, -1000000.0, 99.0)]
    fold = engine.strict_prepare(rows, (0, 1), (2,))
    assert fold.medians == (2.0, 4.0, 7.0)
    assert fold.means == (2.0, 4.0, 7.0)
    assert fold.scales == (1.0, 0.0, 0.0)
    assert fold.x_held_out == ((999998.0, 0.0, 0.0),)
    fit = engine.checked_fit(fold, (2.0, 6.0, -1000.0))
    assert fit.intercept == 4
    assert fit.baseline_predictions == (4.0,)
    assert fit.coefficients[0] == pytest.approx(4 / 3)  # X'X=2, alpha=1
    assert fit.predictions[0] == pytest.approx(4 + 999998 * 4 / 3)


@pytest.mark.parametrize(
    "rows", [[(None,), (None,), (3.0,)], [(1.0,), (float("nan"),), (3.0,)]]
)
def test_invalid_training_inputs_fail(rows):
    with pytest.raises(ValueError):
        engine.strict_prepare(rows, (0, 1), (2,))


def test_observed_indexing_baseline_and_train_quantiles(prepared):
    result = engine.fit_models(prepared.folds, prepared.targets[300], observed=True)
    y = prepared.targets[300]
    for model in engine.MODELS:
        assert len(result[model]["predictions"]) == 7 * 1202
        for k, fold in enumerate(prepared.folds[model]):
            direct = engine.fit_prepared_fold(fold, y)
            assert (
                result[model]["predictions"][k * 1202 : (k + 1) * 1202]
                == direct.predictions
            )
            assert (
                result[model]["baseline_predictions"][k * 1202 : (k + 1) * 1202]
                == direct.baseline_predictions
            )
        actual = y[1202:]
        assert result[model]["metrics"]["spearman"] == engine.spearman(
            actual, result[model]["predictions"]
        )
        assert result[model]["metrics"]["mse"] == engine.mean_squared_error(
            actual, result[model]["predictions"]
        )


def test_quantile_type7_and_ties():
    assert engine.quantiles((0.0, 10.0))[0] == 5
    assert engine.quantiles((0.0, 10.0))[-1] == 9.9
    assert engine.quantiles((2.0,) * 10) == (2.0,) * 6


def test_null_offsets_exact_seed_order_exclusion_and_rotation(prepared):
    offsets = list(engine.shift_offsets(prepared.groups))
    assert len(offsets) == 1000
    assert offsets == list(engine.shift_offsets(prepared.groups))
    assert set(offsets) == {(601,) * 8}  # only admissible offset for n=1202
    groups = tuple(tuple(range(i * 1211, (i + 1) * 1211)) for i in range(8))
    rng = random.Random(20260916)
    expected = [
        tuple(rng.randrange(601, len(g) - 600) for g in groups) for _ in range(1000)
    ]
    assert list(engine.shift_offsets(groups)) == expected
    y = tuple(float(i) for i in range(8 * 1211))
    shifted = engine.rotate_targets(y, groups, expected[0])
    for group, offset in zip(groups, expected[0]):
        assert all(
            shifted[destination] == y[group[(j + offset) % len(group)]]
            for j, destination in enumerate(group)
        )
        assert min(offset, len(group) - offset) > 600


def test_1000_paired_refits_and_add_one_formula(prepared, monkeypatch):
    calls = []

    def synthetic_fit(folds, shifted, *, observed):
        assert observed is False and folds is prepared.folds
        calls.append(shifted)
        rho = 0.4 if len(calls) <= 49 else 0.1
        return {
            "combined": {"metrics": {"spearman": rho}},
            "l2": {"metrics": {"spearman": 0.05}},
        }

    monkeypatch.setattr(engine, "fit_models", synthetic_fit)
    observed = {
        "combined": {"metrics": {"spearman": 0.4}},
        "l2": {"metrics": {"spearman": 0.05}},
    }
    result = engine.paired_null(prepared, observed)
    assert len(calls) == len(result["statistics"]) == 1000
    assert result["combined_p"] == result["incremental_p"] == 50 / 1001
    assert calls[0] == engine.rotate_targets(
        prepared.targets[300], prepared.groups, (601,) * 8
    )


def test_every_null_fold_reuses_only_features_and_refits(prepared, monkeypatch):
    calls = []
    original = engine.fit_prepared_fold

    def spy(fold, targets):
        calls.append((fold, targets))
        return original(fold, targets)

    monkeypatch.setattr(engine, "fit_prepared_fold", spy)
    shifted = engine.rotate_targets(prepared.targets[300], prepared.groups, (601,) * 8)
    result = engine.fit_models(prepared.folds, shifted, observed=False)
    assert len(calls) == 14 and all(y is shifted for _, y in calls)
    assert result["l2"]["baseline_predictions"][0] == sum(shifted[:1202]) / 1202
    assert engine.primary_statistics(result) == engine.primary_statistics(
        engine.fit_models(prepared.folds, shifted, observed=False)
    )


def test_actual_1000_null_refits_on_small_synthetic_fit_matrices():
    # Large enough circular-shift domains, tiny fit matrices: exact count/refit/statistics
    # without multiplying the full 40-column integration test's cost by 1000.
    groups = tuple(tuple(range(k * 1211, (k + 1) * 1211)) for k in range(8))
    y = tuple(math.sin(i / 9) + (i % 17) / 100 for i in range(8 * 1211))
    x = tuple((math.sin(i / 7),) for i in range(len(y)))
    folds = tuple(
        engine.strict_prepare(
            x, tuple(i for g in groups[:k] for i in g[:4]), groups[k][:4]
        )
        for k in range(1, 8)
    )
    prepared = engine.PreparedDataset(
        (), groups, x, {300: y}, {"l2": folds, "combined": folds}
    )
    observed = engine.fit_models(prepared.folds, y, observed=False)
    first = engine.paired_null(prepared, observed)
    second = engine.paired_null(prepared, observed)
    assert first == second and first["valid_permutations"] == 1000
    assert len(first["statistics"]) == 1000 and first["incremental_p"] == 1
    assert all(math.isfinite(v) for pair in first["statistics"] for v in pair)
    assert (
        first["combined_p"]
        == (
            1
            + sum(
                pair[0] >= observed["combined"]["metrics"]["spearman"]
                for pair in first["statistics"]
            )
        )
        / 1001
    )


def primary():
    return {
        "horizon_seconds": 300,
        "l2": {
            "metrics": {"spearman": 0.1, "mse": 2.0},
            "folds": [{"spearman": 0.1}] * 7,
        },
        "combined": {
            "metrics": {"spearman": 0.2, "mse": 1.0, "baseline_mse": 3.0},
            "folds": [{"spearman": 0.2}] * 7,
        },
    }


def passing_null():
    return {"valid_permutations": 1000, "combined_p": 0.01, "incremental_p": 0.01}


ECONOMIC = {
    "tail_coverage": True,
    "tail_base_net": True,
    "anchor_coverage": True,
    "anchor_base_net": True,
}


@pytest.mark.parametrize(
    "change",
    [
        lambda p, n: p["combined"]["metrics"].update(spearman=0.0),
        lambda p, n: p["combined"]["metrics"].update(spearman=0.1),
        lambda p, n: p["combined"]["metrics"].update(mse=2.0),
        lambda p, n: p["combined"]["metrics"].update(baseline_mse=1.0),
        lambda p, n: n.update(combined_p=0.05),
        lambda p, n: n.update(incremental_p=0.05),
        lambda p, n: p["combined"].update(
            folds=[{"spearman": 0.2}] * 4 + [{"spearman": 0.0}] * 3
        ),
        lambda p, n: p["l2"].update(
            folds=[{"spearman": 0.1}] * 4 + [{"spearman": 0.2}] * 3
        ),
    ],
)
def test_information_gate_strict_boundaries(change):
    p, n = primary(), passing_null()
    change(p, n)
    result = engine.classify_primary(p, n, ECONOMIC)
    assert result["classification"] == "NO_STABLE_COMBINED_SIGNAL"


def test_five_of_seven_and_economic_failure_labels():
    p = primary()
    p["combined"]["folds"] = [{"spearman": 0.2}] * 5 + [{"spearman": -0.1}] * 2
    assert (
        engine.classify_primary(p, passing_null(), ECONOMIC)["classification"]
        == "ECONOMIC_SIGNAL_PRESENT"
    )
    for key in ECONOMIC:
        assert (
            engine.classify_primary(p, passing_null(), {**ECONOMIC, key: False})[
                "classification"
            ]
            == "INFORMATION_PRESENT_BUT_TOO_SMALL"
        )


@pytest.mark.parametrize("horizon", [5, 30, 60])
def test_no_secondary_classification(horizon):
    p = primary()
    p["horizon_seconds"] = horizon
    with pytest.raises(ValueError, match="only 300s"):
        engine.classify_primary(p, passing_null(), ECONOMIC)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_undefined_required_statistics_stop_classification(value):
    p = primary()
    p["combined"]["folds"] = [{"spearman": value}] * 7
    with pytest.raises(ValueError, match="required statistic"):
        engine.classify_primary(p, passing_null(), ECONOMIC)


SMALL_POSITIVE = D(".0000001")


def regions(
    counts=(20, 20, 20, 20, 20, 0, 0),
    anchor_counts=(5, 5, 5, 5, 0, 0, 0),
    mean=SMALL_POSITIVE,
):
    def region(values):
        return {
            "count": sum(values),
            "session_counts": dict(enumerate(values)),
            "per_session": {
                i: {"mean_base_net_bps": mean if n else None}
                for i, n in enumerate(values)
            },
            "base": {"mean_bps": {"net": mean if sum(values) else None}},
        }

    return region(counts), region(anchor_counts)


def test_economic_coverage_and_exact_unrounded_boundaries():
    assert all(reporting.economic_gates(*regions()).values())
    assert not reporting.economic_gates(*regions(counts=(20, 20, 20, 20, 19, 0, 0)))[
        "tail_coverage"
    ]
    assert not reporting.economic_gates(*regions(counts=(25, 25, 25, 25, 0, 0, 0)))[
        "tail_coverage"
    ]
    assert not reporting.economic_gates(*regions(anchor_counts=(5, 5, 5, 4, 0, 0, 0)))[
        "anchor_coverage"
    ]
    assert not reporting.economic_gates(*regions(anchor_counts=(10, 5, 5, 0, 0, 0, 0)))[
        "anchor_coverage"
    ]
    zero = reporting.economic_gates(*regions(mean=D(0)))
    assert not zero["tail_base_net"] and not zero["anchor_base_net"]
    empty = reporting.economic_gates(*regions(counts=(0,) * 7, anchor_counts=(0,) * 7))
    assert not any(empty.values())


def test_nonoverlapping_run_onset_anchors_and_missing_rows(sessions):
    session = sessions[1]
    selected = [0, 1, 300, 301, 601, 602, 903]
    assert reporting.audit_anchors(session, selected) == [0, 601, 903]
    # Integer grid requires 301s, not 300s, between accepted anchors; no within-run rescue.
    assert reporting.audit_anchors(session, [0, 300, 301, 302, 303]) == [0]
    sparse = replace(session, rows=(session.rows[0], session.rows[2]))
    assert reporting.signal_runs(sparse, [0, 1]) == [[0], [1]]


def test_persistence_clustering_and_turnover(sessions):
    value = reporting.persistence(sessions[1:], [[0, 1, 3]] + [[]] * 6, 5)
    assert value["run_count"] == 2 and value["mean_run_seconds"] == 1.5
    assert value["median_run_seconds"] == 1.5
    assert value["inter_onset_seconds"] == [3.0]
    assert (
        len(value["per_session"][sessions[1].session_id]["five_minute_onset_counts"])
        == 36
    )
    assert value["turnover_demand_proxy"]["signals"] == 3
    assert value["turnover_demand_proxy"]["quote_side_operations"] == 6
    assert (
        value["turnover_demand_proxy"]["maximum_overlapping_hypothetical_intervals"]
        == 3
    )
    assert value["eligible_test_hours"] == 7 * 1202 / 3600


def test_empty_negative_tied_bins_and_no_outcome_selection(sessions):
    result = {
        "predictions": (-2.0,) * (7 * 1202),
        "folds": [{"training_prediction_quantiles": (-2.0,) * 6}] * 7,
    }
    with localcontext() as context:
        context.prec = 50
        report, tail = reporting.response_report(sessions[1:], result, 300)
    assert report["regions"]["bin_0"]["count"] == 7 * 1202
    assert report["regions"]["bin_1"]["count"] == 0
    assert report["regions"]["bin_1"]["mean_realized_mid_bps"] is None
    assert report["monotonicity_descriptive_only"]["status"] == "UNDEFINED"
    assert all(not indices for indices in tail)


def test_synthetic_full_reporting_is_deterministic_and_primary_only(
    sessions, prepared, monkeypatch
):
    monkeypatch.setattr(reporting, "prepare_dataset", lambda inputs: prepared)
    # The 1000-replicate mechanics/refits are tested separately; no expensive duplicate null here.
    monkeypatch.setattr(
        reporting,
        "paired_null",
        lambda *args: {
            **passing_null(),
            "seed": 20260916,
            "statistics": [(0.0, 0.0)] * 1000,
            "offsets": [(601,) * 8] * 1000,
        },
    )
    result = reporting.build_diagnostic(sessions)
    assert encoded(result) == encoded(reporting.build_diagnostic(sessions))
    report, predictions, null = result
    assert report["primary"]["horizon_seconds"] == 300
    assert report["evidence_role"] == "DISCOVERY ONLY"
    assert (
        report["warning"]
        == "Positive classification does not establish executable profitability."
    )
    assert set(predictions) == {"5", "30", "60", "300"}
    assert len(null["statistics"]) == 1000
    # No secondary data enters the classifier: its entire input is the isolated primary mapping.
    p = primary()
    original = engine.classify_primary(p, passing_null(), ECONOMIC)
    unrelated_secondary = deepcopy(p)
    unrelated_secondary["combined"]["metrics"]["spearman"] = -1
    assert engine.classify_primary(p, passing_null(), ECONOMIC) == original
