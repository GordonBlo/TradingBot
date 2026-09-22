"""All prespecified V10 descriptive regions; no profitable-region selection."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import pairwise
from statistics import median

from src.diagnostics.v10_economic_discovery import (
    MODELS,
    QUANTILES,
    THRESHOLDS,
    classify_primary,
    finite,
    fit_models,
    paired_null,
    prepare_dataset,
    primary_statistics,
    spearman,
)
from src.diagnostics.v10_economic_features import HORIZONS, SessionSamples
from src.research.v10_collection_preregistration import timestamp
from src.research.v10_collection_readiness import require


def average(values):
    return sum(values) / len(values) if values else None


def describe_region(sessions, predictions, horizon, selections) -> dict:
    selected_rows = []
    selected_predictions: list[float] = []
    counts, per_session = {}, {}
    cursor = 0
    for session, selected in zip(sessions, selections):
        rows = [session.rows[i] for i in selected]
        counts[session.session_id] = len(rows)
        base_net = [row.economics[horizon]["base"]["net"] for row in rows]
        per_session[session.session_id] = {
            "count": len(rows),
            "mean_realized_mid_bps": average([row.targets[horizon] for row in rows]),
            "mean_base_net_bps": average(base_net),
        }
        selected_rows.extend(rows)
        selected_predictions.extend(predictions[cursor + i] for i in selected)
        cursor += len(session.rows)
    realized = [row.targets[horizon] for row in selected_rows]
    result = {
        "count": len(realized),
        "session_counts": counts,
        "per_session": per_session,
        "mean_prediction_bps": average(selected_predictions),
        "mean_realized_mid_bps": average(realized),
        "median_realized_mid_bps": median(realized) if realized else None,
        "mid_win_rate": average([int(y > 0) for y in realized]),
        "mean_unadjusted_quote_return_bps": average(
            [
                row.economics[horizon]["unadjusted_quote_return_bps"]
                for row in selected_rows
            ]
        ),
    }
    for scenario in ("base", "stress"):
        nets = [row.economics[horizon][scenario]["net"] for row in selected_rows]
        result[scenario] = {
            "mean_bps": {
                name: average(
                    [row.economics[horizon][scenario][name] for row in selected_rows]
                )
                for name in (
                    "mid_move",
                    "executable_gross",
                    "spread",
                    "adverse",
                    "fees",
                    "net",
                )
            },
            "median_net_bps": median(nets) if nets else None,
            "positive_net_fraction": average([int(value > 0) for value in nets]),
        }
    return result


def signal_runs(session: SessionSamples, selected: list[int]) -> list[list[int]]:
    require(selected == sorted(set(selected)), "noncanonical signal selection")
    runs: list[list[int]] = []
    for index in selected:
        if runs and session.rows[index].at == session.rows[runs[-1][-1]].at + timedelta(
            seconds=1
        ):
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def audit_anchors(session: SessionSamples, selected: list[int]) -> list[int]:
    anchors: list[int] = []
    for run in signal_runs(session, selected):
        index = run[0]
        if not anchors or session.rows[index].at >= session.rows[
            anchors[-1]
        ].at + timedelta(seconds=300, milliseconds=100):
            anchors.append(index)
    return anchors


def persistence(sessions, selections, horizon: int) -> dict:
    all_lengths, all_gaps, all_bins = [], [], []
    details = {}
    total_signals = sum(map(len, selections))
    max_concurrent = 0
    for session, selected in zip(sessions, selections):
        runs = signal_runs(session, selected)
        onsets = [session.rows[run[0]].at for run in runs]
        lengths = [len(run) for run in runs]
        gaps = [(b - a).total_seconds() for a, b in pairwise(onsets)]
        bins = [0] * int(
            (session.ended_at - session.started_at) // timedelta(seconds=300)
        )
        for at in onsets:
            index = (at - session.started_at) // timedelta(seconds=300)
            if index < len(bins):
                bins[index] += 1
        active: deque = deque()
        for index in selected:
            at = session.rows[index].at
            while active and active[0] <= at:
                active.popleft()
            active.append(at + timedelta(seconds=horizon))
            max_concurrent = max(max_concurrent, len(active))
        all_lengths.extend(lengths)
        all_gaps.extend(gaps)
        all_bins.extend(bins)
        details[session.session_id] = {
            "run_count": len(runs),
            "run_lengths_seconds": lengths,
            "inter_onset_seconds": gaps,
            "five_minute_onset_counts": bins,
        }
    eligible_hours = sum(len(s.rows) for s in sessions) / 3600
    bin_mean = average(all_bins)
    fano = (
        average([(n - bin_mean) ** 2 for n in all_bins]) / bin_mean
        if bin_mean
        else None
    )
    return {
        "run_count": len(all_lengths),
        "mean_run_seconds": average(all_lengths),
        "median_run_seconds": median(all_lengths) if all_lengths else None,
        "max_run_seconds": max(all_lengths) if all_lengths else None,
        "inter_onset_seconds": all_gaps,
        "runs_per_eligible_hour": len(all_lengths) / eligible_hours,
        "clustering_fano": fano,
        "per_session": details,
        "turnover_demand_proxy": {
            "signals": total_signals,
            "quote_side_operations": 2 * total_signals,
            "operations_per_eligible_hour": 2 * total_signals / eligible_hours,
            "maximum_overlapping_hypothetical_intervals": max_concurrent,
        },
        "eligible_test_hours": eligible_hours,
        "nominal_test_session_hours": sum(
            (s.ended_at - s.started_at).total_seconds() for s in sessions
        )
        / 3600,
    }


def response_report(
    sessions, model_result, horizon: int
) -> tuple[dict, list[list[int]]]:
    predictions = model_result["predictions"]
    selections: dict[str, list[list[int]]] = {f"bin_{i}": [] for i in range(7)}
    selections.update({f"positive_q{q}": [] for q in QUANTILES})
    selections.update({f"above_{level}bps": [] for level in THRESHOLDS})
    cursor = 0
    for session, fold in zip(sessions, model_result["folds"]):
        estimates = predictions[cursor : cursor + len(session.rows)]
        edges = fold["training_prediction_quantiles"]
        for i in range(7):
            selections[f"bin_{i}"].append(
                [j for j, p in enumerate(estimates) if bisect_left(edges, p) == i]
            )
        for q, edge in zip(QUANTILES, edges):
            selections[f"positive_q{q}"].append(
                [j for j, p in enumerate(estimates) if p > max(0, edge)]
            )
        for level in THRESHOLDS:
            selections[f"above_{level}bps"].append(
                [j for j, p in enumerate(estimates) if p > level]
            )
        cursor += len(session.rows)
    regions = {}
    for name, selection in selections.items():
        region = describe_region(sessions, predictions, horizon, selection)
        region["fraction_of_held_out_rows"] = region["count"] / len(predictions)
        if not name.startswith("bin_"):
            region["persistence"] = persistence(sessions, selection, horizon)
        regions[name] = region
    means = [regions[f"bin_{i}"]["mean_realized_mid_bps"] for i in range(7)]
    monotonicity: dict = {"status": "UNDEFINED", "spearman_bin_mean": None}
    if all(value is not None for value in means):
        differences = [b - a for a, b in pairwise(means)]
        monotonicity = {
            "status": "PASS"
            if all(d >= 0 for d in differences) and any(d > 0 for d in differences)
            else "FAIL",
            "spearman_bin_mean": spearman(list(range(7)), [float(m) for m in means]),
        }
    return {
        "regions": regions,
        "monotonicity_descriptive_only": monotonicity,
    }, selections["positive_q0.95"]


def economic_gates(tail: dict, anchors: dict) -> dict:
    counts = list(tail["session_counts"].values())
    require(
        len(counts) == len(anchors["session_counts"]) == 7,
        "missing held-out economic sessions",
    )
    require(
        sum(counts) == tail["count"]
        and sum(anchors["session_counts"].values()) == anchors["count"],
        "economic coverage mismatch",
    )
    means = [row["mean_base_net_bps"] for row in tail["per_session"].values()]
    pooled, anchor_mean = (
        tail["base"]["mean_bps"]["net"],
        anchors["base"]["mean_bps"]["net"],
    )
    for value in [*means, pooled, anchor_mean]:
        require(
            value is None or (isinstance(value, Decimal) and value.is_finite()),
            "nonfinite economic mean",
        )
    return {
        "tail_coverage": tail["count"] >= 100 and sum(n >= 20 for n in counts) >= 5,
        "tail_base_net": pooled is not None
        and pooled > 0
        and sum(m is not None and m > 0 for m in means) >= 5,
        "anchor_coverage": anchors["count"] >= 20
        and sum(n > 0 for n in anchors["session_counts"].values()) >= 4,
        "anchor_base_net": anchor_mean is not None and anchor_mean > 0,
    }


def build_diagnostic(sessions) -> tuple[dict, dict, dict]:
    """Only the future reserved runner may supply real samples. Tests supply synthetic samples."""
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        prepared = prepare_dataset(sessions)
        observed = {
            h: fit_models(prepared.folds, prepared.targets[h], observed=True)
            for h in HORIZONS
        }
        for horizon in HORIZONS:
            observed[horizon]["horizon_seconds"] = horizon
        primary = observed[300]
        primary_statistics(primary)
        for model in MODELS:
            for fold in primary[model]["folds"]:
                finite(fold["spearman"], "primary per-session Spearman")
        null = paired_null(prepared, primary)
        horizons = {}
        economic = None
        predictions = {
            str(h): {
                model: {
                    key: observed[h][model][key]
                    for key in ("predictions", "baseline_predictions")
                }
                for model in MODELS
            }
            for h in HORIZONS
        }
        for horizon in HORIZONS:
            horizon_report = {}
            for model in MODELS:
                result = observed[horizon][model]
                response, q95 = response_report(sessions[1:], result, horizon)
                horizon_report[model] = {
                    "metrics": result["metrics"],
                    "folds": result["folds"],
                    **response,
                }
                if horizon == 300 and model == "combined":
                    anchor_indices = [
                        audit_anchors(s, selected)
                        for s, selected in zip(sessions[1:], q95)
                    ]
                    anchors = describe_region(
                        sessions[1:], result["predictions"], 300, anchor_indices
                    )
                    anchors["timestamps_by_session"] = {
                        s.session_id: [timestamp(s.rows[i].at) for i in indices]
                        for s, indices in zip(sessions[1:], anchor_indices)
                    }
                    horizon_report[model]["non_overlapping_audit_anchors"] = anchors
                    economic = economic_gates(
                        response["regions"]["positive_q0.95"], anchors
                    )
            l2, both = observed[horizon]["l2"], observed[horizon]["combined"]
            a, b = l2["metrics"]["spearman"], both["metrics"]["spearman"]
            deltas = [
                c["spearman"] - l["spearman"]
                if c["spearman"] is not None and l["spearman"] is not None
                else None
                for l, c in zip(l2["folds"], both["folds"])
            ]
            horizon_report["paired_comparison"] = {
                "combined_minus_l2_spearman": b - a
                if a is not None and b is not None
                else None,
                "l2_minus_combined_mse": l2["metrics"]["mse"] - both["metrics"]["mse"],
                "per_session_spearman_difference": deltas,
                "positive_incremental_session_fraction": sum(
                    d is not None and d > 0 for d in deltas
                )
                / 7,
            }
            horizons[str(horizon)] = horizon_report
        comparisons: dict = {}
        for model in MODELS:
            primary_report = horizons["300"][model]
            comparisons[model] = {}
            for h in (5, 30, 60):
                shorter = horizons[str(h)][model]
                pr, sr = primary_report["metrics"], shorter["metrics"]
                tail_delta = {}
                for q in QUANTILES:
                    name = f"positive_q{q}"
                    high = primary_report["regions"][name]["base"]["mean_bps"]["net"]
                    low = shorter["regions"][name]["base"]["mean_bps"]["net"]
                    tail_delta[name] = (
                        high - low if high is not None and low is not None else None
                    )
                comparisons[model][str(h)] = {
                    "spearman_difference": pr["spearman"] - sr["spearman"]
                    if sr["spearman"] is not None
                    else None,
                    "mean_prediction_difference_bps": pr["mean_prediction"]
                    - sr["mean_prediction"],
                    "prediction_quantile_differences_bps": [
                        p - s
                        for p, s in zip(
                            pr["prediction_quantiles"], sr["prediction_quantiles"]
                        )
                    ],
                    "positive_tail_base_net_differences_bps": tail_delta,
                }
            delta = comparisons[model]["30"]["positive_tail_base_net_differences_bps"][
                "positive_q0.95"
            ]
            comparisons[model]["material_300s_vs_30s_descriptive_only"] = (
                delta >= 12 if delta is not None else None
            )
        if economic is None:
            raise ValueError("missing primary economic gates")
        report = {
            "evidence_role": "DISCOVERY ONLY",
            "dataset_status": "CONSUMED_DISCOVERY_EVIDENCE",
            "independent_confirmatory_strategy_validation": False,
            "warning": "Positive classification does not establish executable profitability.",
            "depth_capacity": "NOT_ESTIMATED",
            "blind_holdout": "LOCKED; not accessed",
            "sessions": [
                {
                    "session_id": s.session_id,
                    "sample_count": len(s.rows),
                    "exclusions": s.exclusions,
                }
                for s in sessions
            ],
            "primary": classify_primary(primary, null, economic),
            "horizons": horizons,
            "supporting_horizons_descriptive_only": [5, 30, 60],
            "horizon_comparison_descriptive_only": comparisons,
            "null": {
                key: null[key]
                for key in ("seed", "valid_permutations", "combined_p", "incremental_p")
            },
        }
        return report, predictions, null
