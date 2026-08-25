"""Deterministic synthesis of already-consumed V8 derivatives evidence only."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


VERSION = "8.2-research-synthesis"
CLOSE = "CLOSE_V8_DERIVATIVES_BRANCH"
CONTINUE = "CONTINUE_V8_WITH_JUSTIFIED_MECHANISM"

EXPECTED_RUN_IDS = {
    "h0_funding": "089b62c96a7de979",
    "h1_open_interest": "6e7c9147b8c3187e",
    "discovery": "d15c2c2b5bafefd2",
    "h2_validation": "b334dac73e746982",
}
EXPECTED_VALIDATION_DATASET_ID = "52302ca4385f5240"


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    root_manifest: Path = Path("research/hypothesis_manifest.json")
    h0_manifest: Path = Path("research/v8_funding_context/089b62c96a7de979/manifest.json")
    h0_result: Path = Path("reports/v8_funding_context/089b62c96a7de979/summary.json")
    h1_manifest: Path = Path("research/v8_open_interest/6e7c9147b8c3187e/manifest.json")
    h1_result: Path = Path("reports/v8_open_interest/6e7c9147b8c3187e/summary.json")
    discovery_manifest: Path = Path("research/v8_derivatives_discovery/d15c2c2b5bafefd2/manifest.json")
    discovery_result: Path = Path("reports/v8_derivatives_discovery/d15c2c2b5bafefd2/summary.json")
    discovery_univariate: Path = Path("reports/v8_derivatives_discovery/d15c2c2b5bafefd2/univariate_features.csv")
    discovery_coefficients: Path = Path("reports/v8_derivatives_discovery/d15c2c2b5bafefd2/coefficient_stability.csv")
    h2_manifest: Path = Path("research/v8_h2_validation_preregistration/b334dac73e746982/manifest.json")
    h2_dataset_manifest: Path = Path("research/v8_h2_validation/52302ca4385f5240/manifest.json")
    h2_result: Path = Path("reports/v8_h2_validation/b334dac73e746982/summary.json")


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V8 synthesis artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(payload: dict, expected_run_id: str) -> None:
    definition = payload.get("definition")
    if definition is None:
        definition = {
            key: value
            for key, value in payload.items()
            if key not in ("run_id", "definition_sha256")
        }
    digest = hashlib.sha256(_canonical_json(definition).encode("utf-8")).hexdigest()
    if (
        payload.get("run_id") != expected_run_id
        or payload.get("definition_sha256") != digest
        or digest[:16] != expected_run_id
    ):
        raise ValueError(f"Frozen V8 manifest identity failed: {expected_run_id}")


def _verify_result(payload: dict, expected_run_id: str, classification_key: str) -> None:
    if payload.get("run_id") != expected_run_id or classification_key not in payload:
        raise ValueError(f"Frozen V8 result identity failed: {expected_run_id}")


def _holdout_access_is_false(payload: dict) -> bool:
    if not isinstance(payload, dict) or not str(payload.get("status", "")).startswith("LOCKED"):
        return False
    return all(payload.get(field) is False for field in ("revealed", "consumed", "evaluated")) and (
        "loaded" not in payload or payload["loaded"] is False
    )


def classify_branch(
    *, evidence_complete: bool, integrity_passes: bool,
    untested_mechanism_supported_without_posthoc_mining: bool,
) -> str:
    if (
        evidence_complete
        and integrity_passes
        and untested_mechanism_supported_without_posthoc_mining
    ):
        return CONTINUE
    return CLOSE


def _csv_by_feature(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["feature"]: row for row in csv.DictReader(stream)}


def build_synthesis(paths: ArtifactPaths = ArtifactPaths()) -> dict:
    root = _load_json(paths.root_manifest)
    h0_manifest = _load_json(paths.h0_manifest)
    h0 = _load_json(paths.h0_result)
    h1_manifest = _load_json(paths.h1_manifest)
    h1 = _load_json(paths.h1_result)
    discovery_manifest = _load_json(paths.discovery_manifest)
    discovery = _load_json(paths.discovery_result)
    h2_manifest = _load_json(paths.h2_manifest)
    h2_dataset = _load_json(paths.h2_dataset_manifest)
    h2 = _load_json(paths.h2_result)

    verify_manifest(h0_manifest, EXPECTED_RUN_IDS["h0_funding"])
    verify_manifest(h1_manifest, EXPECTED_RUN_IDS["h1_open_interest"])
    verify_manifest(discovery_manifest, EXPECTED_RUN_IDS["discovery"])
    verify_manifest(h2_manifest, EXPECTED_RUN_IDS["h2_validation"])
    _verify_result(h0, EXPECTED_RUN_IDS["h0_funding"], "final_classification")
    _verify_result(h1, EXPECTED_RUN_IDS["h1_open_interest"], "final_classification")
    _verify_result(discovery, EXPECTED_RUN_IDS["discovery"], "classification")
    _verify_result(h2, EXPECTED_RUN_IDS["h2_validation"], "final_classification")

    if (
        h2_dataset.get("dataset_id") != EXPECTED_VALIDATION_DATASET_ID
        or h2.get("validation_dataset_id") != EXPECTED_VALIDATION_DATASET_ID
        or h2_dataset.get("classification") != "VALIDATION_DATA_READY"
    ):
        raise ValueError("Frozen V8-H2 validation dataset identity failed.")

    root_holdout = root.get("blind_holdout", {})
    if (
        root.get("holdout_status") != "LOCKED_BLIND_HOLDOUT"
        or root_holdout.get("status") != "LOCKED_BLIND_HOLDOUT"
        or root_holdout.get("reveal_timestamp") is not None
        or root_holdout.get("consumed_timestamp") is not None
    ):
        raise ValueError("Authoritative blind holdout is not locked and untouched.")

    holdouts = {
        "h0_funding": h0["blind_holdout_integrity"],
        "h1_open_interest": h1["blind_holdout_integrity"],
        "discovery": discovery["integrity"]["blind_holdout"],
        "h2_validation": h2["blind_holdout"],
    }
    if not all(_holdout_access_is_false(item) for item in holdouts.values()):
        raise ValueError("A V8 stage reports blind-holdout access.")

    univariate = _csv_by_feature(paths.discovery_univariate)
    coefficients = _csv_by_feature(paths.discovery_coefficients)
    strongest_feature = discovery["coefficient_stability"]["strongest_stable_features"][0]
    if strongest_feature != "DERIVED_MARK_INDEX_PREMIUM_LEVEL":
        raise ValueError("Strongest frozen discovery feature changed.")
    strongest_univariate = univariate[strongest_feature]
    strongest_coefficient = coefficients[strongest_feature]

    h0_comparison = h0["comparison_vs_v6"]
    h1_comparison = h1["comparison_vs_v6"]
    h2_v6 = h2["v6_metrics"]
    h2_metrics = h2["h2_metrics"]
    evidence = [
        {
            "stage": "V8_H0_FUNDING",
            "run_id": EXPECTED_RUN_IDS["h0_funding"],
            "evidence_type": "CONSUMED_RESEARCH_STRATEGY_REPLAY",
            "classification": h0["final_classification"],
            "trades": h0["v8_base"]["retained_candidate_count"],
            "gross_expectancy_r": h0_comparison["gross_expectancy_v8"],
            "net_expectancy_r": h0_comparison["net_expectancy_v8"],
            "profit_factor_r": h0_comparison["profit_factor_v8"],
            "reference_gross_expectancy_r": h0_comparison["gross_expectancy_v6"],
            "reference_net_expectancy_r": h0_comparison["net_expectancy_v6"],
            "reference_profit_factor_r": h0_comparison["profit_factor_v6"],
            "positive_net_windows": h0["v8_base"]["positive_net_windows"],
            "consistency_denominator": 11,
            "net_better_windows": h0_comparison["net_better_windows"],
            "subset_integrity": h0["subset_diagnostics"]["non_v6_entries"] == 0,
        },
        {
            "stage": "V8_H1_OPEN_INTEREST",
            "run_id": EXPECTED_RUN_IDS["h1_open_interest"],
            "evidence_type": "CONSUMED_RESEARCH_STRATEGY_REPLAY",
            "classification": h1["final_classification"],
            "trades": h1["v8_base"]["retained_candidate_count"],
            "gross_expectancy_r": h1_comparison["gross_expectancy_v8"],
            "net_expectancy_r": h1_comparison["net_expectancy_v8"],
            "profit_factor_r": h1_comparison["profit_factor_v8"],
            "reference_gross_expectancy_r": h1_comparison["gross_expectancy_v6"],
            "reference_net_expectancy_r": h1_comparison["net_expectancy_v6"],
            "reference_profit_factor_r": h1_comparison["profit_factor_v6"],
            "positive_net_windows": h1["v8_base"]["positive_net_windows"],
            "consistency_denominator": 11,
            "net_better_windows": h1_comparison["net_better_windows"],
            "subset_integrity": h1["subset_diagnostics"]["non_v6_entries"] == 0,
        },
        {
            "stage": "V8_DERIVATIVES_DISCOVERY",
            "run_id": EXPECTED_RUN_IDS["discovery"],
            "evidence_type": "CONSUMED_RESEARCH_DIAGNOSTIC_NOT_TRADING_VALIDATION",
            "classification": discovery["classification"],
            "candidates": discovery["candidate_count"],
            "gross_expectancy_r": "NOT_APPLICABLE",
            "net_expectancy_r": "DESCRIPTIVE_TARGET_ONLY",
            "profit_factor_r": "NOT_APPLICABLE",
            "heldout_spearman_gross_r": discovery["aggregate_out_of_window"]["prediction_gross_r_spearman"],
            "positive_direction_windows": discovery["aggregate_out_of_window"]["positive_direction_windows"],
            "consistency_denominator": 11,
            "model_mse": discovery["aggregate_out_of_window"]["model_mse"],
            "constant_baseline_mse": discovery["aggregate_out_of_window"]["constant_training_baseline_mse"],
            "model_beats_baseline": discovery["aggregate_out_of_window"]["model_beats_baseline"],
            "permutation_p_value": discovery["permutation_null"]["one_sided_p_value"],
            "future_timestamp_violations": discovery["integrity"]["future_feature_timestamp_violations"],
            "heldout_leakage": discovery["integrity"]["held_out_preprocessing_or_target_leakage"],
        },
        {
            "stage": "V8_H2_MARK_INDEX_PREMIUM",
            "run_id": EXPECTED_RUN_IDS["h2_validation"],
            "dataset_id": EXPECTED_VALIDATION_DATASET_ID,
            "evidence_type": "INDEPENDENT_OUTCOME_UNSEEN_STRATEGY_VALIDATION",
            "classification": h2["final_classification"],
            "trades": h2["h2_retained_count"],
            "gross_expectancy_r": h2_metrics["frictionless_expectancy_r"],
            "net_expectancy_r": h2_metrics["net_expectancy_r"],
            "profit_factor_r": h2_metrics["profit_factor_r"],
            "reference_gross_expectancy_r": h2_v6["frictionless_expectancy_r"],
            "reference_net_expectancy_r": h2_v6["net_expectancy_r"],
            "reference_profit_factor_r": h2_v6["profit_factor_r"],
            "gross_better_blocks": h2["gross_better_chronological_blocks"],
            "positive_net_blocks": h2["positive_net_blocks"],
            "consistency_denominator": 7,
            "subset_integrity": h2["non_v6_entry_count"] == 0,
            "forensic_raw_selection_statistic": h2["permutation_test"]["observed_statistic"],
            "forensic_conditional_centered_statistic": h2["permutation_test"]["centered_observed_statistic"],
            "forensic_one_sided_p_value": h2["permutation_test"]["p_value"],
            "forensic_interpretation": h2["permutation_test"]["interpretation"],
        },
    ]

    evidence_complete = (
        len(evidence) == 4
        and all(item["classification"] for item in evidence)
        and len(h0["eligible_windows"]) == len(h1["eligible_windows"]) == 11
        and discovery["candidate_count"] == 443
        and discovery["feature_count"] == 27
        and h2["v6_candidate_count"] == 291
        and h2["h2_retained_count"] + h2["h2_filtered_count"] == 291
    )
    artifact_holdout_ranges = {
        name: {"start": item.get("start"), "end": item.get("end")}
        for name, item in holdouts.items()
    }
    authoritative_range = {
        "start": root_holdout["start"], "end": root_holdout["end"]
    }
    holdout_metadata_matches = {
        name: value == authoritative_range
        for name, value in artifact_holdout_ranges.items()
    }

    untested_supported = False
    classification = classify_branch(
        evidence_complete=evidence_complete,
        integrity_passes=all(item.get("subset_integrity", True) for item in evidence),
        untested_mechanism_supported_without_posthoc_mining=untested_supported,
    )
    core = {
        "version": VERSION,
        "scope": "ALREADY_CONSUMED_RESULTS_ONLY_NO_NEW_RESEARCH",
        "source_artifacts": {
            "h0_manifest_sha256": _sha256(paths.h0_manifest),
            "h0_result_sha256": _sha256(paths.h0_result),
            "h1_manifest_sha256": _sha256(paths.h1_manifest),
            "h1_result_sha256": _sha256(paths.h1_result),
            "discovery_manifest_sha256": _sha256(paths.discovery_manifest),
            "discovery_result_sha256": _sha256(paths.discovery_result),
            "h2_manifest_sha256": _sha256(paths.h2_manifest),
            "h2_dataset_manifest_sha256": _sha256(paths.h2_dataset_manifest),
            "h2_result_sha256": _sha256(paths.h2_result),
        },
        "evidence": evidence,
        "discovery_to_validation_reconciliation": {
            "strongest_discovery_feature": strongest_feature,
            "discovery_univariate_spearman_gross_r": strongest_univariate["spearman_gross_r"],
            "discovery_univariate_sign_consistency": (
                f'{strongest_univariate["window_sign_matches_aggregate"]}/'
                f'{strongest_univariate["window_sign_eligible"]}'
            ),
            "discovery_mean_standardized_coefficient": strongest_coefficient["mean_standardized_coefficient"],
            "discovery_coefficient_sign_consistency": (
                f'{strongest_coefficient["modal_nonzero_sign_folds"]}/11'
            ),
            "independent_validation_supported": False,
            "reason": "The strongest discovery feature failed independent H2 gross, net, PF, and block-consistency gates.",
        },
        "strongest_remaining_evidence": {
            "status": "NO_JUSTIFIED_UNTESTED_MECHANISM",
            "discovery_only_signal": "Small positive held-out rank association with significant permutation result.",
            "limitation": "The discovery model failed its constant baseline and was classified weak/unstable; selecting another ranked feature or threshold now would be post-hoc mining.",
        },
        "classification_logic": {
            "evidence_complete": evidence_complete,
            "tested_hypotheses_supported": False,
            "strongest_discovery_feature_independently_supported": False,
            "untested_mechanism_supported_without_posthoc_mining": untested_supported,
            "continue_requires_justified_untested_mechanism": True,
        },
        "integrity": {
            "all_manifest_hashes_verified": True,
            "all_result_run_ids_verified": True,
            "all_stages_report_no_blind_holdout_access": True,
            "authoritative_holdout_range": authoritative_range,
            "artifact_holdout_metadata_matches_authoritative": holdout_metadata_matches,
            "h1_holdout_metadata_note": "H1 reports no access, but its recorded holdout range differs from the authoritative root manifest.",
            "new_strategy_replays": 0,
            "new_feature_mining": False,
            "alternative_rules_or_thresholds_tested": False,
        },
        "final_classification": classification,
        "reasoning": [
            "H0 and H1 both underperformed frozen V6 on gross expectancy, net expectancy, and profit factor.",
            "Discovery produced weak/unstable evidence and did not beat its constant out-of-window baseline.",
            "The strongest discovery feature failed independent H2 strategic validation.",
            "No untested derivatives mechanism is justified without selecting another feature or threshold post hoc.",
        ],
    }
    digest = hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest()
    return {"synthesis_id": digest[:16], "definition_sha256": digest, **core}


def write_synthesis(
    *, output_root: str | Path = "reports/v8_derivatives_synthesis",
    paths: ArtifactPaths = ArtifactPaths(),
) -> Path:
    payload = build_synthesis(paths)
    path = Path(output_root) / payload["synthesis_id"] / "summary.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("Existing V8 synthesis differs from deterministic evidence.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    payload = build_synthesis()
    path = write_synthesis()
    print(f"V8 DERIVATIVES SYNTHESIS | ID: {payload['synthesis_id']}")
    print(f"FINAL CLASSIFICATION: {payload['final_classification']}")
    print("NO NEW RESEARCH OR REPLAY EXECUTED")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
