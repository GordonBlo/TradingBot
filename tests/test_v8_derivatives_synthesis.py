from __future__ import annotations

import json

import pytest

from src.research.v8_derivatives_synthesis import (
    CLOSE,
    CONTINUE,
    EXPECTED_RUN_IDS,
    build_synthesis,
    classify_branch,
    verify_manifest,
    write_synthesis,
)


def test_frozen_evidence_is_complete_and_identified() -> None:
    payload = build_synthesis()
    by_stage = {item["stage"]: item for item in payload["evidence"]}
    assert payload["classification_logic"]["evidence_complete"] is True
    assert {item["run_id"] for item in payload["evidence"]} == set(EXPECTED_RUN_IDS.values())
    assert set(by_stage) == {
        "V8_H0_FUNDING", "V8_H1_OPEN_INTEREST",
        "V8_DERIVATIVES_DISCOVERY", "V8_H2_MARK_INDEX_PREMIUM",
    }


def test_current_evidence_closes_derivatives_branch() -> None:
    payload = build_synthesis()
    assert payload["final_classification"] == CLOSE
    assert payload["classification_logic"]["tested_hypotheses_supported"] is False
    assert payload["classification_logic"]["untested_mechanism_supported_without_posthoc_mining"] is False


def test_continue_requires_complete_integrity_and_justified_mechanism() -> None:
    assert classify_branch(
        evidence_complete=True, integrity_passes=True,
        untested_mechanism_supported_without_posthoc_mining=True,
    ) == CONTINUE
    assert classify_branch(
        evidence_complete=False, integrity_passes=True,
        untested_mechanism_supported_without_posthoc_mining=True,
    ) == CLOSE
    assert classify_branch(
        evidence_complete=True, integrity_passes=False,
        untested_mechanism_supported_without_posthoc_mining=True,
    ) == CLOSE
    assert classify_branch(
        evidence_complete=True, integrity_passes=True,
        untested_mechanism_supported_without_posthoc_mining=False,
    ) == CLOSE


def test_discovery_and_independent_validation_are_not_conflated() -> None:
    payload = build_synthesis()
    evidence = {item["stage"]: item for item in payload["evidence"]}
    discovery = evidence["V8_DERIVATIVES_DISCOVERY"]
    h2 = evidence["V8_H2_MARK_INDEX_PREMIUM"]
    assert discovery["classification"] == "WEAK_OR_UNSTABLE_SIGNAL"
    assert discovery["model_beats_baseline"] is False
    assert h2["evidence_type"] == "INDEPENDENT_OUTCOME_UNSEEN_STRATEGY_VALIDATION"
    assert h2["classification"] == "NOT_SUPPORTED"
    assert payload["discovery_to_validation_reconciliation"]["independent_validation_supported"] is False


def test_h2_forensic_interpretation_is_complete() -> None:
    h2 = build_synthesis()["evidence"][-1]
    assert float(h2["forensic_raw_selection_statistic"]) < 0
    assert float(h2["forensic_conditional_centered_statistic"]) > 0
    assert h2["forensic_interpretation"] == (
        "CONDITIONAL_WITHIN_BLOCK_SELECTION_ASSOCIATION_NOT_AGGREGATE_GROSS_OUTPERFORMANCE"
    )


def test_holdout_access_is_absent_and_metadata_difference_is_visible() -> None:
    integrity = build_synthesis()["integrity"]
    assert integrity["all_stages_report_no_blind_holdout_access"] is True
    assert integrity["artifact_holdout_metadata_matches_authoritative"] == {
        "h0_funding": True,
        "h1_open_interest": False,
        "discovery": True,
        "h2_validation": True,
    }


def test_synthesis_id_and_writing_are_deterministic(tmp_path) -> None:
    first = build_synthesis()
    second = build_synthesis()
    assert first == second
    assert first["synthesis_id"] == first["definition_sha256"][:16]
    path = write_synthesis(output_root=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == first
    assert write_synthesis(output_root=tmp_path) == path


def test_manifest_identity_rejects_tampering() -> None:
    payload = {"run_id": "0" * 16, "definition_sha256": "0" * 64, "definition": {}}
    with pytest.raises(ValueError, match="identity"):
        verify_manifest(payload, "0" * 16)
