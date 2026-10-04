"""Guard V11 closure with metadata hashes; never rerun calibration or load data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "research/v11_l2_entry_filter/synthesis.json"
CLOSURE_SHA256 = "fa0fb18113246388b3d3f24b0b3410fa0b393a815ad9396e7e9f89426fd8f4db"
RESULT_PATH = "research/v11_methodology_resolution/RESULT.json"
RESULT_SHA256 = "4e2a500735edf8d68926423c59ca8de506cc25892c8673dafe4cc6261ef807d5"
SCORE_SHA256 = "c2fb47fb459a97024ebedc12af0354d6c28792e2dce87c1207eba2ba5e4664e2"


def closure():
    raw = CLOSURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CLOSURE_SHA256
    return json.loads(raw)


def test_immutable_closure_preserves_the_bound_methodology_result():
    value = closure()
    assert value["artifact_type"] == "V11_IMMUTABLE_L2_ENTRY_FILTER_METHODOLOGY_CLOSURE"
    methodology = value["methodology"]
    assert methodology["artifact_bindings"][RESULT_PATH] == RESULT_SHA256
    raw = (ROOT / RESULT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    result = json.loads(raw)
    assert value["verdict"] == result["verdict"] == "METHODOLOGY_UNRESOLVED"
    assert methodology["selected_inference_method"] is result["selected_method"] is None
    assert methodology["selected_duration_days"] is result["prospective_duration_days"] is None
    assert methodology["numerical_success_gates_frozen"] is result["primary_gate_frozen"] is False
    assert methodology["calibration_status"] == "COMPLETED"
    assert methodology["all_three_prespecified_methods_failed_required_calibration"] is True
    assert set(methodology["methods"]) == {"CBB_LONG", "HAC_LONG", "SELF_NORMALIZED"}
    for name, method in methodology["methods"].items():
        assert method["status"] == "FAILED_REQUIRED_CALIBRATION"
        assert method["failed_scenario_cells"] == result["method_summary"][name]["failed_cells"] > 0
        assert method["total_scenario_cells"] == result["method_summary"][name]["total_cells"]
    # Ignored report hashes are checked by the closure audit; tests require only
    # repository metadata/source and the prepared static bundle, never outcomes.
    for name, expected in methodology["artifact_bindings"].items():
        if name.startswith("research/"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert methodology["artifact_bindings"][
        "reports/v11_methodology_resolution/calibration/artifact_hashes.json"
    ] == result["calibration_hash_manifest_sha256"]


def test_parent_score_filter_and_completed_engineering_remain_bound():
    value = closure()
    parent = value["fixed_parent"]
    assert parent["name"] == "V6_H0_MTF_CONTINUATION"
    assert parent["preregistration_id"] == "e1eef7bdd0c37ad4"
    assert parent["unchanged"] is True
    assert value["frozen_v9_score"]["unchanged"] is True
    assert value["frozen_v9_score"]["sha256"] == SCORE_SHA256
    assert hashlib.sha256((ROOT / value["frozen_v9_score"]["path"]).read_bytes()).hexdigest() == SCORE_SHA256
    assert value["filter"] == {"rule": "prediction > 0", "unchanged": True, "subset_only": True}
    assert value["engineering"]["status"] == "COMPLETED"
    assert value["engineering"]["readiness_result"] == "OFFLINE_INTEGRATION_TESTED"
    for group in (parent, value["engineering"]):
        for name, expected in group["artifact_bindings"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_deferred_branch_never_started_validation_and_cannot_advance():
    value = closure()
    assert value["research_status"] == "DEFERRED"
    assert value["branch_status"] == "CLOSED_FOR_NOW"
    assert value["prospective_validation"] == {
        "status": "NEVER_STARTED",
        "preregistration_created": False,
        "collection_started": False,
        "real_outcomes_evaluated": False,
    }
    assert value["evidence_status"] == {"V9": "CONSUMED", "V10": "CONSUMED"}
    assert value["v10_branch_status"] == "CLOSED"
    assert value["blind_holdout"] == {"status": "LOCKED", "accessed_for_closure": False}
    assert value["advancement"] == {
        "prospective_validation": False,
        "strategy_validation": False,
        "paper_trading": False,
        "live_trading": False,
    }
    assert all(flag is False for flag in value["closure_operations"].values())
    assert value["profitable_strategy_validated"] is False
    assert value["resume_policy"] == (
        "Do not resume this branch by adding methods or relaxing gates "
        "without a separately preregistered methodology study."
    )
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    v11_state = agents.split("V11 L2 entry filter:", 1)[1].split(
        "Do not reopen a CLOSED research branch", 1
    )[0]
    assert "METHODOLOGY_UNRESOLVED" in v11_state
    assert "DEFERRED / CLOSED_FOR_NOW" in v11_state
    assert "prospective V11 validation was NEVER_STARTED" in v11_state
    assert "separately preregistered methodology study" in v11_state
    assert "no profitable strategy has been validated" in v11_state
    assert "research/v11_l2_entry_filter/synthesis.json" in v11_state
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "V11 L2 entry filter engineering completed" in readme
    assert "Prospective V11 validation was never started" in readme
    assert "research/v11_l2_entry_filter/synthesis.json" in readme
