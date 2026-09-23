"""Protect the immutable closure using versioned metadata only; never open outcomes."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "research/v10_microstructure_economic_discovery/a296e5ed304640d7"
CLOSURE_SHA256 = "a941cf4d84cf59c4c7ec097e89923ffdc0daf89a0a109ae985b8de619f00050a"
REPORT_SHA256 = "727333c40cc7fe385222cc0728383c5a7f16bbd3cf720acc478b6fc4155c6c84"


def test_immutable_closure_binds_the_frozen_experiment_and_sealed_report():
    raw = (DIRECTORY / "synthesis.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CLOSURE_SHA256
    closure = json.loads(raw)
    manifest = json.loads((DIRECTORY / "manifest.json").read_bytes())
    assert closure["preregistration_id"] == manifest["preregistration_id"]
    assert closure["definition_sha256"] == manifest["definition_sha256"]
    assert closure["sealed_report_sha256"] == REPORT_SHA256
    assert closure["final_classification"] == "NO_STABLE_COMBINED_SIGNAL"
    assert closure["forensic_audit"]["verdict"] == "PASS"
    assert closure["forensic_audit"]["invalidating_issue_identified"] is False


def test_failed_discovery_remains_closed_consumed_and_cannot_advance():
    closure = json.loads((DIRECTORY / "synthesis.json").read_bytes())
    assert len(closure["information_gates"]) == 7
    assert all(value is False for value in closure["information_gates"].values())
    assert closure["economic_gates"] == {
        "tail_coverage": True,
        "tail_base_net": False,
        "anchor_coverage": True,
        "anchor_base_net": False,
    }
    assert closure["hypothesis_status"] == "CLOSED"
    assert closure["evidence_role"] == "DISCOVERY_ONLY"
    assert closure["dataset_status"] == "CONSUMED_DISCOVERY_EVIDENCE"
    assert closure["reuse_as_fresh_validation_permitted"] is False
    assert closure["blind_holdout"] == {"status": "LOCKED", "accessed_for_closure": False}
    assert closure["advancement"] == {
        "strategy_validation": False,
        "paper_trading": False,
        "live_trading": False,
    }
    assert closure["profitable_strategy_validated"] is False
