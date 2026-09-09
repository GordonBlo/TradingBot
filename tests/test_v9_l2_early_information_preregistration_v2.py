from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from src.research.v9_l2_early_information_preregistration_v2 import (
    CIRCULAR_SHIFT_EXCLUSION_SAMPLES,
    FROZEN_FEATURES,
    MANIFEST_ROOT,
    PROTOCOL_VERSION,
    SUPERSEDES_RUN_ID,
    build_manifest,
    prospective_cutoff,
    verify_manifest,
    write_or_verify_manifest,
)


ORIGINAL_MANIFEST = Path(
    "research/v9_l2_early_information/e1cc53ee9f55cb9e/manifest.json"
)
ORIGINAL_MANIFEST_SHA256 = "919b1d160d4504d227e9a328bce48e25fde69b46b14202358b5c5bb49a1c9473"


def test_original_v1_preregistration_remains_byte_for_byte_unchanged() -> None:
    assert hashlib.sha256(ORIGINAL_MANIFEST.read_bytes()).hexdigest() == ORIGINAL_MANIFEST_SHA256


def test_v2_cutoff_is_strictly_next_full_utc_quarter_hour() -> None:
    assert prospective_cutoff(datetime(2026, 9, 9, 12, 15, tzinfo=UTC)) == datetime(
        2026, 9, 9, 12, 30, tzinfo=UTC
    )
    assert prospective_cutoff(datetime(2026, 9, 9, 12, 29, 59, tzinfo=UTC)) == datetime(
        2026, 9, 9, 12, 30, tzinfo=UTC
    )


def test_v2_manifest_freezes_exact_pre_outcome_amendments() -> None:
    manifest = build_manifest(datetime(2026, 9, 9, 12, 16, tzinfo=UTC))
    definition = manifest["definition"]
    assert definition["schema_version"] == 2
    assert definition["protocol_version"] == PROTOCOL_VERSION
    assert definition["supersedes_run_id"] == SUPERSEDES_RUN_ID
    assert definition["features"] == list(FROZEN_FEATURES)
    assert definition["targets"]["primary_horizon_seconds"] == 30
    assert definition["targets"]["secondary_descriptive_horizons_seconds"] == [1, 5, 60]
    assert definition["targets"]["secondary_horizons_influence_primary_classification"] is False
    assert definition["readiness_gate"] == {
        "minimum_eligible_closed_sessions": 8,
        "minimum_total_eligible_hours": 20,
        "minimum_utc_dates": 2,
        "all_sessions_pass_integrity": True,
    }
    null = definition["evaluation"]["null"]
    assert null["method"] == "session-wise circular primary-target shifts"
    assert null["valid_offset_rule"] == "min(offset, session_sample_count-offset) > 60"
    assert CIRCULAR_SHIFT_EXCLUSION_SAMPLES == 60
    assert definition["evaluation"]["undefined_required_null_statistic"] == (
        "INTEGRITY_FAILURE_NO_DROPPING"
    )
    assert definition["evaluation"]["required_valid_null_statistics"] == 1_000
    assert definition["sampling"]["all_missing_training_feature"] == (
        "FAIL_FOLD_AND_DIAGNOSTIC"
    )
    assert definition["artifact_binding"]["required"] is True
    assert definition["restrictions"]["blind_holdout"]["status"] == "LOCKED"
    verify_manifest(manifest)


def test_checked_in_v2_manifest_is_unique_and_valid() -> None:
    paths = sorted(MANIFEST_ROOT.glob("*/manifest.json"))
    assert len(paths) == 1
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    verify_manifest(manifest)
    assert paths[0].parent.name == manifest["run_id"]


def test_v2_manifest_write_is_deterministic_and_never_retimestamps(tmp_path) -> None:
    created = datetime(2026, 9, 9, 12, 16, tzinfo=UTC)
    path, first = write_or_verify_manifest(tmp_path, now=lambda: created)
    second_path, second = write_or_verify_manifest(
        tmp_path, now=lambda: datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert path == second_path
    assert first == second == json.loads(path.read_text(encoding="utf-8"))
