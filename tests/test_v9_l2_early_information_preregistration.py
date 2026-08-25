from __future__ import annotations

import json
from datetime import UTC, datetime

from src.research.v9_l2_early_information_preregistration import (
    FROZEN_FEATURES,
    build_manifest,
    prospective_cutoff,
    verify_manifest,
    write_or_verify_manifest,
)


def test_cutoff_is_strictly_next_full_utc_quarter_hour() -> None:
    assert prospective_cutoff(datetime(2026, 8, 25, 12, 7, 9, tzinfo=UTC)) == datetime(
        2026, 8, 25, 12, 15, tzinfo=UTC
    )
    assert prospective_cutoff(datetime(2026, 8, 25, 12, 15, tzinfo=UTC)) == datetime(
        2026, 8, 25, 12, 30, tzinfo=UTC
    )


def test_manifest_freezes_exact_protocol_and_is_deterministic() -> None:
    created = datetime(2026, 8, 25, 12, 7, 9, tzinfo=UTC)
    left = build_manifest(created)
    right = build_manifest(created)
    assert left == right
    assert len(left["run_id"]) == 16
    definition = left["definition"]
    assert definition["features"] == list(FROZEN_FEATURES)
    assert len(definition["features"]) == 15
    assert definition["targets"]["primary_horizon_seconds"] == 30
    assert definition["targets"]["secondary_descriptive_horizons_seconds"] == [1, 5, 60]
    assert definition["readiness_gate"] == {
        "minimum_eligible_closed_sessions": 8,
        "minimum_total_eligible_hours": 20,
        "minimum_utc_dates": 2,
        "all_sessions_pass_integrity": True,
    }
    evaluation = definition["evaluation"]
    assert evaluation["ridge_alpha"] == 1.0
    assert evaluation["permutations"] == 1_000
    assert evaluation["significance_alpha"] == 0.05
    assert definition["restrictions"]["blind_holdout"]["status"] == "LOCKED"
    verify_manifest(left)


def test_manifest_write_is_deterministic_and_never_retimestamps(tmp_path) -> None:
    created = datetime(2026, 8, 25, 12, 7, 9, tzinfo=UTC)
    path, first = write_or_verify_manifest(tmp_path, now=lambda: created)
    second_path, second = write_or_verify_manifest(
        tmp_path, now=lambda: datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert path == second_path
    assert first == second == json.loads(path.read_text(encoding="utf-8"))
