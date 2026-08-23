from copy import deepcopy
from pathlib import Path

import pytest

import src.cli.run_v7_h0 as guard
from src.cli.run_v7_h0 import (
    main,
    validate_eligible_windows,
    validate_v7_dataset_manifest,
    validate_v7_h0_implementation,
    validate_v7_h0_manifest,
)
from src.research.v7_orderflow_preregistration import build_manifest
from src.strategy.v7_orderflow_confirmation import (
    V7OrderFlowConfirmationStrategy,
)


DATASET_MANIFEST = Path(
    "data/orderflow/aggregated/15m/BTCUSDC/dataset_manifest.json"
)


def dataset_manifest() -> dict:
    return guard._load_json(DATASET_MANIFEST)


def test_v7_h0_guard_accepts_valid_frozen_state() -> None:
    manifest = build_manifest()

    validate_v7_h0_manifest(manifest)
    validate_v7_dataset_manifest(dataset_manifest())
    validate_v7_h0_implementation(manifest)
    validate_eligible_windows({window_id: 1 for window_id in guard.EXPECTED_WINDOW_IDS})


def test_v7_h0_guard_rejects_preregistration_run_id_mismatch() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"

    with pytest.raises(ValueError, match="run ID"):
        validate_v7_h0_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_id", "wrong", "dataset ID"),
        ("dataset_definition_sha256", "wrong", "dataset SHA-256"),
    ),
)
def test_v7_h0_guard_rejects_dataset_identity_mismatch(
    field: str, value: str, message: str
) -> None:
    payload = dataset_manifest()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_v7_dataset_manifest(payload)


def test_v7_h0_guard_rejects_dataset_coverage_or_reconciliation_mismatch() -> None:
    payload = dataset_manifest()
    payload["coverage"]["missing"] = 1

    with pytest.raises(ValueError, match="coverage reconciliation"):
        validate_v7_dataset_manifest(payload)


def test_v7_h0_guard_rejects_holdout_coverage_overlap() -> None:
    payload = dataset_manifest()
    payload["safe_ranges"][0] = [
        "2025-08-01T00:00:00+00:00",
        "2025-08-01T00:15:00+00:00",
    ]

    with pytest.raises(ValueError, match="overlaps the blind holdout"):
        validate_v7_dataset_manifest(payload)


def test_v7_h0_guard_rejects_threshold_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        V7OrderFlowConfirmationStrategy,
        "QUOTE_IMBALANCE_THRESHOLD",
        guard.Decimal("0.01"),
    )

    with pytest.raises(ValueError, match="strategy constants"):
        validate_v7_h0_implementation(build_manifest())


def test_v7_h0_guard_rejects_timestamp_semantics_mismatch() -> None:
    manifest = deepcopy(build_manifest())
    manifest["causality"]["nearest_neighbor_lookup_allowed"] = True

    with pytest.raises(ValueError, match="timestamp/causality"):
        validate_v7_h0_implementation(manifest)


def test_v7_h0_guard_rejects_subset_invariant_mismatch() -> None:
    manifest = deepcopy(build_manifest())
    manifest["reference_relationship"]["may_create_non_v6_entry"] = True

    with pytest.raises(ValueError, match="subset invariant"):
        validate_v7_h0_implementation(manifest)


def test_v7_h0_guard_rejects_wrong_eligible_windows() -> None:
    wrong = {window_id: 1 for window_id in guard.EXPECTED_WINDOW_IDS[:-1]}

    with pytest.raises(ValueError, match="exactly the 11"):
        validate_eligible_windows(wrong)


def test_v7_h0_real_execution_remains_disabled() -> None:
    with pytest.raises(SystemExit, match="REPLAY DISABLED"):
        main([])


def test_v7_h0_dry_run_passes_without_replay(capsys) -> None:
    assert main(["--dry-run"]) == 0
    assert capsys.readouterr().out.rstrip().endswith(
        "DRY RUN PASSED — NO V7-H0 REPLAY EXECUTED"
    )
