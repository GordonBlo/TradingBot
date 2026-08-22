from copy import deepcopy

import pytest

from src.cli.run_v5_h0 import (
    main,
    validate_eligible_windows,
    validate_v5_h0_implementation,
    validate_v5_h0_manifest,
)
from src.research.v5_mean_reversion_preregistration import build_manifest
from src.strategy.v5_mean_reversion import V5MeanReversionStrategy


def test_v5_h0_guard_accepts_frozen_configuration() -> None:
    manifest = build_manifest()

    validate_v5_h0_manifest(manifest)
    validate_v5_h0_implementation(manifest)


def test_v5_h0_guard_rejects_manifest_run_id_mismatch() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"

    with pytest.raises(ValueError, match="run ID"):
        validate_v5_h0_manifest(manifest)


def test_v5_h0_guard_rejects_strategy_constant_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(V5MeanReversionStrategy, "REFERENCE_LOOKBACK_BARS", 21)

    with pytest.raises(ValueError, match="strategy constants"):
        validate_v5_h0_implementation(build_manifest())


def test_v5_h0_guard_rejects_wrong_eligible_window_count() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        validate_eligible_windows({str(index): {} for index in range(10)})


def test_v5_h0_guard_rejects_holdout_contamination() -> None:
    manifest = deepcopy(build_manifest())
    manifest["dataset_policy"]["blind_holdout"]["loaded"] = True

    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v5_h0_implementation(manifest)


def test_v5_h0_real_execution_remains_disabled() -> None:
    with pytest.raises(SystemExit, match="REPLAY DISABLED"):
        main([])
