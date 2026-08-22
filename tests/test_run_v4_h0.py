from copy import deepcopy

import pytest

from src.cli.run_v4_h0 import (
    main,
    validate_v4_h0_implementation,
    validate_v4_h0_manifest,
)
from src.research.v4_breakout_preregistration import build_manifest


def test_v4_h0_guard_accepts_frozen_manifest_and_implementation() -> None:
    manifest = build_manifest()

    validate_v4_h0_manifest(manifest)
    validate_v4_h0_implementation(manifest)


def test_v4_h0_guard_rejects_changed_definition() -> None:
    manifest = deepcopy(build_manifest())
    manifest["entry"]["lookback_bars"] = 21

    with pytest.raises(ValueError, match="frozen definition"):
        validate_v4_h0_manifest(manifest)


def test_v4_h0_replay_is_refused_without_dry_run() -> None:
    with pytest.raises(SystemExit, match="REPLAY DISABLED"):
        main([])
