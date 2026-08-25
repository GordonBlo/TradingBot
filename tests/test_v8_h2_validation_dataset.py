from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.cli import build_v8_h2_validation_dataset as cli
from src.hypotheses.manifest import (
    BlindHoldoutRegistration,
    ConsumedDatasetRange,
    HoldoutStatus,
    ResearchManifest,
)
from src.research.v8_h2_validation_dataset import (
    TimeRange,
    _dataset_id,
    archive_plan,
    subtract_ranges,
    validation_plan,
)
from src.derivatives.models import utc_timestamp


UTC = timezone.utc


def moment(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def frozen_manifest(*, locked: bool = True) -> ResearchManifest:
    holdout = BlindHoldoutRegistration(
        symbol="BTCUSDC", interval="15m", start=moment("2025-08-01T00:00:00"),
        end=moment("2026-02-01T00:00:00"),
        status=HoldoutStatus.LOCKED_BLIND_HOLDOUT if locked else HoldoutStatus.REVEALED,
        registered_at=moment("2026-08-19T00:00:00"),
    )
    return ResearchManifest(
        project_version="test", baseline_name="test", baseline_version="test", source_research_run_id="test",
        creation_timestamp=moment("2026-08-19T00:00:00"), hypotheses=(),
        consumed_dataset_ranges=(
            ConsumedDatasetRange("BTCUSDC", "15m", moment("2023-03-12T06:30:00"), moment("2025-08-01T00:00:00"), "old"),
            ConsumedDatasetRange("BTCUSDC", "15m", moment("2026-02-01T00:00:00"), moment("2026-08-18T00:00:00"), "new"),
        ),
        holdout_status=holdout.status, blind_holdout=holdout,
    )


def test_plan_selects_longest_outcome_unseen_interval_and_excludes_holdout() -> None:
    consumed, holdout, eligible, selected = validation_plan(frozen_manifest())
    assert selected == TimeRange(moment("2020-01-01T00:00:00"), moment("2022-09-29T03:00:00"))
    assert eligible[-1] == TimeRange(moment("2026-08-18T00:00:00"), moment("2026-08-24T00:00:00"))
    assert all(not selected.overlaps(item) for item in consumed)
    assert not selected.overlaps(holdout)


def test_subtract_ranges_splits_without_using_market_or_outcome_values() -> None:
    available = (TimeRange(moment("2020-01-01T00:00:00"), moment("2020-01-10T00:00:00")),)
    blocked = (TimeRange(moment("2020-01-03T00:00:00"), moment("2020-01-05T00:00:00")),)
    assert subtract_ranges(available, blocked) == (
        TimeRange(moment("2020-01-01T00:00:00"), moment("2020-01-03T00:00:00")),
        TimeRange(moment("2020-01-05T00:00:00"), moment("2020-01-10T00:00:00")),
    )


def test_holdout_must_remain_locked() -> None:
    with pytest.raises(ValueError, match="LOCKED"):
        validation_plan(frozen_manifest(locked=False))


def test_minimal_archive_plan_is_only_spot_mark_and_index() -> None:
    _, _, _, selected = validation_plan(frozen_manifest())
    planned = archive_plan(selected, raw_root="ignored")
    assert len(planned) == 99
    assert {item.source for item in planned} == {"spot_klines", "markPriceKlines", "indexPriceKlines"}
    assert all("monthly" in item.location.url for item in planned)


def test_dataset_id_is_deterministic() -> None:
    payload = {"selected_interval": {"start": "2020-01-01T00:00:00+00:00"}, "candidate_count": 0}
    assert _dataset_id(payload) == _dataset_id(dict(reversed(tuple(payload.items()))))


def test_official_microsecond_kline_timestamp_is_explicitly_normalized() -> None:
    assert utc_timestamp("1787011200000000", unit="microseconds") == moment("2026-08-18T00:00:00")


def test_cli_defaults_to_plan_only(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli.ResearchManifestStore, "load", lambda _: frozen_manifest())
    monkeypatch.setattr(cli, "build_validation_dataset", lambda **_: pytest.fail("must not acquire"))
    assert cli.main([]) == 0
    assert "PLAN COMPLETE — NO VALIDATION DATA ACQUIRED" in capsys.readouterr().out
