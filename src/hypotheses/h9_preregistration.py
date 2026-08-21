"""Immutable preregistration for H9 Early Failure Exit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from src.research.multiregime.manifest import plain


@dataclass(frozen=True, slots=True)
class H9Definition:
    hypothesis_id: str = "H9"
    name: str = "Early Failure Exit"

    adverse_trigger_r: Decimal = Decimal("0.5")
    observation_bars: int = 4

    trigger_semantics: str = (
        "CLOSED_CANDLE_LOW_REACHES_ENTRY_MINUS_0.5R"
    )
    execution_timing: str = "NEXT_BAR_OPEN"

    preserve_h5_q25_entries: bool = True
    preserve_original_stop_before_execution: bool = True
    preserve_original_take_profit_before_execution: bool = True

    partial_exit: bool = False
    trailing_stop: bool = False

    def __post_init__(self) -> None:
        if self.adverse_trigger_r != Decimal("0.5"):
            raise ValueError(
                "H9 adverse trigger is frozen at -0.5R."
            )

        if self.observation_bars != 4:
            raise ValueError(
                "H9 observation window is frozen at 4 bars."
            )

        if self.execution_timing != "NEXT_BAR_OPEN":
            raise ValueError(
                "H9 execution timing is frozen at NEXT_BAR_OPEN."
            )

        if not self.preserve_h5_q25_entries:
            raise ValueError(
                "H9 must preserve frozen H5_Q25 entry logic."
            )

        if not self.preserve_original_stop_before_execution:
            raise ValueError(
                "Original stop must remain active before H9 execution."
            )

        if not self.preserve_original_take_profit_before_execution:
            raise ValueError(
                "Original target must remain active before H9 execution."
            )

        if self.partial_exit:
            raise ValueError(
                "H9 does not use partial exits."
            )

        if self.trailing_stop:
            raise ValueError(
                "H9 does not use trailing stops."
            )


@dataclass(frozen=True, slots=True)
class H9SupportCriteria:
    minimum_eligible_windows: int = 11
    consistency_percent_required: Decimal = Decimal("60")

    minimum_trade_count_ratio: Decimal = Decimal("0.80")
    minimum_baseline_entry_match_ratio: Decimal = Decimal("0.80")

    maximum_drawdown_worse_ratio: Decimal = Decimal("1.25")

    def __post_init__(self) -> None:
        frozen = {
            "minimum_eligible_windows": 11,
            "consistency_percent_required": Decimal("60"),
            "minimum_trade_count_ratio": Decimal("0.80"),
            "minimum_baseline_entry_match_ratio": Decimal("0.80"),
            "maximum_drawdown_worse_ratio": Decimal("1.25"),
        }

        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} is frozen at {expected} for H9."
                )


@dataclass(frozen=True, slots=True)
class H9Preregistration:
    version: str

    source_v33_run_id: str
    source_v331_run_id: str
    source_early_failure_sha256: str

    definition: H9Definition
    criteria: H9SupportCriteria

    dataset_status: str = "CONSUMED_RESEARCH_DATA"

    reference_candidate: str = "H5_Q25"
    reference_trade_count: int = 197

    blind_holdout_status: str = "LOCKED_BLIND_HOLDOUT"
    blind_holdout_revealed: bool = False
    blind_holdout_consumed: bool = False
    blind_holdout_evaluated: bool = False


class H9PreregistrationStore:
    def __init__(
        self,
        root: str | Path = "research/h9",
        *,
        clock: Callable[[], datetime] = (
            lambda: datetime.now(timezone.utc)
        ),
    ) -> None:
        self.root = Path(root)
        self._clock = clock

    def prepare(
        self,
        *,
        source_v33_run_id: str,
        source_v331_run_id: str,
        source_early_failure_sha256: str,
    ) -> tuple[str, Path, dict]:
        registration = H9Preregistration(
            version="3.3.3",
            source_v33_run_id=source_v33_run_id,
            source_v331_run_id=source_v331_run_id,
            source_early_failure_sha256=(
                source_early_failure_sha256
            ),
            definition=H9Definition(),
            criteria=H9SupportCriteria(),
        )

        configuration = {
            **asdict(registration),

            "rationale": {
                "early_failure_trades": 85,
                "early_failure_percent": (
                    "43.14720812182741116751269036"
                ),
                "early_failure_gross_expectancy_r": (
                    "-0.4474553959403378918893699656"
                ),
                "early_failure_net_expectancy_r": (
                    "-0.9819638249747073323629158309"
                ),
                "final_positive_percent": (
                    "15.29411764705882352941176471"
                ),
                "strict_later_positive_two_r_percent": (
                    "15.29411764705882352941176471"
                ),
                "same_bar_positive_two_r_ambiguous": 0,
            },

            "comparison_policy": {
                "reference": "H5_Q25",

                "combined_frictionless_expectancy": (
                    "H9 > H5_Q25"
                ),
                "combined_net_expectancy": (
                    "H9 > H5_Q25"
                ),
                "profit_factor": (
                    "H9 >= H5_Q25"
                ),

                "frictionless_window_consistency": (
                    "H9 beats H5_Q25 in >=60% eligible windows"
                ),
                "net_window_consistency": (
                    "H9 beats H5_Q25 in >=60% eligible windows"
                ),

                "trade_count_ratio": (
                    "H9 trades >=80% of H5_Q25 trades"
                ),
                "baseline_entry_match_ratio": (
                    ">=80% of frozen baseline entry signals remain matched"
                ),

                "maximum_drawdown": (
                    "H9 <=1.25 times H5_Q25 maximum drawdown"
                ),

                "cost_stress": (
                    "H9 net expectancy >= H5_Q25 "
                    "under identical doubled costs"
                ),
            },

            "next_stage_eligibility": {
                "all_support_gates_required": True,
                "combined_net_expectancy_r": "> 0",
                "profit_factor": "> 1",
                "positive_net_windows_percent": ">= 60",
            },

            "required_counterfactual_diagnostics": [
                "h9_triggered_trades",
                "h9_exit_trades",
                "baseline_stop_losses_improved",
                "baseline_trend_exits_improved",
                "baseline_take_profits_cut_early",
                "matched_baseline_entries",
                "unmatched_baseline_entries",
                "new_h9_entries",
                "gross_R_saved_or_lost",
                "net_R_saved_or_lost",
                "average_net_R_of_h9_exits",
            ],

            "research_integrity": {
                "parameter_search": False,
                "alternative_adverse_R_values": False,
                "alternative_observation_windows": False,
                "entry_logic_changes": False,
                "stop_parameter_changes": False,
                "target_changes": False,
                "partial_exit_research": False,
                "trailing_stop_research": False,
                "blind_holdout_access": False,
            },
        }

        canonical = json.dumps(
            plain(configuration),
            sort_keys=True,
            separators=(",", ":"),
        )

        configuration_sha256 = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

        run_id = configuration_sha256[:16]

        path = self.root / run_id / "manifest.json"

        payload = {
            "run_id": run_id,
            "configuration_sha256": configuration_sha256,
            "created_at": self._clock().astimezone(timezone.utc),
            **configuration,
        }

        serialized = (
            json.dumps(
                plain(payload),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        if path.exists():
            existing = json.loads(
                path.read_text(encoding="utf-8")
            )

            if (
                existing.get("configuration_sha256")
                != configuration_sha256
            ):
                raise ValueError(
                    "Existing H9 preregistration is not immutable."
                )

            payload = existing

        else:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = path.with_suffix(".json.tmp")

            try:
                temporary.write_text(
                    serialized,
                    encoding="utf-8",
                )
                temporary.replace(path)

            finally:
                if temporary.exists():
                    temporary.unlink()

        return run_id, path, payload