"""Immutable preregistration for H8 +1R price break-even protection."""

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
class H8Definition:
    hypothesis_id: str = "H8"
    name: str = "+1R Price Break-Even Protection"

    trigger_r: Decimal = Decimal("1.0")

    # H8 is activated only AFTER the trigger candle is closed.
    # The protective stop can therefore affect only later bars.
    activation_timing: str = "NEXT_BAR_AFTER_CLOSED_TRIGGER"

    protective_stop: str = "ENTRY_PRICE"

    # Important: this is price break-even, not net break-even
    # after commissions/slippage.
    protective_stop_semantics: str = "PRICE_BREAK_EVEN_NOT_NET_BREAK_EVEN"

    take_profit_r: Decimal = Decimal("2.0")

    preserve_h5_q25_entries: bool = True
    preserve_original_atr_stop_until_activation: bool = True
    preserve_take_profit: bool = True
    partial_exit: bool = False

    ambiguous_bar_policy: str = "STOP_FIRST"

    def __post_init__(self) -> None:
        if self.trigger_r != Decimal("1.0"):
            raise ValueError("H8 trigger is frozen at +1R.")

        if self.take_profit_r != Decimal("2.0"):
            raise ValueError("H8 take profit is frozen at +2R.")

        if self.activation_timing != "NEXT_BAR_AFTER_CLOSED_TRIGGER":
            raise ValueError("H8 activation timing is frozen.")

        if self.protective_stop != "ENTRY_PRICE":
            raise ValueError("H8 protective stop must remain entry price.")

        if not self.preserve_h5_q25_entries:
            raise ValueError("H8 must preserve frozen H5_Q25 entry logic.")

        if not self.preserve_original_atr_stop_until_activation:
            raise ValueError("Original stop must remain unchanged before activation.")

        if not self.preserve_take_profit:
            raise ValueError("Original +2R target must remain unchanged.")

        if self.partial_exit:
            raise ValueError("H8 does not use partial exits.")

        if self.ambiguous_bar_policy != "STOP_FIRST":
            raise ValueError("H8 preserves V2 STOP_FIRST semantics.")


@dataclass(frozen=True, slots=True)
class H8SupportCriteria:
    minimum_eligible_windows: int = 11

    consistency_percent_required: Decimal = Decimal("60")

    minimum_trade_count_ratio: Decimal = Decimal("0.80")

    maximum_drawdown_worse_ratio: Decimal = Decimal("1.25")

    def __post_init__(self) -> None:
        frozen = {
            "minimum_eligible_windows": 11,
            "consistency_percent_required": Decimal("60"),
            "minimum_trade_count_ratio": Decimal("0.80"),
            "maximum_drawdown_worse_ratio": Decimal("1.25"),
        }

        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} is frozen at {expected} for H8 research."
                )


@dataclass(frozen=True, slots=True)
class H8Preregistration:
    version: str
    source_v33_run_id: str
    source_v331_diagnostic_run_id: str

    definition: H8Definition
    criteria: H8SupportCriteria

    dataset_status: str = "CONSUMED_RESEARCH_DATA"

    reference_candidate: str = "H5_Q25"
    reference_trade_count: int = 197

    blind_holdout_status: str = "LOCKED_BLIND_HOLDOUT"
    blind_holdout_revealed: bool = False
    blind_holdout_consumed: bool = False
    blind_holdout_evaluated: bool = False


class H8PreregistrationStore:
    """Create an immutable H8 preregistration before H8 is implemented."""

    def __init__(
        self,
        root: str | Path = "research/h8",
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
        source_v331_diagnostic_run_id: str,
    ) -> tuple[str, Path, dict]:
        definition = H8Definition()
        criteria = H8SupportCriteria()

        registration = H8Preregistration(
            version="3.3.2",
            source_v33_run_id=source_v33_run_id,
            source_v331_diagnostic_run_id=source_v331_diagnostic_run_id,
            definition=definition,
            criteria=criteria,
        )

        configuration = {
            **asdict(registration),

            "rationale": {
                "losing_trades_reached_positive_1R_percent": (
                    "21.25984251968503937007874016"
                ),
                "stop_loss_trades_reached_positive_1R_percent": (
                    "20.20202020202020202020202020"
                ),
                "interpretation": (
                    "A meaningful subset of losing H5_Q25 trades first "
                    "achieved +1R favorable excursion before reversing."
                ),
            },

            "comparison_policy": {
                "reference": "H5_Q25",
                "combined_frictionless_expectancy": (
                    "H8 > H5_Q25"
                ),
                "combined_net_expectancy": (
                    "H8 > H5_Q25"
                ),
                "profit_factor": (
                    "H8 >= H5_Q25"
                ),
                "frictionless_window_consistency": (
                    "H8 beats H5_Q25 in >=60% eligible windows"
                ),
                "net_window_consistency": (
                    "H8 beats H5_Q25 in >=60% eligible windows"
                ),
                "trade_count_ratio": (
                    "H8 trades >= 80% of H5_Q25 trades"
                ),
                "maximum_drawdown": (
                    "H8 <= 1.25 times H5_Q25 maximum drawdown"
                ),
                "cost_stress": (
                    "H8 net expectancy >= H5_Q25 under identical doubled costs"
                ),
            },

            "next_stage_eligibility": {
                "all_support_gates_required": True,
                "combined_net_expectancy_r": "> 0",
                "profit_factor": "> 1",
                "positive_net_windows_percent": ">= 60",
            },

            "required_counterfactual_diagnostics": [
                "protective_stop_activated_trades",
                "protective_stop_exit_trades",
                "baseline_stop_losses_improved",
                "baseline_take_profits_cut_early",
                "net_R_saved_by_protection",
                "gross_R_saved_by_protection",
                "average_R_of_protective_exits",
            ],

            "research_integrity": {
                "parameter_search": False,
                "alternative_trigger_R_values": False,
                "partial_exit_research": False,
                "trailing_stop_research": False,
                "entry_logic_changes": False,
                "target_changes": False,
                "ATR_stop_changes_before_activation": False,
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
                    "Existing H8 preregistration is not immutable."
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