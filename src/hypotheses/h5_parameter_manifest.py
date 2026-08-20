"""Immutable preregistration for V3.3 controlled H5 parameter research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from src.backtest.models import BacktestConfig
from src.hypotheses.h5_parameter_research import (
    h5_parameter_registry,
    validate_h5_parameter_registry,
)
from src.research.multiregime.manifest import (
    PreparedMultiRegimeManifest,
    plain,
)
from src.strategy.models import TrendMomentumConfig


@dataclass(frozen=True, slots=True)
class H5ParameterSupportCriteria:
    minimum_eligible_windows: int = 6
    consistency_percent_required: Decimal = Decimal("60")
    trade_count_ratio_required: Decimal = Decimal("0.50")
    maximum_drawdown_worse_ratio: Decimal = Decimal("1.25")

    def __post_init__(self) -> None:
        fixed = {
            "minimum_eligible_windows": 6,
            "consistency_percent_required": Decimal("60"),
            "trade_count_ratio_required": Decimal("0.50"),
            "maximum_drawdown_worse_ratio": Decimal("1.25"),
        }

        for name, expected in fixed.items():
            actual = getattr(self, name)

            if actual != expected:
                raise ValueError(
                    f"{name} is frozen at {expected} for V3.3."
                )


class H5ParameterManifestStore:
    """Write one immutable preregistration before any V3.3 replay."""

    def __init__(
        self,
        root: str | Path = "research/v3_3_h5",
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
        symbol: str,
        interval: str,
        source_mechanism_run_id: str,
        source_mechanism_manifest_sha256: str,
        source_multiregime_run_id: str,
        source_r_audit_sha256: str,
        strategy_config: TrendMomentumConfig,
        backtest_config: BacktestConfig,
        criteria: H5ParameterSupportCriteria | None = None,
    ) -> PreparedMultiRegimeManifest:
        validate_h5_parameter_registry()

        support = criteria or H5ParameterSupportCriteria()

        configuration = {
            "project_version": "3.3",
            "research_type": "CONTROLLED_H5_PARAMETER_RESEARCH",

            "symbol": symbol.strip().upper(),
            "interval": interval.strip(),

            "source": {
                "mechanism_run_id": source_mechanism_run_id,
                "mechanism_manifest_sha256": (
                    source_mechanism_manifest_sha256
                ),
                "multiregime_run_id": source_multiregime_run_id,
                "r_normalized_audit_sha256": (
                    source_r_audit_sha256
                ),
            },

            "frozen_reference": "H5_Q25",

            "parameter_scope": {
                "parameter": "H5 lower ATR percentile",
                "lookback": 100,
                "upper_percentile": "75",
                "only_parameter_allowed_to_change": True,
            },

            "candidates": [
                asdict(item)
                for item in h5_parameter_registry()
            ],

            "frozen_baseline_parameters": asdict(
                strategy_config
            ),

            "base_cost_assumptions": asdict(
                backtest_config
            ),

            "stress_cost_assumptions": {
                **asdict(backtest_config),
                "fee_bps": backtest_config.fee_bps * Decimal("2"),
                "slippage_bps": (
                    backtest_config.slippage_bps * Decimal("2")
                ),
            },

            "primary_metric": (
                "combined_net_expectancy_R_per_trade"
            ),

            "secondary_metrics": [
                "frictionless_expectancy_R_per_trade",
                "average_total_friction_R_per_trade",
                "profit_factor",
                "win_rate",
                "payoff_ratio",
                "maximum_drawdown",
                "trade_count",
                "window_consistency",
            ],

            "support_criteria": {
                **asdict(support),

                "combined_frictionless_expectancy": (
                    "candidate > H5_Q25"
                ),

                "combined_net_expectancy": (
                    "candidate > H5_Q25"
                ),

                "combined_profit_factor": (
                    "candidate >= H5_Q25"
                ),

                "frictionless_window_consistency": (
                    "candidate beats H5_Q25 in at least 60% "
                    "of eligible windows"
                ),

                "net_window_consistency": (
                    "candidate beats H5_Q25 in at least 60% "
                    "of eligible windows"
                ),

                "trade_count": (
                    "candidate >= 50% of H5_Q25 trades"
                ),

                "maximum_drawdown": (
                    "candidate <= 1.25 times H5_Q25"
                ),

                "stress_net_expectancy": (
                    "candidate >= H5_Q25 under identical "
                    "doubled execution costs"
                ),
            },

            "v3_4_eligibility": {
                "all_support_gates_required": True,
                "combined_net_expectancy_R": "> 0",
                "combined_profit_factor": "> 1",
                "positive_net_windows_percent": ">= 60",
            },

            "selection_policy": {
                "automatic_best_candidate_selection": False,
                "highest_return_alone_is_not_sufficient": True,
                "multiple_candidates_may_pass": True,
                "if_none_pass_stop": True,
            },

            "holdout_policy": {
                "required_status": "LOCKED_BLIND_HOLDOUT",
                "revealed": False,
                "consumed": False,
                "evaluated": False,
                "warmup_allowed": False,
            },

            "research_integrity": {
                "dataset_status": "CONSUMED_RESEARCH_DATA",
                "grid_search": False,
                "random_search": False,
                "automatic_optimization": False,
                "candidate_combinations": False,
                "ema_tuning": False,
                "rsi_tuning": False,
                "volume_tuning": False,
                "stop_tuning": False,
                "target_tuning": False,
                "lookback_tuning": False,
                "upper_percentile_tuning": False,
                "blind_holdout_reveal": False,
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
                    "Existing V3.3 H5 manifest is not immutable."
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

        return PreparedMultiRegimeManifest(
            run_id=run_id,
            configuration_sha256=configuration_sha256,
            path=path,
            payload=payload,
        )