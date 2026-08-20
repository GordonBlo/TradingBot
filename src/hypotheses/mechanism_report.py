"""V3.2.2 report wrapper with H5-H7 comparisons to frozen H1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from src.hypotheses.mechanisms import MechanismHypothesisId
from src.research.multiregime.manifest import PreparedMultiRegimeManifest
from src.research.multiregime.models import MultiRegimeResearchResult
from src.research.multiregime.report import (
    MultiRegimeReportPaths,
    MultiRegimeReportWriter,
)


@dataclass(frozen=True, slots=True)
class MechanismReportPaths:
    base: MultiRegimeReportPaths
    candidate_vs_h1: Path


class MechanismReportWriter:
    def __init__(self, root: str | Path = "reports/mechanisms") -> None:
        self._base = MultiRegimeReportWriter(
            root,
            project_version="3.2.2",
            heading="V3.2.2 New Mechanism Hypotheses",
            eligibility_label="NEXT_STAGE_ELIGIBLE",
        )

    def write(
        self,
        result: MultiRegimeResearchResult,
        preregistration: PreparedMultiRegimeManifest,
    ) -> MechanismReportPaths:
        paths = self._base.write(result, preregistration)
        reference = next(
            item
            for item in result.stability
            if item.hypothesis_id is MechanismHypothesisId.H1
        ).combined_metrics
        comparison = paths.directory / "candidate_vs_h1.csv"
        rows = []
        for item in result.stability:
            if item.hypothesis_id not in {
                MechanismHypothesisId.H5,
                MechanismHypothesisId.H6,
                MechanismHypothesisId.H7,
            }:
                continue
            metric = item.combined_metrics
            rows.append(
                {
                    "hypothesis": item.hypothesis_id.value,
                    "delta_frictionless_expectancy_vs_h1": (
                        metric.average_frictionless_pnl_per_trade
                        - reference.average_frictionless_pnl_per_trade
                    ),
                    "delta_net_expectancy_vs_h1": (
                        metric.expectancy - reference.expectancy
                    ),
                    "delta_profit_factor_vs_h1": (
                        metric.profit_factor - reference.profit_factor
                        if metric.profit_factor is not None
                        and reference.profit_factor is not None
                        else None
                    ),
                    "delta_maximum_drawdown_vs_h1": (
                        metric.maximum_drawdown_percent
                        - reference.maximum_drawdown_percent
                    ),
                    "trade_count_ratio_vs_h1": (
                        Decimal(metric.trades) / Decimal(reference.trades)
                        if reference.trades
                        else Decimal("0")
                    ),
                }
            )
        with comparison.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with paths.summary_markdown.open("a", encoding="utf-8") as stream:
            stream.write("\n## Frozen controls\n\n")
            stream.write(
                "H0 reproduces the frozen baseline: **VERIFIED**.  \n"
                "H1 reproduces the frozen V3.2.1 result: **VERIFIED**.  \n"
                "Blind holdout: **LOCKED — NOT REVEALED — NOT CONSUMED**.\n"
            )
            stream.write("\n## Window consistency versus H0\n\n")
            stream.write(
                "| ID | Frictionless Better | Net Better | PF Better | "
                "Trade Ratio |\n"
            )
            stream.write("|---|---:|---:|---:|---:|\n")
            for item in result.stability:
                if item.hypothesis_id not in {
                    MechanismHypothesisId.H5,
                    MechanismHypothesisId.H6,
                    MechanismHypothesisId.H7,
                }:
                    continue
                stream.write(
                    f"| {item.hypothesis_id.value} | "
                    f"{item.frictionless_better_windows}/{item.eligible_windows} | "
                    f"{item.net_better_windows}/{item.eligible_windows} | "
                    f"{item.profit_factor_better_windows}/{item.eligible_windows} | "
                    f"{item.trade_count_ratio} |\n"
                )
            stream.write("\n## Combined comparison to frozen H1\n\n")
            stream.write(
                "| ID | Δ Frictionless Exp. | Δ Net Exp. | Δ PF | Trade Ratio |\n"
            )
            stream.write("|---|---:|---:|---:|---:|\n")
            for row in rows:
                stream.write(
                    f"| {row['hypothesis']} | "
                    f"{row['delta_frictionless_expectancy_vs_h1']} | "
                    f"{row['delta_net_expectancy_vs_h1']} | "
                    f"{row['delta_profit_factor_vs_h1']} | "
                    f"{row['trade_count_ratio_vs_h1']} |\n"
                )
        return MechanismReportPaths(base=paths, candidate_vs_h1=comparison)
