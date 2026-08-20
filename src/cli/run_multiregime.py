"""Run the frozen V3.2.1 H0-H4 suite across fixed historical regimes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.backtest.report import dataset_checksum
from src.cli.common import parse_utc_datetime
from src.config.settings import load_settings
from src.diagnostics.loader import ResearchRunLoader
from src.historical.dataset import HistoricalDataset
from src.historical.storage import HistoricalDatasetStore
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.market.intervals import next_open_time
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.manifest import MultiRegimeManifestStore
from src.research.multiregime.models import (
    PartitionKind,
    PartitionStatus,
    ResearchPartitionMetadata,
    ResearchRegion,
)
from src.research.multiregime.report import MultiRegimeReportWriter
from src.research.multiregime.runner import MultiRegimeResearchRunner
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.utils.logger import configure_logging, get_logger


SYMBOL = "BTCUSDC"
INTERVAL = "15m"
REQUESTED_EXPANSION_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
EXPANSION_END = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_START = EXPANSION_END
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)
CONSUMED_START = HOLDOUT_END
CONSUMED_END = datetime(2026, 8, 18, tzinfo=timezone.utc)
EXPECTED_PREVIOUS_H0_TRADES = 94


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pre-registered V3.2.1 multi-regime research offline. The fixed "
            "blind holdout is never loaded, used for warm-up, or evaluated."
        )
    )
    parser.add_argument("--manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--expansion-data-root", default="data/historical/multiregime_expansion"
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--research-root", default="reports/research")
    parser.add_argument("--research-run", default="adeef00722e9704d")
    parser.add_argument("--reports-root", default="reports/multiregime")
    parser.add_argument("--preregistration-root", default="research/multiregime")
    parser.add_argument(
        "--cost-stress",
        action="store_true",
        help="Evaluate the frozen candidates with exactly 2x fees and slippage too.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        settings = load_settings()
        if settings.trading_symbol != SYMBOL or settings.candle_interval != INTERVAL:
            raise ValueError("V3.2.1 is frozen to BTCUSDC 15m.")

        root_store = ResearchManifestStore(args.manifest)
        root_manifest = root_store.load()
        _require_locked_holdout(root_manifest)

        expansion_segments = _load_expansion(args.expansion_data_root)
        actual_expansion_start = expansion_segments[0].candles[0].timestamp
        expansion_candle_count = sum(
            len(segment.candles) for segment in expansion_segments
        )
        expansion_days = expansion_candle_count / 96
        if expansion_days < settings.multiregime_config().minimum_expansion_days:
            raise ValueError(
                "STOP: fewer than 365 days of usable BTCUSDC 15m expansion data exist; "
                "real multi-regime evaluation is refused."
            )

        consumed = _load_exact_range(
            args.consumed_data_root, CONSUMED_START, CONSUMED_END
        )
        _assert_holdout_absent(*expansion_segments, consumed)

        previous_loader = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        )
        previous = previous_loader.load(args.research_run)
        _verify_previous_h0(previous, previous_loader.resolve(args.research_run))

        strategy_config = previous.strategy_config
        backtest_config = previous.backtest_config
        hypothesis_config = settings.hypothesis_suite_config()
        multiregime_config = settings.multiregime_config()
        partitions = _partition_metadata(
            expansion_segments=expansion_segments,
            consumed=consumed,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
            holdout_registered_at=root_manifest.blind_holdout.registered_at,
        )

        # This immutable preregistration is written before any expanded H0-H4 replay.
        prepared = MultiRegimeManifestStore(args.preregistration_root).prepare(
            symbol=SYMBOL,
            interval=INTERVAL,
            requested_expansion_start=REQUESTED_EXPANSION_START,
            actual_expansion_start=actual_expansion_start,
            expansion_end=EXPANSION_END,
            partitions=partitions,
            multiregime_config=multiregime_config,
            hypothesis_config=hypothesis_config,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            cost_stress_enabled=args.cost_stress,
        )
        root_store.record_multiregime_run(
            run_id=prepared.run_id,
            manifest_path=str(prepared.path),
            configuration_sha256=prepared.configuration_sha256,
        )
        # Data becomes permanently consumed at the boundary immediately before replay.
        root_store.mark_expansion_consumed(
            symbol=SYMBOL,
            interval=INTERVAL,
            start=actual_expansion_start,
            end=EXPANSION_END,
            run_id=prepared.run_id,
        )

        logger.info("V3.2.1 MULTI-REGIME RESEARCH (OFFLINE)")
        logger.info("Symbol: %s | Interval: %s", SYMBOL, INTERVAL)
        logger.info(
            "Expansion: [%s, %s) | %d candles | %d gap-safe segment(s) | "
            "CONSUMED_RESEARCH_DATA",
            actual_expansion_start.isoformat(),
            EXPANSION_END.isoformat(),
            expansion_candle_count,
            len(expansion_segments),
        )
        logger.warning(
            "Blind holdout: [%s, %s) | LOCKED_BLIND_HOLDOUT | NOT LOADED",
            HOLDOUT_START.isoformat(),
            HOLDOUT_END.isoformat(),
        )
        result = MultiRegimeResearchRunner(
            multiregime_config=multiregime_config,
            hypothesis_config=hypothesis_config,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            minimum_trades_warning=previous.minimum_trades_warning,
        ).run(
            run_id=prepared.run_id,
            regions=(
                *(
                    ResearchRegion(partition, segment)
                    for partition, segment in zip(
                        (
                            item
                            for item in partitions
                            if item.kind is PartitionKind.RESEARCH_EXPANSION
                        ),
                        expansion_segments,
                        strict=True,
                    )
                ),
                ResearchRegion(
                    next(
                        item
                        for item in partitions
                        if item.kind is PartitionKind.CONSUMED_RESEARCH
                    ),
                    consumed,
                ),
            ),
            partitions=partitions,
            cost_stress=args.cost_stress,
            previous_h0_reproduction_verified=True,
        )

        final_manifest = root_store.load()
        _require_locked_holdout(final_manifest)
        paths = MultiRegimeReportWriter(args.reports_root).write(result, prepared)
        logger.info(
            "Windows: %d total | %d stability-eligible",
            len(result.windows),
            sum(window.eligible for window in result.windows),
        )
        logger.info("ID  TRADES  FRICTIONLESS EXP  NET EXP  PF  CLASSIFICATION")
        for item in result.stability:
            metric = item.combined_metrics
            logger.info(
                "%s  %d  %s  %s  %s  %s",
                item.hypothesis_id.value,
                metric.trades,
                metric.average_frictionless_pnl_per_trade,
                metric.expectancy,
                metric.profit_factor if metric.profit_factor is not None else "N/A",
                item.classification.value,
            )
            if item.hypothesis_id.value != "H0":
                logger.info(
                    "  consistency vs H0: frictionless=%d/%d (%s%%) | "
                    "net=%d/%d (%s%%) | PF=%d/%d (%s%%) | lower DD=%d/%d (%s%%)",
                    item.frictionless_better_windows,
                    item.eligible_windows,
                    item.frictionless_better_percent,
                    item.net_better_windows,
                    item.eligible_windows,
                    item.net_better_percent,
                    item.profit_factor_better_windows,
                    item.eligible_windows,
                    item.profit_factor_better_percent,
                    item.lower_drawdown_windows,
                    item.eligible_windows,
                    item.lower_drawdown_percent,
                )
        logger.info("Previous H0 reproduction: VERIFIED (%d trades)", EXPECTED_PREVIOUS_H0_TRADES)
        logger.info("Blind holdout status: LOCKED_BLIND_HOLDOUT (not revealed)")
        logger.info("Report directory: %s", paths.directory.resolve())
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("V3.2.1 multi-regime research failed: %s", exc)
        return 1


def _require_locked_holdout(manifest: object) -> None:
    status = getattr(manifest, "holdout_status")
    holdout = getattr(manifest, "blind_holdout")
    if (
        status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        or holdout is None
        or holdout.status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        or holdout.symbol != SYMBOL
        or holdout.interval != INTERVAL
        or holdout.start != HOLDOUT_START
        or holdout.end != HOLDOUT_END
        or holdout.reveal_timestamp is not None
        or holdout.consumed_timestamp is not None
    ):
        raise ValueError("The exact V3.2.1 blind holdout is not locked and untouched.")


def _load_expansion(root: str | Path) -> tuple[HistoricalDataset, ...]:
    dataset = HistoricalDatasetStore(root, allow_source_gaps=True).load(
        SYMBOL, INTERVAL
    )
    command = (
        "Set-Location 'C:\\Coding\\TradingBot'\n"
        ".\\.venv\\Scripts\\python.exe -m src.cli.download_history "
        "--symbol BTCUSDC --interval 15m --start 2023-01-01T00:00:00Z "
        "--end 2025-08-01T00:00:00Z "
        "--data-root data/historical/multiregime_expansion --allow-source-gaps"
    )
    if dataset is None:
        raise ValueError(f"Expansion cache is missing. Run exactly:\n{command}")
    selected = dataset.slice(REQUESTED_EXPANSION_START, EXPANSION_END)
    if not selected.candles:
        raise ValueError(f"Expansion cache has no requested candles. Run exactly:\n{command}")
    if selected.candles[-1].timestamp != parse_utc_datetime("2025-07-31T23:45:00Z"):
        raise ValueError(f"Expansion cache does not reach its exclusive end. Run exactly:\n{command}")
    return _split_continuous(selected)


def _split_continuous(dataset: HistoricalDataset) -> tuple[HistoricalDataset, ...]:
    """Preserve real source gaps as replay boundaries; never bridge them."""

    segments: list[tuple] = []
    current = []
    for candle in dataset.candles:
        if current and candle.timestamp != next_open_time(
            current[-1].timestamp, dataset.interval
        ):
            segments.append(tuple(current))
            current = []
        current.append(candle)
    if current:
        segments.append(tuple(current))
    return tuple(
        HistoricalDataset(
            symbol=dataset.symbol,
            interval=dataset.interval,
            source=dataset.source,
            candles=candles,
        )
        for candles in segments
    )


def _load_exact_range(root: str | Path, start: datetime, end: datetime) -> HistoricalDataset:
    dataset = HistoricalDatasetStore(root).load(SYMBOL, INTERVAL)
    if dataset is None:
        raise ValueError("The existing validated V3 research cache is missing.")
    selected = dataset.slice(start, end)
    if (
        not selected.candles
        or selected.candles[0].timestamp != start
        or selected.candles[-1].timestamp != parse_utc_datetime("2026-08-17T23:45:00Z")
    ):
        raise ValueError("The consumed V3 research cache does not cover its exact range.")
    return selected


def _assert_holdout_absent(*datasets: HistoricalDataset) -> None:
    if any(
        HOLDOUT_START <= candle.timestamp < HOLDOUT_END
        for dataset in datasets
        for candle in dataset.candles
    ):
        raise ValueError("A supplied dataset contains a locked blind-holdout candle.")


def _verify_previous_h0(source: object, research_directory: Path) -> None:
    total = 0
    for period in source.periods:
        evaluation = evaluate_strategy_period(
            period.dataset,
            strategy=TrendMomentumBaselineStrategy(source.strategy_config),
            strategy_config=source.strategy_config,
            backtest_config=source.backtest_config,
        )
        if evaluation.backtest.trades != period.trades or evaluation.signals != period.signals:
            raise ValueError(
                f"Frozen H0 identity failed in previous {period.period.value} period."
            )
        baseline_directory = research_directory / period.period.value / "baseline"
        summary = json.loads(
            (baseline_directory / "summary.json").read_text(encoding="utf-8")
        )
        for name, expected in summary.items():
            actual = getattr(evaluation.backtest.metrics, name)
            typed_expected = Decimal(str(expected)) if isinstance(actual, Decimal) else expected
            if actual != typed_expected:
                raise ValueError(
                    f"Frozen H0 metric {name} failed in previous "
                    f"{period.period.value} period."
                )
        with (baseline_directory / "equity.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            equity_rows = tuple(csv.DictReader(stream))
        if len(equity_rows) != len(evaluation.backtest.equity_curve):
            raise ValueError(
                f"Frozen H0 equity length failed in previous {period.period.value} period."
            )
        for point, row in zip(
            evaluation.backtest.equity_curve, equity_rows, strict=True
        ):
            if (
                point.timestamp.isoformat() != row["timestamp"]
                or point.cash != Decimal(row["cash"])
                or point.position_value != Decimal(row["position_value"])
                or point.total_equity != Decimal(row["total_equity"])
            ):
                raise ValueError(
                    f"Frozen H0 equity failed in previous {period.period.value} period."
                )
        total += len(evaluation.backtest.trades)
    if total != EXPECTED_PREVIOUS_H0_TRADES:
        raise ValueError(
            f"Frozen H0 reproduction expected 94 trades and produced {total}."
        )


def _partition_metadata(
    *,
    expansion_segments: tuple[HistoricalDataset, ...],
    consumed: HistoricalDataset,
    expansion_data_root: str | Path,
    consumed_data_root: str | Path,
    holdout_registered_at: datetime,
) -> tuple[ResearchPartitionMetadata, ...]:
    expansion_created_at = _cache_created_at(expansion_data_root)
    expansion = tuple(
        ResearchPartitionMetadata(
            kind=PartitionKind.RESEARCH_EXPANSION,
            symbol=SYMBOL,
            interval=INTERVAL,
            start=segment.candles[0].timestamp,
            end=next_open_time(segment.candles[-1].timestamp, segment.interval),
            status=PartitionStatus.CONSUMED_RESEARCH_DATA,
            candle_count=len(segment.candles),
            created_at=expansion_created_at,
            source=(
                "BINANCE_PUBLIC_SPOT "
                f"(GAP_SAFE_SEGMENT_{index}_OF_{len(expansion_segments)})"
            ),
            dataset_sha256=dataset_checksum(segment),
        )
        for index, segment in enumerate(expansion_segments, start=1)
    )
    return (
        *expansion,
        ResearchPartitionMetadata(
            kind=PartitionKind.LOCKED_BLIND_HOLDOUT,
            symbol=SYMBOL,
            interval=INTERVAL,
            start=HOLDOUT_START,
            end=HOLDOUT_END,
            status=PartitionStatus.LOCKED_BLIND_HOLDOUT,
            candle_count=None,
            created_at=holdout_registered_at,
            source="NOT_DOWNLOADED",
            dataset_sha256=None,
        ),
        ResearchPartitionMetadata(
            kind=PartitionKind.CONSUMED_RESEARCH,
            symbol=SYMBOL,
            interval=INTERVAL,
            start=CONSUMED_START,
            end=CONSUMED_END,
            status=PartitionStatus.CONSUMED_RESEARCH_DATA,
            candle_count=len(consumed.candles),
            created_at=_cache_created_at(consumed_data_root),
            source="BINANCE_PUBLIC_SPOT",
            dataset_sha256=dataset_checksum(consumed),
        ),
    )


def _cache_created_at(root: str | Path) -> datetime:
    path = HistoricalDatasetStore(root).metadata_path(SYMBOL, INTERVAL)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_utc_datetime(str(raw["download_update_timestamp"]))


if __name__ == "__main__":
    raise SystemExit(main())
