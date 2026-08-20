"""Register immutable V3.3 H5 research before any new parameter replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.diagnostics.loader import ResearchRunLoader
from src.hypotheses.h5_parameter_manifest import H5ParameterManifestStore
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore


EXPECTED_H5_TRADES = 197


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required source artifact not found: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_h5_source(report_directory: Path) -> None:
    path = report_directory / "candidate_stability.csv"

    if not path.is_file():
        raise ValueError(f"V3.2.2 stability report not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))

    row = next(
        (item for item in rows if item["hypothesis"] == "H5"),
        None,
    )

    if row is None:
        raise ValueError("H5 is missing from V3.2.2 stability report.")

    if row["classification"] != "NEXT_STAGE_ELIGIBLE":
        raise ValueError(
            "H5 is not NEXT_STAGE_ELIGIBLE in the frozen V3.2.2 report."
        )

    if int(row["combined_trades"]) != EXPECTED_H5_TRADES:
        raise ValueError(
            "Frozen H5 trade count changed: "
            f"expected {EXPECTED_H5_TRADES}, "
            f"got {row['combined_trades']}."
        )


def _verify_r_audit(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"H5 R-audit not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("hypothesis") != "H5":
        raise ValueError("R-audit is not registered to H5.")

    if payload.get("dataset_status") != "CONSUMED_RESEARCH_DATA":
        raise ValueError("H5 R-audit must use consumed research data.")

    audit = payload.get("audit", {})
    r_normalized = payload.get("r_normalized_audit", {})

    if int(audit.get("trade_count", -1)) != EXPECTED_H5_TRADES:
        raise ValueError("Risk/capital audit does not reproduce 197 H5 trades.")

    if int(r_normalized.get("trade_count", -1)) != EXPECTED_H5_TRADES:
        raise ValueError("R-normalized audit does not reproduce 197 H5 trades.")

    holdout = payload.get("holdout", {})

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError("R-audit does not preserve locked blind holdout.")

    if any(
        bool(holdout.get(name))
        for name in ("revealed", "consumed", "evaluated")
    ):
        raise ValueError("R-audit indicates blind-holdout contamination.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register immutable V3.3 H5 parameter research."
    )

    parser.add_argument(
        "--root-manifest",
        default="research/hypothesis_manifest.json",
    )

    parser.add_argument(
        "--research-run",
        default="adeef00722e9704d",
    )

    parser.add_argument(
        "--research-root",
        default="reports/research",
    )

    parser.add_argument(
        "--data-root",
        default="data/historical",
    )

    parser.add_argument(
        "--mechanism-run",
        default="bc2496aed05555b5",
    )

    parser.add_argument(
        "--mechanism-manifest-root",
        default="research/mechanisms",
    )

    parser.add_argument(
        "--mechanism-report-root",
        default="reports/mechanisms",
    )

    parser.add_argument(
        "--multiregime-run",
        default="3db30d3c8eacef4e",
    )

    parser.add_argument(
        "--multiregime-manifest-root",
        default="research/multiregime",
    )

    parser.add_argument(
        "--r-audit",
        default="reports/audits/h5_risk_capital_v322/summary.json",
    )

    parser.add_argument(
        "--output-root",
        default="research/v3_3_h5",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        root_manifest = ResearchManifestStore(
            args.root_manifest
        ).load()

        if root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT:
            raise ValueError("Blind holdout is not locked.")

        holdout = root_manifest.blind_holdout

        if holdout is None:
            raise ValueError("Blind holdout registration is missing.")

        if holdout.reveal_timestamp is not None:
            raise ValueError("Blind holdout has already been revealed.")

        if holdout.consumed_timestamp is not None:
            raise ValueError("Blind holdout has already been consumed.")

        mechanism_manifest = (
            Path(args.mechanism_manifest_root)
            / args.mechanism_run
            / "manifest.json"
        )

        mechanism_report = (
            Path(args.mechanism_report_root)
            / args.mechanism_run
        )

        multiregime_manifest = (
            Path(args.multiregime_manifest_root)
            / args.multiregime_run
            / "manifest.json"
        )

        r_audit = Path(args.r_audit)

        _verify_h5_source(mechanism_report)
        _verify_r_audit(r_audit)

        mechanism_payload = json.loads(
            mechanism_manifest.read_text(encoding="utf-8")
        )

        if mechanism_payload.get("run_id") != args.mechanism_run:
            raise ValueError("Mechanism manifest run ID mismatch.")

        multiregime_payload = json.loads(
            multiregime_manifest.read_text(encoding="utf-8")
        )

        if multiregime_payload.get("run_id") != args.multiregime_run:
            raise ValueError("Multi-regime manifest run ID mismatch.")

        source = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.data_root,
        ).load(args.research_run)

        if holdout.symbol != source.symbol:
            raise ValueError("Blind holdout symbol mismatch.")

        if holdout.interval != source.interval:
            raise ValueError("Blind holdout interval mismatch.")

        registration = H5ParameterManifestStore(
            args.output_root
        ).prepare(
            symbol=source.symbol,
            interval=source.interval,
            source_mechanism_run_id=args.mechanism_run,
            source_mechanism_manifest_sha256=_sha256(
                mechanism_manifest
            ),
            source_multiregime_run_id=args.multiregime_run,
            source_r_audit_sha256=_sha256(r_audit),
            strategy_config=source.strategy_config,
            backtest_config=source.backtest_config,
        )

        print("V3.3 H5 PREREGISTRATION COMPLETE")
        print(f"Run ID: {registration.run_id}")
        print(f"SHA-256: {registration.configuration_sha256}")
        print(f"Manifest: {registration.path.resolve()}")
        print("")
        print("Candidates: H5_Q25, H5_Q35, H5_Q45, H5_Q50")
        print("Frozen lookback: 100")
        print("Frozen upper percentile: Q75")
        print("")
        print("Blind holdout: LOCKED")
        print("Revealed: NO")
        print("Consumed: NO")
        print("Evaluated: NO")
        print("")
        print("NO PARAMETER REPLAY HAS BEEN EXECUTED.")

        return 0

    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V3.3 preregistration failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())