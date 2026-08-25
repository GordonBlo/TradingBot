from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.research.v9_l2_early_information_preregistration import write_or_verify_manifest
from src.research.v9_l2_readiness import scan_session_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9 L2 diagnostic PLAN/READINESS guard")
    parser.add_argument(
        "--mode",
        choices=("PLAN", "READINESS", "EVALUATE"),
        default="PLAN",
        help="EVALUATE is guarded and disabled in this preregistration step",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/orderbook/v9"))
    parser.add_argument(
        "--feature-root", type=Path, default=Path("data/orderbook/v9/features")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path, manifest = write_or_verify_manifest()
    readiness = scan_session_readiness(
        manifest,
        data_root=args.data_root,
        feature_root=args.feature_root,
    )
    print(f"Preregistration ID: {manifest['run_id']}")
    print(f"Prospective cutoff: {readiness.cutoff_utc}")
    print(f"Manifest: {path}")
    print(json.dumps(readiness.to_dict(), indent=2, sort_keys=True))
    if args.mode == "EVALUATE":
        if not readiness.ready:
            print("PREDICTIVE EVALUATION REFUSED — READINESS GATE NOT PASSED")
        else:
            print("PREDICTIVE EVALUATION DISABLED — USE A FUTURE AUTHORIZED RUNNER")
        return 2
    status = "READY" if readiness.ready else "NOT_READY"
    print(f"{args.mode} COMPLETE — {status} — NO PREDICTIVE OUTCOMES EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
