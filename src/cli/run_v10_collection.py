"""PLAN or READINESS for the frozen V10 public acquisition protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.microstructure.v10 import sha256_file
from src.research.v10_collection_preregistration import load_manifest
from src.research.v10_collection_readiness import DATA_ROOT, scan_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("PLAN", "READINESS"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=None,
                        help="session validation processes (default: up to 4 CPUs; 1: serial)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path, manifest = load_manifest()
        before = sha256_file(path)
        result = manifest if args.mode == "PLAN" else scan_readiness(
            manifest, data_root=args.data_root, workers=args.workers)
        if sha256_file(path) != before:
            raise ValueError("preregistration changed during scan")
        print(json.dumps({"manifest_sha256": before, **result}, sort_keys=True, indent=2))
        return 0 if args.mode == "PLAN" or result["status"] == "READY" else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"INTEGRITY_FAILURE: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
