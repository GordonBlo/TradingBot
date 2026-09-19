"""Read-only PLAN/VERIFY for a preregistered diagnostic that is not authorized to run."""

from __future__ import annotations

import argparse
import json

from src.research.v10_economic_discovery_preregistration import load_manifest, verify_dataset_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("PLAN", "VERIFY"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, manifest = load_manifest()
        if args.mode == "VERIFY":
            verify_dataset_binding(manifest)
        print(json.dumps({"mode": args.mode, "preregistration_id": manifest["preregistration_id"],
                          "definition_sha256": manifest["definition_sha256"],
                          "predictive_outcomes_evaluated": False, "execution_authorized": False,
                          "evidence_role": "DISCOVERY_ONLY", "manifest": manifest}, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        print(f"INTEGRITY_FAILURE: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
