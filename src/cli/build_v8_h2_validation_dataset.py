"""Build independent, outcome-unseen historical context for frozen V8-H2."""

from __future__ import annotations

import argparse

from src.hypotheses.manifest import ResearchManifestStore
from src.research.v8_h2_validation_dataset import archive_plan, build_validation_dataset, validation_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument("--raw-root", default="data/v8_h2_validation/raw")
    parser.add_argument("--output-root", default="research/v8_h2_validation")
    args = parser.parse_args(argv)
    manifest = ResearchManifestStore(args.root_manifest).load()
    consumed, holdout, eligible, selected = validation_plan(manifest)
    planned = archive_plan(selected, raw_root=args.raw_root)
    print(f"Eligible intervals: {[item.as_dict() for item in eligible]}")
    print(f"Selected interval: {selected.as_dict()} | archives: {len(planned)} ZIP + {len(planned)} CHECKSUM")
    print(f"Excluded consumed ranges: {[item.as_dict() for item in consumed]} | holdout: {holdout.as_dict()}")
    if not args.execute:
        print("PLAN COMPLETE — NO VALIDATION DATA ACQUIRED")
        return 0
    result, path = build_validation_dataset(manifest=manifest, raw_root=args.raw_root, output_root=args.output_root)
    print(f"{result['classification']} | dataset_id={result['dataset_id']} | manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
