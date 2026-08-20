"""Display research-data and blind-holdout lifecycle status without mutation."""

from __future__ import annotations

import argparse

from src.hypotheses.manifest import ResearchManifestStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show immutable research status.")
    parser.add_argument("--manifest", default="research/hypothesis_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = ResearchManifestStore(args.manifest).load()
        print(f"Project version: {manifest.project_version}")
        print("Consumed research ranges:")
        for item in manifest.consumed_dataset_ranges:
            print(
                f"  [{item.start.isoformat()}, {item.end.isoformat()}) "
                f"{item.symbol} {item.interval} {item.data_status.value} "
                f"({item.source_period})"
            )
        holdout = manifest.blind_holdout
        if holdout is None:
            print("Blind holdout: UNSEEN")
        else:
            print(
                f"Blind holdout: [{holdout.start.isoformat()}, "
                f"{holdout.end.isoformat()}) {holdout.status.value}"
            )
            print(f"Revealed: {holdout.reveal_timestamp is not None}")
            print(f"Consumed: {holdout.consumed_timestamp is not None}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Research status failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
