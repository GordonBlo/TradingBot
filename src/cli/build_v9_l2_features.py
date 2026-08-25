"""Build causal V9 BTCUSDC L2 features from persisted raw depth data only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.orderbook.features import FeatureExtractor, deterministic_replay_check


DEFAULT_RAW_SESSION = Path(
    "data/orderbook/v9/BTCUSDC/2026/08/25/20260825T084713536106Z.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_SESSION)
    parser.add_argument("--output-root", type=Path, default=Path("data/orderbook/v9/features"))
    parser.add_argument("--max-levels", type=int, default=5_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_root / args.raw.stem
    replay_output = args.output_root / f".{args.raw.stem}.determinism_check"
    replay_created = False
    try:
        if replay_output.exists():
            raise FileExistsError(f"V9 deterministic replay output already exists: {replay_output}")
        primary = FeatureExtractor(max_levels_per_side=args.max_levels).extract(args.raw, output)
        replay_created = True
        replay = deterministic_replay_check(
            args.raw,
            extractor_factory=lambda: FeatureExtractor(max_levels_per_side=args.max_levels),
            temporary_output_root=replay_output,
        )
        deterministic = primary["file_sha256"] == replay["file_sha256"]
        primary["deterministic_replay_hash_check"] = {
            "passed": deterministic,
            "primary_file_sha256": primary["file_sha256"],
            "replay_file_sha256": replay["file_sha256"],
        }
        report_path = output / "validation_report.json"
        report_path.write_text(json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not deterministic:
            raise RuntimeError("V9 L2 deterministic replay hashes differ.")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"V9 L2 FEATURE BUILD FAILED: {exc}")
        return 1
    finally:
        if replay_created and replay_output.exists():
            shutil.rmtree(replay_output)
    print(json.dumps(primary["counters"], indent=2, sort_keys=True))
    print(f"REPORT: {report_path}")
    print("V9 L2 FEATURES COMPLETE - RAW REPLAY ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
