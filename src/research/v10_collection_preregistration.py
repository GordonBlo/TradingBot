"""Immutable acquisition-only V10 protocol; no evaluation or collection entry point."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = WORKSPACE / "research/v10_public_microstructure_collection"
PROTOCOL_VERSION = "V10_PUBLIC_MICROSTRUCTURE_COLLECTION_V1"
SOURCE_FILES = (
    "src/microstructure/v10.py",
    "src/orderbook/book.py",
    "src/orderbook/models.py",
    "src/orderbook/recorder.py",
    "src/orderflow/models.py",
    "src/exchange/public_market_client.py",
    "src/cli/record_v10_microstructure.py",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")
    return parsed.astimezone(UTC)


def timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def prospective_cutoff(created_at: datetime) -> datetime:
    created = utc(timestamp(created_at))
    return created.replace(minute=created.minute // 15 * 15, second=0, microsecond=0) + timedelta(minutes=15)


def strict_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must be an object")
    return value


def source_hashes(commit: str, workspace: Path = WORKSPACE) -> dict[str, str]:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("missing or invalid source commit identity")
    result = {}
    for name in SOURCE_FILES:
        try:
            content = subprocess.run(
                ["git", "show", f"{commit}:{name}"], cwd=workspace,
                capture_output=True, check=True, timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("collector source commit unavailable locally") from exc
        result[name] = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    return result


def build_manifest(created_at: datetime, source_commit: str, sources: dict[str, str]) -> dict:
    definition = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": timestamp(created_at),
        "prospective_cutoff_utc": timestamp(prospective_cutoff(created_at)),
        "purpose": "DATA_ACQUISITION_ONLY",
        "market": {"symbol": "BTCUSDC", "venue": "BINANCE_PUBLIC_SPOT"},
        "collector_version": "V10_PUBLIC_L2_AGGTRADES_1",
        "creation_source_commit_sha": source_commit,
        "collector_source_sha256_lf_normalized": sources,
        "streams": {
            "depth": {
                "identity": "BTCUSDC_DIFF_DEPTH_100MS",
                "websocket_url": "wss://data-stream.binance.vision/ws/btcusdc@depth@100ms",
                "rest_snapshot": "PUBLIC_API_V3_DEPTH",
                "raw_file": "depth.jsonl",
            },
            "aggtrades": {
                "identity": "BTCUSDC_AGGTRADE",
                "websocket_url": "wss://data-stream.binance.vision/ws/btcusdc@aggTrade",
                "raw_file": "aggtrades.jsonl",
            },
        },
        "session": {
            "identity": "shared immutable ID equal to UTC start formatted YYYYmmddTHHMMSSffffffZ",
            "raw_persistence": "separate append-only logs, contiguous zero-based source record indices",
            "start_rule": "session_start > prospective_cutoff; equality and straddling REJECT",
            "closure": "CLOSED_NORMAL and CLOSED_INTEGRITY_PASSED",
            "interrupted": "REJECT; requires a future separately frozen recovery protocol",
            "snapshot_limits": [100, 500, 1000, 5000],
            "max_levels_per_side": "positive integer <= 5000, declared immutably per session",
            "duration_seconds": "positive integer <= 86400, declared immutably per session",
            "required_artifacts": ["session.manifest.json", "closure.summary.json", "depth.jsonl", "aggtrades.jsonl"],
            "artifact_verification": "exact identity, raw sizes/counts/SHA256, manifest SHA256, closure SHA256 binding; rehash after scan",
            "source_identity": "resolvable Git source commit with the frozen collector source hashes; local replay source must match",
            "duplicates": "byte-identical artifact sets count once; conflicting copies invalidate every copy of that session ID",
            "overlapping_sessions": "reject every overlapping eligible session; touching endpoints allowed",
        },
        "depth_integrity": {
            "synchronization": "existing V9 snapshot/diff U/u bridging and reconstruction",
            "raw_validation": "strict integer update IDs; exact finite decimal text; positive prices, nonnegative depth quantities; duplicate depth prices REJECT",
            "sequence_gaps": 0,
            "resolved_gaps": "also REJECT in V1; no recovery exception",
            "invalid_events": 0,
            "crossed_invalid_book_states": 0,
            "required": "at least one snapshot and reconstructed update; final book valid",
            "resync_boundaries": "preserved RESYNC_BOUNDARY records; count must match live/replay; replacement snapshot requires a preceding boundary",
        },
        "aggtrade_integrity": {
            "numbers": "strict integer IDs/timestamps; exact decimal text, finite positive price/quantity",
            "buyer_is_maker": "exact boolean m, preserved without sign reversal",
            "duplicates": "consecutive identical full payloads retained raw and skipped in derived contexts",
            "regressions_conflicts_invalid": "REJECT",
            "aggregate_and_underlying_id_gaps": "REJECT; counters preserved; no inferred missing trades",
            "required": "at least one accepted unique aggTrade",
        },
        "reconnect_policy": "REJECT any reconnect in either stream: collector V1 has counters but incomplete raw reconnect boundary provenance",
        "causality": {
            "merge_key": ["local_receive_utc", "depth_before_aggTrade", "source_record_index"],
            "context": "latest valid reconstructed L2 with availability <= trade receive timestamp",
            "snapshot_bridge_availability": "buffered diff effects become available no earlier than snapshot receipt",
            "missing_context": "remains missing; counted and allowed; never repaired by future data",
            "future_interpolation_or_state_repair": "FORBIDDEN",
            "timestamps": "UTC-aware; nonregressing receive time per source; raw receipt within closed session; exchange event <= receive",
            "clock_skew": "future exchange timestamps fail integrity; no clock correction inferred",
        },
        "replay": "two independent raw replays must exactly match sealed replay metadata and live counters; hash every causal context during readiness",
        "readiness": {
            "minimum_eligible_closed_sessions": 8,
            "minimum_total_eligible_hours": 24,
            "minimum_utc_dates": 3,
            "hours": "sum min(requested duration, closed end minus start) for unique non-overlapping eligible sessions",
            "dates": "distinct UTC session START dates",
            "meaning": "data sufficiency only, not a trading hypothesis or profitability evidence",
        },
        "evidence_policy": {
            "later_discovery": "permitted only under a separately authorized research task",
            "independent_confirmatory_strategy_evidence": "FORBIDDEN unless the strategy hypothesis was frozen before session generation and its eligibility rules pass",
            "consumed_v9": "never fresh evidence",
            "blind_holdout": "LOCKED; never accessed",
        },
        "modes": ["PLAN", "READINESS"],
        "predictive_evaluation": False,
        "trading_strategy": False,
        "authentication": False,
        "orders": False,
    }
    digest = hashlib.sha256(canonical(definition)).hexdigest()
    return {"preregistration_id": digest[:16], "definition_sha256": digest, "definition": definition}


def verify_manifest(manifest: dict) -> None:
    definition = manifest["definition"]
    sources = definition["collector_source_sha256_lf_normalized"]
    if set(sources) != set(SOURCE_FILES) or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in sources.values()
    ):
        raise ValueError("invalid frozen source hashes")
    commit = definition["creation_source_commit_sha"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid preregistration source commit")
    expected = build_manifest(utc(definition["created_at_utc"]), commit, sources)
    if canonical(manifest) != canonical(expected):
        raise ValueError("V10 acquisition preregistration mismatch")


def load_manifest(root: Path = MANIFEST_ROOT) -> tuple[Path, dict]:
    paths = sorted(root.glob("*/manifest.json"))
    if len(paths) != 1:
        raise ValueError("require exactly one existing V10 acquisition preregistration")
    manifest = strict_json(paths[0].read_text(encoding="utf-8"))
    verify_manifest(manifest)
    if paths[0].parent.name != manifest["preregistration_id"]:
        raise ValueError("preregistration artifact identity mismatch")
    return paths[0], manifest


def verify_replay_source(manifest: dict, workspace: Path = WORKSPACE) -> None:
    for name, expected in manifest["definition"]["collector_source_sha256_lf_normalized"].items():
        actual = hashlib.sha256((workspace / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen collector/replay source changed: {name}")
