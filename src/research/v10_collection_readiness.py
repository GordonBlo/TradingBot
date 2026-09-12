"""Read-only eligibility and data sufficiency scan for the frozen V10 protocol."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from src.microstructure.v10 import (
    AGGTRADE_SCHEMA, AGGTRADE_STREAM, DEPTH_CONTROL_STREAM, DEPTH_SCHEMA,
    DEPTH_SNAPSHOT_STREAM, DEPTH_STREAM, MANIFEST_SCHEMA, SUMMARY_SCHEMA,
    _canonical, classify_session, iter_causal_trade_contexts,
    replay_synchronized_session, sha256_file,
)
from src.research.v10_collection_preregistration import (
    WORKSPACE, canonical, source_hashes, strict_json, timestamp, utc,
    verify_manifest, verify_replay_source,
)


DATA_ROOT = WORKSPACE / "data/microstructure/v10"
ARTIFACTS = ("session.manifest.json", "closure.summary.json", "depth.jsonl", "aggtrades.jsonl")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def microseconds(delta) -> int:
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def artifact_hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in ARTIFACTS}


def _raw_check(path: Path, *, session_id: str, schema: str, started, ended) -> tuple[int, int]:
    previous = None
    count = boundaries = 0
    snapshot_since_boundary = False
    streams = (
        {"DIFF_DEPTH": DEPTH_STREAM, "REST_SNAPSHOT": DEPTH_SNAPSHOT_STREAM,
         "RESYNC_BOUNDARY": DEPTH_CONTROL_STREAM}
        if schema == DEPTH_SCHEMA else {"AGG_TRADE": AGGTRADE_STREAM}
    )
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = strict_json(line)
            require(row.get("session_id") == session_id and row.get("schema_version") == schema,
                    "raw session/schema identity mismatch")
            require(type(row.get("record_index")) is int and row["record_index"] == index,
                    "raw record indices not contiguous")
            kind = row.get("record_type")
            require(kind in streams and row.get("stream_identity") == streams[kind],
                    "raw stream identity mismatch")
            received = utc(row["received_at_utc"])
            require(started <= received <= ended, "raw receipt outside closed session")
            require(previous is None or previous <= received, "raw receipt timestamp regression")
            previous = received
            payload = row["payload"]
            require(isinstance(payload, dict), "raw payload must be an object")
            require(row.get("exchange_event_timestamp_ms") == payload.get("E"),
                    "exchange timestamp metadata mismatch")
            if kind in {"DIFF_DEPTH", "AGG_TRADE"}:
                event_ms = payload.get("E")
                require(type(event_ms) is int and event_ms >= 0, "invalid exchange timestamp")
                epoch_us = microseconds(received - datetime(1970, 1, 1, tzinfo=UTC))
                require(event_ms * 1000 <= epoch_us, "future exchange timestamp")
            if kind in {"DIFF_DEPTH", "REST_SNAPSHOT"}:
                if kind == "REST_SNAPSHOT":
                    require(not snapshot_since_boundary, "snapshot replacement lacks resync boundary")
                    snapshot_since_boundary = True
                fields = ("U", "u") if kind == "DIFF_DEPTH" else ("lastUpdateId",)
                require(all(type(payload.get(key)) is int and payload[key] >= 0 for key in fields),
                        "invalid depth update ID")
                sides = ("b", "a") if kind == "DIFF_DEPTH" else ("bids", "asks")
                for side in sides:
                    require(isinstance(payload.get(side), list), "invalid depth levels")
                    seen = set()
                    for level in payload[side]:
                        require(isinstance(level, list) and len(level) == 2
                                and all(isinstance(value, str) for value in level),
                                "depth levels require exact price/quantity text")
                        price, quantity = map(Decimal, level)
                        require(price.is_finite() and quantity.is_finite()
                                and price > 0 and quantity >= 0, "invalid depth price/quantity")
                        require(price not in seen, "duplicate depth price")
                        seen.add(price)
            if kind == "RESYNC_BOUNDARY":
                require(payload == {"reason": "BOOK_STATE_INVALIDATED"}, "invalid resync boundary")
                boundaries += 1
                snapshot_since_boundary = False
            count += 1
    return count, boundaries


def _validate_session(path: Path, manifest: dict, source_cache: dict, *, now: datetime) -> dict:
    definition = manifest["definition"]
    directory = path.parent
    session = strict_json(path.read_text(encoding="utf-8"))
    started = utc(session["started_at_utc"])
    cutoff = utc(definition["prospective_cutoff_utc"])
    result = {
        "session_id": session["session_id"], "manifest_path": str(path),
        "started_at_utc": timestamp(started), "eligible_microseconds": 0,
        "classification": "INELIGIBLE", "reason": "",
    }
    summary_path = directory / "closure.summary.json"
    if not summary_path.is_file():
        return {**result, "classification": "INTERRUPTED", "reason": "normal closure summary missing"}
    summary = strict_json(summary_path.read_text(encoding="utf-8"))
    ended = utc(summary["ended_at_utc"])
    result["ended_at_utc"] = timestamp(ended)
    if started <= cutoff:
        classification = "CUTOFF_STRADDLING" if started < cutoff < ended else "AT_OR_BEFORE_CUTOFF"
        return {**result, "classification": classification, "reason": "session start must be strictly after cutoff"}
    require(started < ended <= now, "invalid or future session interval")
    require(session["session_id"] == started.strftime("%Y%m%dT%H%M%S%fZ")
            == directory.name, "session ID/start/directory mismatch")
    require(session.get("schema_version") == MANIFEST_SCHEMA
            and summary.get("schema_version") == SUMMARY_SCHEMA, "session schema mismatch")
    for item in (session, summary):
        require(item.get("collector_version") == definition["collector_version"], "collector identity mismatch")
        require(item.get("symbol") == "BTCUSDC", "symbol mismatch")
        require(item.get("uses_authentication") is False and item.get("orders_enabled") is False,
                "public-only restrictions missing")
    require(session.get("source") == "BINANCE_PUBLIC_SPOT", "nonpublic source")
    require(summary.get("predictive_outcomes_evaluated") is False, "predictive evaluation flag invalid")
    require(summary.get("started_at_utc") == session.get("started_at_utc"), "session start mismatch")
    require(summary.get("session_id") == session["session_id"], "closure session identity mismatch")
    require(classify_session(path) == "CLOSED_INTEGRITY_PASSED", "collection integrity or closure failed")
    before = artifact_hashes(directory)
    commit = session.get("source_commit_sha")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            "missing or invalid source commit identity")
    if commit not in source_cache:
        source_cache[commit] = source_hashes(commit)
    require(source_cache[commit] == definition["collector_source_sha256_lf_normalized"],
            "source commit collector semantics differ from frozen protocol")
    streams = session["streams"]
    require(set(streams) == {"depth", "aggtrades"}, "unexpected stream definitions")
    depth_stream = dict(streams["depth"])
    limit = depth_stream.pop("snapshot_limit")
    levels = depth_stream.pop("max_levels_per_side")
    require(type(limit) is int and limit in definition["session"]["snapshot_limits"], "invalid snapshot limit")
    require(type(levels) is int and 0 < levels <= 5000, "invalid maximum depth levels")
    require(depth_stream == definition["streams"]["depth"]
            and streams["aggtrades"] == definition["streams"]["aggtrades"], "stream definitions mismatch")
    requested = session["requested_duration_seconds"]
    require(type(requested) is int and 0 < requested <= 86400, "invalid requested duration")
    duration_us = microseconds(ended - started)
    require(Decimal(str(summary["duration_seconds_actual"])) == Decimal(duration_us) / 1_000_000,
            "closure duration accounting mismatch")
    require(summary.get("deterministic_replay_hash_check_passed") is True, "sealed deterministic replay failed")
    accounting = summary["integrity_accounting"]
    require(set(accounting) == {"depth_live_matches_replay", "aggtrades_live_matches_replay",
                               "raw_record_counts_match_replay", "passed"}
            and all(value is True for value in accounting.values()), "sealed integrity accounting failed")
    require(set(summary["raw_artifacts"]) == {"depth", "aggtrades"}, "unexpected raw artifact identities")
    raw_counts = {}
    resync_boundaries = 0
    for kind, filename, schema, identity in (
        ("depth", "depth.jsonl", DEPTH_SCHEMA, DEPTH_STREAM),
        ("aggtrades", "aggtrades.jsonl", AGGTRADE_SCHEMA, AGGTRADE_STREAM),
    ):
        count, boundaries = _raw_check(directory / filename, session_id=session["session_id"],
                                       schema=schema, started=started, ended=ended)
        raw_counts[kind] = count
        resync_boundaries += boundaries
        artifact = summary["raw_artifacts"][kind]
        require(type(artifact.get("record_count")) is int and artifact["record_count"] == count,
                "raw record count mismatch")
        require(artifact.get("stream_identity") == identity, "artifact stream identity mismatch")
    args = (directory / "depth.jsonl", directory / "aggtrades.jsonl")
    kwargs = {"session_id": session["session_id"], "max_levels_per_side": levels}
    replay = replay_synchronized_session(*args, **kwargs).to_dict()
    repeat = replay_synchronized_session(*args, **kwargs).to_dict()
    require(canonical(replay) == canonical(repeat) == canonical(summary["offline_replay"]),
            "deterministic replay/metadata hash mismatch")
    for kind, key in (("depth", "depth_counters"), ("aggtrades", "aggtrade_counters")):
        live = summary["live_integrity"][kind]
        require(type(live.get("reconnect_count")) is int and live["reconnect_count"] == 0,
                "reconnect boundary provenance unavailable in collector V1")
        require(canonical(live) == canonical(replay[key]), "live/replay counter mismatch")
    depth, trades = replay["depth_counters"], replay["aggtrade_counters"]
    for key in ("sequence_gaps", "invalid_events", "crossed_invalid_book_states"):
        require(type(depth[key]) is int and depth[key] == 0, f"depth integrity failure: {key}")
    require(depth["snapshots"] >= 1 and depth["reconstructed_updates"] >= 1
            and replay["final_depth_valid"] is True, "invalid snapshot/diff synchronization")
    require(resync_boundaries == depth["resync_count"], "resync boundary accounting mismatch")
    require(raw_counts["depth"] == depth["snapshots"] + depth["diff_events"] + depth["resync_count"],
            "depth raw accounting mismatch")
    for key in ("conflicting_duplicates", "id_regressions", "timestamp_regressions",
                "aggregate_id_gap_events", "missing_aggregate_trade_ids",
                "underlying_trade_id_gap_events", "missing_underlying_trade_ids", "invalid_events"):
        require(type(trades[key]) is int and trades[key] == 0, f"aggTrade integrity failure: {key}")
    require(trades["accepted_events"] >= 1
            and raw_counts["aggtrades"] == trades["raw_events"]
            == trades["accepted_events"] + trades["duplicate_events"], "aggTrade accounting mismatch")
    contexts_hash = hashlib.sha256()
    count = missing = 0
    for context in iter_causal_trade_contexts(*args, **kwargs):
        count += 1
        if context.depth is None:
            missing += 1
        else:
            require(context.depth.available_at <= context.observed_trade.received_at, "future L2 context")
        contexts_hash.update((_canonical(asdict(context)) + "\n").encode())
    require(count == replay["causal_trade_count"] == trades["accepted_events"]
            and missing == replay["trades_without_depth"], "causal context accounting mismatch")
    require(before == artifact_hashes(directory), "session artifacts changed during readiness")
    return {
        **result, "classification": "ELIGIBLE", "reason": "all frozen acquisition checks passed",
        "eligible_microseconds": min(requested * 1_000_000, duration_us),
        "artifact_sha256": before, "source_commit_sha": commit,
        "synchronized_timeline_sha256": replay["synchronized_timeline_sha256"],
        "causal_contexts_sha256": contexts_hash.hexdigest(),
        "trades_without_depth": missing,
    }


def scan_readiness(manifest: dict, *, data_root: Path = DATA_ROOT, now: datetime | None = None) -> dict:
    verify_manifest(manifest)
    verify_replay_source(manifest)
    root = data_root.resolve()
    v9_root = (WORKSPACE / "data/orderbook/v9").resolve()
    require(not root.is_relative_to(v9_root) and not v9_root.is_relative_to(root), "V9 data root forbidden")
    observed_now = utc(timestamp(now or datetime.now(UTC)))
    sessions = []
    groups = {}
    source_cache = {}
    for path in sorted(root.rglob("session.manifest.json")) if root.exists() else []:
        identity = str(path)
        fingerprint = None
        try:
            require(path.resolve().is_relative_to(root), "manifest escapes data root")
            session = strict_json(path.read_text(encoding="utf-8"))
            require(isinstance(session.get("session_id"), str), "invalid session identity")
            identity = session["session_id"]
            for name in ARTIFACTS:
                require((path.parent / name).resolve().is_relative_to(root), "artifact escapes data root")
            if all((path.parent / name).is_file() for name in ARTIFACTS):
                fingerprint = artifact_hashes(path.parent)
            result = _validate_session(path, manifest, source_cache, now=observed_now)
        except (ValueError, OSError, KeyError, TypeError, ArithmeticError, AttributeError) as exc:
            result = {"session_id": identity, "manifest_path": str(path),
                      "classification": "INTEGRITY_FAILED", "eligible_microseconds": 0, "reason": str(exc)}
        sessions.append(result)
        groups.setdefault(identity, []).append((result, fingerprint))
    # Bind the complete scan, including earlier sessions while later ones replayed.
    for result in sessions:
        if result["classification"] == "ELIGIBLE":
            require(artifact_hashes(Path(result["manifest_path"]).parent) == result["artifact_sha256"],
                    "session artifacts changed during dataset readiness")
    verify_replay_source(manifest)
    for copies in groups.values():
        if len(copies) < 2:
            continue
        if copies[0][1] is None or any(item[1] != copies[0][1] for item in copies[1:]):
            for result, _ in copies:
                result.update(classification="CONFLICTING_DUPLICATE", eligible_microseconds=0,
                              reason="conflicting or incomplete copies of the same session ID")
        else:
            for result, _ in copies[1:]:
                result.update(classification="IDENTICAL_DUPLICATE", eligible_microseconds=0,
                              reason="identical artifact set counted once")
    eligible = sorted((item for item in sessions if item["classification"] == "ELIGIBLE"),
                      key=lambda item: (item["started_at_utc"], item["session_id"]))
    overlaps = set()
    furthest = None
    for item in eligible:
        if furthest is not None and utc(item["started_at_utc"]) < utc(furthest["ended_at_utc"]):
            overlaps.update((item["session_id"], furthest["session_id"]))
        if furthest is None or utc(item["ended_at_utc"]) > utc(furthest["ended_at_utc"]):
            furthest = item
    for item in eligible:
        if item["session_id"] in overlaps:
            item.update(classification="OVERLAPPING_SESSION", eligible_microseconds=0,
                        reason="overlapping session coverage cannot count as independent acquisition")
    eligible = [item for item in eligible if item["classification"] == "ELIGIBLE"]
    total_us = sum(item["eligible_microseconds"] for item in eligible)
    dates = sorted({utc(item["started_at_utc"]).date().isoformat() for item in eligible})
    gates = manifest["definition"]["readiness"]
    checks = {
        "eligible_sessions": len(eligible) >= gates["minimum_eligible_closed_sessions"],
        "eligible_hours": total_us >= gates["minimum_total_eligible_hours"] * 3_600_000_000,
        "utc_dates": len(dates) >= gates["minimum_utc_dates"],
    }
    return {
        "preregistration_id": manifest["preregistration_id"],
        "preregistration_definition_sha256": manifest["definition_sha256"],
        "prospective_cutoff_utc": manifest["definition"]["prospective_cutoff_utc"],
        "status": "READY" if all(checks.values()) else "NOT_READY",
        "checks": checks, "eligible_sessions": len(eligible),
        "eligible_session_ids": [item["session_id"] for item in eligible],
        "eligible_hours": str(Decimal(total_us) / Decimal(3_600_000_000)),
        "eligible_utc_dates": dates, "sessions": sessions,
        "data_sufficiency_only": True, "independent_confirmatory_strategy_evidence": False,
        "predictive_outcomes_evaluated": False, "blind_holdout": "LOCKED",
    }
