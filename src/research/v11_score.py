"""Consumed V9 preparation only; no validation, search, or prospective collection.

The production entry point reserves its output directory before its one fit. A
failed/interrupted preparation is not automatically retried. The sealed V9
diagnostic and its original train/held-out results are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from src.diagnostics.v8_derivatives_discovery import fit_prepared_fold, prepare_fold
from src.diagnostics.v9_l2_early_information import (
    _INPUT_COLUMNS,
    _validate_readiness_and_samples,
    load_readiness_bound_samples,
)
from src.research.v9_l2_early_information_preregistration_v2 import FROZEN_FEATURES
from src.research.v9_l2_readiness import (
    ReadinessReport,
    SessionArtifactBinding,
    SessionEligibility,
)

V9_ID = "853870051677af08"
REPORT = Path(f"reports/v9_l2_early_information_v2/{V9_ID}/confirmatory_evaluation.json")
REPORT_SHA = "298a57f5a5fc01f1fd896ebe338906d6c121baf499d04bca73c06cda9f5bc9c2"
MANIFEST = Path(f"research/v9_l2_early_information_v2/{V9_ID}/manifest.json")
TARGET = "log(mid_price[T+30s] / mid_price[T]); exact timestamps; same CLOSED session"
PREPROCESSING = {
    "imputation": "training median; all-missing training feature is an integrity failure",
    "centering": "training imputed arithmetic mean",
    "scaling": "training population standard deviation; zero scale transforms to zero",
    "intercept": "arithmetic mean of training targets; unpenalized",
    "solver": "existing prepare_fold / fit_prepared_fold; alpha=1",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("finite numeric score value required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite score value")
    return result


def validate_model(model: dict) -> None:
    if (model.get("feature_order") != list(FROZEN_FEATURES)
            or model.get("input_columns") != list(_INPUT_COLUMNS)
            or model.get("ridge_alpha") != 1.0
            or model.get("target_definition") != TARGET
            or model.get("preprocessing") != PREPROCESSING
            or model.get("accept_rule") != "prediction > 0"):
        raise ValueError("frozen V9 score semantics mismatch")
    _finite(model["intercept"])
    for name in ("coefficients", "medians", "means", "scales"):
        if len(model[name]) != len(FROZEN_FEATURES):
            raise ValueError("score width mismatch")
        for value in model[name]:
            if _finite(value) < 0 and name == "scales":
                raise ValueError("negative scale")


def score(model: dict, row: Sequence[float | None]) -> float:
    """Same operation order as the frozen implementation, including zero scales."""
    validate_model(model)
    if len(row) != len(FROZEN_FEATURES):
        raise ValueError("score row width mismatch")
    transformed = tuple(
        ((_finite(value) if value is not None else model["medians"][i]) - model["means"][i])
        / model["scales"][i] if model["scales"][i] > 0 else 0.0
        for i, value in enumerate(row)
    )
    # Even zero-scale columns must reject malformed observations.
    for value in row:
        if value is not None:
            _finite(value)
    result = model["intercept"] + sum(
        value * coefficient for value, coefficient in zip(transformed, model["coefficients"])
    )
    return _finite(result)


def fit_static_model(features: Sequence[Sequence[float | None]], targets: Sequence[float]) -> dict:
    """One fit, no holdout metrics. The isolated dummy row satisfies the old API.

    It never contributes to training/preprocessing; its target is never read.
    No real-data predictions or alternative fits are requested.
    """
    if not features or len(features) != len(targets):
        raise ValueError("empty or mismatched preparation rows")
    width = len(FROZEN_FEATURES)
    if any(len(row) != width for row in features):
        raise ValueError("training feature width mismatch")
    for column in range(width):
        observed = [row[column] for row in features if row[column] is not None]
        if not observed:
            raise ValueError("all-missing training feature")
        for value in observed:
            _finite(value)
    values = [_finite(value) for value in targets]
    n = len(features)
    prepared = prepare_fold(
        [*features, (None,) * width], training_indices=range(n),
        held_out_indices=(n,), alpha=1.0,
    )
    fit = fit_prepared_fold(prepared, values)
    model = {
        "coefficients": list(fit.coefficients), "intercept": fit.intercept,
        "medians": list(prepared.medians), "means": list(prepared.means),
        "scales": list(prepared.scales), "feature_order": list(FROZEN_FEATURES),
        "input_columns": list(_INPUT_COLUMNS), "ridge_alpha": 1.0,
        "target_definition": TARGET, "preprocessing": PREPROCESSING,
        "accept_rule": "prediction > 0",
    }
    validate_model(model)
    return model


def load_bundle(path: Path, *, expected_sha256: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("score bundle byte hash mismatch")
    bundle = json.loads(data)
    if bundle["status"] != "CONSUMED_EVIDENCE_MODEL_PREPARATION_ONLY":
        raise ValueError("invalid bundle research status")
    if hashlib.sha256(encoded(bundle["model"])).hexdigest() != bundle["model_sha256"]:
        raise ValueError("model payload hash mismatch")
    validate_model(bundle["model"])
    return bundle


def bound_preparation_inputs(workspace: Path) -> tuple[ReadinessReport, dict]:
    """Resolve ONLY the eight identities from the sealed report, never scan data."""
    if sha256(workspace / REPORT) != REPORT_SHA:
        raise ValueError("sealed V9 report mismatch")
    report = json.loads((workspace / REPORT).read_text())
    if sha256(workspace / MANIFEST) != report["preregistration"]["v2_manifest_sha256"]:
        raise ValueError("frozen V9 manifest mismatch")
    manifest = json.loads((workspace / MANIFEST).read_text())
    sessions = []
    for record in report["artifact_bindings"]:
        sid = record["session_id"]
        day = datetime.strptime(sid, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
        base = Path("data/orderbook/v9/BTCUSDC") / day.strftime("%Y/%m/%d")
        summary_path = base / f"{sid}.summary.json"
        if sha256(workspace / summary_path) != record["closure_summary_sha256"]:
            raise ValueError("bound closure mismatch")
        summary = json.loads((workspace / summary_path).read_text())
        if any(summary["integrity"][k] != 0 for k in
               ("sequence_gaps", "invalid_events", "crossed_invalid_book_states")):
            raise ValueError("bound session integrity failure")
        binding = SessionArtifactBinding(
            sid, (base / f"{sid}.jsonl").as_posix(), record["raw_sha256"],
            summary_path.as_posix(), record["closure_summary_sha256"],
            f"data/orderbook/v9/features/{sid}/features_1s.csv", record["features_1s_sha256"],
        )
        started, ended = summary["started_at_utc"], summary["ended_at_utc"]
        hours = (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds()/3600
        sessions.append(SessionEligibility(sid, started, ended, hours, "ELIGIBLE", True,
                                           "EXACT_SEALED_CONSUMED_V9_BINDING", binding))
    sessions.sort(key=lambda item: (item.started_at_utc, item.session_id))
    if len(sessions) != 8 or len({s.session_id for s in sessions}) != 8:
        raise ValueError("exactly eight original sessions required")
    readiness = ReadinessReport(
        manifest["definition"]["prospective_cutoff_utc"], 8, 0, 8,
        sum(s.duration_hours for s in sessions), len({s.started_at_utc[:10] for s in sessions}),
        0, 0, 0, True, (), tuple(sessions),
    )
    return readiness, report


def prepare_once(workspace: Path, output: Path) -> dict:
    """Authorized consumed-data fit; output must be a new dedicated directory."""
    if output.exists():
        raise FileExistsError("preparation already reserved; do not refit")
    readiness, report = bound_preparation_inputs(workspace)
    sources = [
        "src/research/v11_score.py", "src/diagnostics/v8_derivatives_discovery.py",
        "src/diagnostics/v9_l2_early_information.py", "src/research/v9_l2_readiness.py",
        "src/research/v9_l2_early_information_preregistration_v2.py",
    ]
    identity = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip(),
        "source_sha256": {name: sha256(workspace / name) for name in sources},
        "python": sys.version, "implementation": platform.python_implementation(),
        "platform": platform.platform(), "float_mant_dig": sys.float_info.mant_dig,
    }
    binding = {
        "evidence_status": "CONSUMED", "v9_id": V9_ID, "report_sha256": REPORT_SHA,
        "manifest_sha256": sha256(workspace / MANIFEST),
        "sessions": [asdict(s) for s in readiness.sessions], "identity": identity,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "preparation_binding.json").write_bytes(encoded(binding))
    (output / "fit_reservation.json").write_bytes(encoded({"maximum_real_fits": 1, "binding_sha256":
        sha256(output / "preparation_binding.json")}))
    samples = _validate_readiness_and_samples(
        load_readiness_bound_samples(readiness, workspace=workspace), readiness,
    )
    if len(samples) != report["primary_result"]["sample_count"]:
        raise ValueError("original eligible primary sample count changed")
    model = fit_static_model([s.features for s in samples], [s.target(30) for s in samples])
    bundle = {
        "schema_version": 1, "status": "CONSUMED_EVIDENCE_MODEL_PREPARATION_ONLY",
        "real_fit_count": 1, "training_rows": len(samples), "model": model,
        "model_sha256": hashlib.sha256(encoded(model)).hexdigest(), "binding": binding,
    }
    path = output / "score_bundle.json"
    with path.open("xb") as stream:
        stream.write(encoded(bundle))
    digest = sha256(path)
    (output / "score_bundle.sha256").write_text(digest + "\n", encoding="ascii")
    load_bundle(path, expected_sha256=digest)
    return {"bundle": str(path), "sha256": digest, "training_rows": len(samples), "fit_count": 1}
