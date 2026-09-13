from __future__ import annotations

import json
from datetime import timedelta

import pytest

from src.cli import validate_v10_overnight as overnight
from src.microstructure.v10 import sha256_file
from src.research.v10_collection_preregistration import WORKSPACE, load_manifest, utc
from tests.test_v10_collection_preregistration import START, SCAN_TIME, edit_json, make_session


@pytest.fixture
def session(tmp_path):
    proof = overnight.preflight(tmp_path, now=START - timedelta(seconds=1))
    paths = make_session(tmp_path / overnight.ROOT,
                         {"definition": {"creation_source_commit_sha": proof["source_commit_sha"]}})
    return tmp_path, paths


def successful_outcomes():
    return {key: "success" for key in overnight.REQUIRED_STEPS}


def test_preflight_strict_cutoff(tmp_path):
    _, manifest = load_manifest()
    cutoff = utc(manifest["definition"]["prospective_cutoff_utc"])
    for at in (cutoff - timedelta(microseconds=1), cutoff):
        with pytest.raises(ValueError, match="strictly after"):
            overnight.preflight(tmp_path, now=at)
        assert not (tmp_path / overnight.ROOT).exists()
    proof = overnight.preflight(tmp_path, now=cutoff + timedelta(microseconds=1))
    assert proof["preregistration_id"] == "8326791b411c27b5"


def test_preflight_rejects_wrong_protocol_before_creating_root(tmp_path, monkeypatch):
    path, manifest = load_manifest()
    changed = {**manifest, "preregistration_id": "wrong"}
    monkeypatch.setattr(overnight, "load_manifest", lambda: (path, changed))
    with pytest.raises(ValueError, match="wrong V10"):
        overnight.preflight(tmp_path, now=START)
    assert not (tmp_path / overnight.ROOT).exists()


def test_preflight_rejects_manifest_byte_changes(tmp_path, monkeypatch):
    _, manifest = load_manifest()
    changed_path = tmp_path / "manifest.json"
    changed_path.write_text(json.dumps(manifest) + "\n\n")
    monkeypatch.setattr(overnight, "load_manifest", lambda: (changed_path, manifest))
    with pytest.raises(ValueError, match="manifest bytes changed"):
        overnight.preflight(tmp_path, now=START)


def test_preflight_rejects_existing_root(session):
    workspace, paths = session
    before = sha256_file(paths.depth_raw)
    with pytest.raises(ValueError, match="absent canonical"):
        overnight.preflight(workspace, now=START)
    assert sha256_file(paths.depth_raw) == before


def test_single_session_passes_scanner_without_campaign_readiness(session):
    workspace, paths = session
    before = {name: sha256_file(paths.directory / name) for name in overnight.ARTIFACTS}
    metadata = overnight.validate(1, "123", "2", workspace=workspace, now=SCAN_TIME)
    assert metadata["result"]["status"] == "PASS"
    assert metadata["result"]["eligible_hours"] == "3"
    assert metadata["readiness"]["eligible_sessions"] == 1
    assert metadata["readiness"]["status"] == "NOT_READY"
    assert metadata["result"]["artifact_hashes"] == before
    assert metadata["preflight"]["preregistration_id"] == overnight.PREREGISTRATION_ID
    assert metadata["result"]["artifact_name"] == f"v10-microstructure-overnight-123-attempt-2-{paths.session_id}"
    result = overnight.publish(1, successful_outcomes(), workspace=workspace)
    assert result["status"] == "PASS"
    assert set(result["artifact_hashes"]) == {*overnight.ARTIFACTS, overnight.VALIDATION}
    assert {name: sha256_file(paths.directory / name) for name in overnight.ARTIFACTS} == before


@pytest.mark.parametrize("source,key", [
    ("depth", "sequence_gaps"), ("depth", "invalid_events"),
    ("depth", "crossed_invalid_book_states"), ("depth", "reconnect_count"),
    ("aggtrades", "aggregate_id_gap_events"), ("aggtrades", "id_regressions"),
    ("aggtrades", "conflicting_duplicates"), ("aggtrades", "invalid_events"),
    ("aggtrades", "reconnect_count"),
])
def test_frozen_integrity_failure_never_eligible(session, source, key):
    workspace, paths = session
    edit_json(paths.summary, lambda value: value["live_integrity"][source].update({key: 1}))
    metadata = overnight.validate(1, "123", "1", workspace=workspace, now=SCAN_TIME)
    assert metadata["result"]["status"] == "FAIL"
    assert metadata["result"]["eligible_hours"] == "0"
    assert metadata["result"]["research_eligibility"] == "INELIGIBLE"
    assert (paths.directory / overnight.VALIDATION).is_file()
    assert overnight.publish(1, successful_outcomes(), workspace=workspace)["status"] == "FAIL"


def test_interrupted_session_archivable_but_ineligible(session):
    workspace, paths = session
    paths.summary.unlink()
    assert overnight.identify(workspace) == paths.directory
    metadata = overnight.validate(1, "123", "1", workspace=workspace, now=SCAN_TIME)
    assert metadata["result"]["status"] == "FAIL"
    assert metadata["result"]["eligible_hours"] == "0"


def test_multiple_sessions_refused(session):
    workspace, _ = session
    proof = json.loads((workspace / overnight.ROOT / overnight.PREFLIGHT).read_text())
    make_session(workspace / overnight.ROOT,
                 {"definition": {"creation_source_commit_sha": proof["source_commit_sha"]}},
                 started=START + timedelta(days=1))
    with pytest.raises(ValueError, match="exactly one"):
        overnight.identify(workspace)


def test_partial_duration_refused(session):
    workspace, paths = session
    edit_json(paths.manifest, lambda value: value.update(requested_duration_seconds=120))
    edit_json(paths.summary, lambda value: value["manifest"].update(sha256=sha256_file(paths.manifest)))
    metadata = overnight.validate(1, "123", "1", workspace=workspace, now=SCAN_TIME)
    assert metadata["result"]["status"] == "FAIL"
    assert "10800" in metadata["result"]["reason"]


def test_collector_failure_is_not_rescued_by_valid_artifacts(session):
    workspace, _ = session
    metadata = overnight.validate(1, "123", "1", workspace=workspace, now=SCAN_TIME, collection_outcome="failure")
    assert metadata["result"]["research_eligibility"] == "INELIGIBLE"


@pytest.mark.parametrize("step", overnight.REQUIRED_STEPS)
def test_failed_required_step_zeroes_published_eligibility(session, step):
    workspace, _ = session
    overnight.validate(1, "123", "1", workspace=workspace, now=SCAN_TIME)
    outcomes = {**successful_outcomes(), step: "failure"}
    result = overnight.publish(1, outcomes, workspace=workspace)
    assert result["status"] == "FAIL"
    assert result["eligible_hours"] == "0"
    assert result["research_eligibility"] == "INELIGIBLE"


def test_hash_tampering_fails_validation(session):
    workspace, paths = session
    with paths.aggtrades_raw.open("a") as stream:
        stream.write("\n")
    result = overnight.validate(1, "123", "1", workspace=workspace, now=SCAN_TIME)["result"]
    assert result["status"] == "FAIL"
    assert "hash mismatch" in result["reason"]


def test_campaign_failure_isolation_and_exact_totals():
    needs = {}
    for number in range(1, 4):
        start = START + timedelta(days=number - 1)
        session_id = start.strftime("%Y%m%dT%H%M%S%fZ")
        row = {
            **overnight.empty_result(number), "session_id": session_id, "status": "PASS",
            "started_at_utc": start.isoformat(), "ended_at_utc": (start + timedelta(hours=3)).isoformat(),
            "utc_start_date": start.date().isoformat(), "research_eligibility": "ELIGIBLE", "eligible_hours": "3",
            "artifact_hashes": {name: "a" * 64 for name in (*overnight.ARTIFACTS, overnight.VALIDATION)},
            "artifact_name": overnight.artifact_name("123", "1", session_id),
        }
        needs[f"session_{number}"] = {"result": "success", "outputs": {"report": json.dumps(row)}}
    report = overnight.campaign_summary(needs)
    assert report["eligible_sessions"] == 3
    assert report["eligible_hours"] == "9"
    assert len(report["utc_start_dates"]) == 3
    needs["session_1"]["result"] = "failure"
    report = overnight.campaign_summary(needs)
    assert [row["status"] for row in report["sessions"]] == ["FAIL", "PASS", "PASS"]
    assert report["eligible_sessions"] == 2 and report["eligible_hours"] == "6"
    assert len(report["utc_start_dates"]) == 2


def test_campaign_missing_or_malformed_outputs_fail_closed():
    report = overnight.campaign_summary({"session_1": {"result": "success", "outputs": {"report": "{"}}})
    assert len(report["sessions"]) == 3
    assert report["eligible_sessions"] == 0
    assert report["eligible_hours"] == "0"
    assert report["predictive_outcomes_evaluated"] is False


def test_artifact_name_includes_run_attempt_and_session():
    session_id = START.strftime("%Y%m%dT%H%M%S%fZ")
    assert overnight.artifact_name("123", "1", session_id) != overnight.artifact_name("123", "2", session_id)
    with pytest.raises(ValueError):
        overnight.artifact_name("123\nunsafe=1", "1", session_id)


def test_overnight_workflow_static_contract():
    workflow = (WORKSPACE / ".github/workflows/v10-microstructure-overnight.yml").read_text()
    assert "on:\n  workflow_dispatch:\n" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "group: v10-microstructure-overnight-prospective-collection" in workflow
    assert "cancel-in-progress: false" in workflow
    assert workflow.count("timeout-minutes: 240") == 3
    assert workflow.count("runs-on: ubuntu-latest") == 4
    assert "needs: session_1\n    if: ${{ always() && !cancelled() }}" in workflow
    assert "needs: session_2\n    if: ${{ always() && !cancelled() }}" in workflow
    assert "needs: [session_1, session_2, session_3]" in workflow
    assert "steps: &session_steps" in workflow and workflow.count("steps: *session_steps") == 2
    assert "--duration-seconds 10800" in workflow
    assert "--output-root data/microstructure/v10" in workflow
    assert "retention-days: 4" in workflow and "actions/upload-artifact@v4" in workflow
    assert "persist-credentials: false" in workflow
    assert "--mode PREFLIGHT" in workflow and "--mode SUMMARY" in workflow
    for name in (*overnight.ARTIFACTS, overnight.VALIDATION):
        assert "${{ steps.identify.outputs.session_dir }}/" + name in workflow
    for forbidden in ("secrets.", "EVALUATE", "strategy:", "schedule:", "push:", "matrix:"):
        assert forbidden not in workflow


def test_existing_smoke_and_manifest_unchanged():
    assert sha256_file(load_manifest()[0]) == overnight.PREREGISTRATION_SHA256
    assert sha256_file(WORKSPACE / ".github/workflows/v10-microstructure-github-smoke.yml") == (
        "6d76db39319288261998de91ba5be491093cc0c9ddc5f0ea84a1eaa4caa08f75"
    )


def test_overnight_rerun_guard_is_first_in_every_job():
    workflow = (WORKSPACE / ".github/workflows/v10-microstructure-overnight.yml").read_text()
    guard = (
        "      - &first_attempt_guard\n"
        "        name: Reject research campaign re-runs\n"
        "        if: ${{ github.run_attempt != 1 }}\n"
        "        shell: bash\n"
        "        run: |\n"
        '          echo "Research campaigns must be started with a NEW workflow_dispatch run. '
        'Do not use Re-run all jobs because GitHub replaces artifacts from the previous attempt."\n'
        "          exit 1\n"
    )
    assert "    steps: &session_steps\n" + guard in workflow
    assert workflow.count("steps: *session_steps") == 2
    assert "    steps:\n      - *first_attempt_guard\n" in workflow.split("  summary:\n")[1]
    assert guard + "\n      - name: Check out repository" in workflow
    # The always-running publication step must not bypass the failed guard.
    assert "id: result\n        if: ${{ always() && github.run_attempt == 1 }}" in workflow
    assert "if: ${{ always() && steps.collect.outcome != 'skipped' }}" in workflow
    assert "if: ${{ always() && steps.identify.outcome == 'success' }}" in workflow
    assert "if: ${{ always() && steps.identify.outputs.session_dir != '' }}" in workflow
    assert "continue-on-error:" not in workflow


def test_overnight_documentation_requires_new_campaign_and_prompt_import():
    documentation = (WORKSPACE / "docs/v10_overnight_collection.md").read_text()
    assert "ALWAYS use `Run workflow` for each new campaign" in documentation
    assert "NEVER use `Re-run all jobs` for a completed research" in documentation
    assert "Download and import session artifacts before the four-day retention expiry." in documentation
