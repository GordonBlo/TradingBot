import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from src.diagnostics.v8_derivatives_discovery import fit_prepared_fold, prepare_fold
from src.diagnostics.v9_l2_early_information import (
    _INPUT_COLUMNS,
    DiagnosticIntegrityError,
    samples_from_one_second_rows,
)
from src.research import v11_score as module
from src.research.v9_l2_readiness import SessionArtifactBinding, SessionEligibility


def rows():
    return [tuple(None if (i + j) % 9 == 0 else (7.0 if j == 14 else float(i*j + i % 3))
                  for j in range(15)) for i in range(20)]


def test_export_reproduces_frozen_reference_bit_for_bit(tmp_path):
    training = rows()
    targets = [i / 10000 - 0.001 for i in range(len(training))]
    probes = [training[0], (None,) * 15, tuple(float(i) for i in range(15))]
    fold = prepare_fold(training + probes, training_indices=range(20),
                        held_out_indices=range(20, 23), alpha=1)
    reference = fit_prepared_fold(fold, targets)
    model = module.fit_static_model(training, targets)
    assert tuple(model['coefficients']) == reference.coefficients
    assert model['intercept'] == reference.intercept == sum(targets) / len(targets)
    assert tuple(model['medians']) == fold.medians
    assert tuple(model['means']) == fold.means
    assert tuple(model['scales']) == fold.scales
    assert model['scales'][-1] == 0
    bundle = {'status': 'CONSUMED_EVIDENCE_MODEL_PREPARATION_ONLY', 'model': model,
              'model_sha256': hashlib.sha256(module.encoded(model)).hexdigest()}
    path = tmp_path / 'bundle.json'
    path.write_bytes(module.encoded(bundle))
    digest = module.sha256(path)
    loaded = module.load_bundle(path, expected_sha256=digest)
    assert tuple(module.score(loaded['model'], row) for row in probes) == reference.predictions
    assert module.encoded(loaded) == path.read_bytes()
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash'):
        module.load_bundle(path, expected_sha256=digest)


def test_one_solver_invocation_and_no_training_dummy_leakage(monkeypatch):
    calls = []
    original = module.fit_prepared_fold
    def counted(fold, targets):
        calls.append(fold)
        assert len(targets) == 20  # Dummy target is absent, not held-out evidence.
        return original(fold, targets)
    monkeypatch.setattr(module, 'fit_prepared_fold', counted)
    module.fit_static_model(rows(), [0.1] * 20)
    assert len(calls) == 1
    assert calls[0].training_indices == tuple(range(20))
    assert calls[0].held_out_indices == (20,)


@pytest.mark.parametrize('kind', ['width', 'all_missing', 'infinity', 'target', 'empty'])
def test_invalid_training_fails_before_fit(kind, monkeypatch):
    data, targets = rows(), [0.1] * 20
    if kind == 'width':
        data[0] = data[0][:-1]
    elif kind == 'all_missing':
        data = [(None, *row[1:]) for row in data]
    elif kind == 'infinity':
        data[0] = (float('inf'), *data[0][1:])
    elif kind == 'target':
        targets[0] = float('nan')
    else:
        data = []
    monkeypatch.setattr(module, 'fit_prepared_fold', lambda *a: pytest.fail('must not fit'))
    with pytest.raises(ValueError):
        module.fit_static_model(data, targets)


def test_bad_model_and_nonfinite_zero_scale_feature():
    model = module.fit_static_model(rows(), [0.1] * 20)
    bad = dict(model, feature_order=list(reversed(model['feature_order'])))
    with pytest.raises(ValueError, match='semantics'):
        module.score(bad, rows()[0])
    with pytest.raises(ValueError, match='nonfinite'):
        module.score(model, (*rows()[0][:-1], float('nan')))


def test_reserved_directory_never_refits(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'bound_preparation_inputs', lambda *_: pytest.fail('must not load'))
    with pytest.raises(FileExistsError, match='do not refit'):
        module.prepare_once(tmp_path, tmp_path)


def test_primary_target_uses_exact_30_seconds_inside_bound_session():
    start = datetime(2026, 9, 24, tzinfo=UTC)
    binding = SessionArtifactBinding('synthetic', 'raw', 'a', 'closure', 'b', 'features', 'c')
    session = SessionEligibility('synthetic', start.isoformat(),
                                 (start + timedelta(seconds=32)).isoformat(),
                                 32/3600, 'ELIGIBLE', True, 'synthetic', binding)
    values = []
    for i in range(33):
        at = start + timedelta(seconds=i)
        values.append({'bucket_open_utc': (at - timedelta(seconds=1)).isoformat(),
                       'bucket_close_utc': at.isoformat(),
                       'last_feature_available_at_utc': at.isoformat(),
                       'mid_price_last': 100 + i, **{name: 1 for name in _INPUT_COLUMNS}})
    samples = samples_from_one_second_rows(session, values)
    assert len(samples) == 3
    assert all(s.target_timestamp(30) == s.timestamp + timedelta(seconds=30) for s in samples)
    missing = samples_from_one_second_rows(session, values[:30] + values[31:])
    assert len(missing) == 2  # No nearest match for the removed T+30 endpoint.
    values[0]['last_feature_available_at_utc'] = (start + timedelta(milliseconds=1)).isoformat()
    with pytest.raises(DiagnosticIntegrityError, match='future'):
        samples_from_one_second_rows(session, values)


def test_production_bundle_identity_without_loading_data_or_refitting():
    # This test remains synthetic-only until the one authorized bundle exists.
    from pathlib import Path
    folder = Path('research/v11_preparation/v9_static_score')
    if not folder.exists():
        return
    digest = (folder / 'score_bundle.sha256').read_text().strip()
    assert digest == 'c2fb47fb459a97024ebedc12af0354d6c28792e2dce87c1207eba2ba5e4664e2'
    bundle = module.load_bundle(folder / 'score_bundle.json', expected_sha256=digest)
    assert bundle['training_rows'] == 86127
    assert bundle['real_fit_count'] == 1
    assert bundle['binding']['evidence_status'] == 'CONSUMED'
    assert bundle['binding']['report_sha256'] == module.REPORT_SHA
    report = json.loads(module.REPORT.read_text())
    actual = [s['artifact_binding'] for s in bundle['binding']['sessions']]
    expected = report['artifact_bindings']
    assert [{k: item[k] for k in expected[0]} for item in actual] == expected
    for name, expected_sha in bundle['binding']['identity']['source_sha256'].items():
        assert module.sha256(Path(name)) == expected_sha
    assert module.sha256(folder / 'preparation_binding.json') == json.loads(
        (folder / 'fit_reservation.json').read_text())['binding_sha256']


def test_failed_preparation_keeps_reservation_and_cannot_retry(tmp_path, monkeypatch):
    from src.research.v9_l2_readiness import ReadinessReport
    readiness = ReadinessReport('2026-09-24T00:00:00Z', 0, 0, 0, 0, 0, 0, 0, 0,
                                False, (), ())
    monkeypatch.setattr(module, 'bound_preparation_inputs', lambda *_: (readiness, {}))
    monkeypatch.setattr(module, 'sha256', lambda *_: '0' * 64)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: 'synthetic-head')
    def fail_loading(*a, **kw):
        raise ValueError('synthetic bound input failure')
    monkeypatch.setattr(module, 'load_readiness_bound_samples', fail_loading)
    monkeypatch.setattr(module, 'fit_static_model', lambda *a: pytest.fail('must not fit'))
    folder = tmp_path / 'reserved'
    with pytest.raises(ValueError, match='synthetic bound input failure'):
        module.prepare_once(tmp_path, folder)
    assert (folder / 'fit_reservation.json').is_file()
    assert not (folder / 'score_bundle.json').exists()
    with pytest.raises(FileExistsError, match='do not refit'):
        module.prepare_once(tmp_path, folder)
