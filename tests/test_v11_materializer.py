from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from src.backtest.v10_l2 import (
    DepthLevel,
    DepthSnapshot,
    ExchangeConstraints,
    QuantityFilter,
)
from src.backtest.v11_entry_filter import BAR, L2IntegrityError, ParentCandidate
from src.models.candle import Candle
from src.research import v11_materializer as module
from src.research.v11_score import load_bundle, score

D = Decimal
START = datetime(2026, 10, 1, tzinfo=UTC)  # Fabricated; never loaded from market data.
CONSTRAINTS = ExchangeConstraints('BTCUSDC', QuantityFilter(D('.0001'), D(100), D('.0001')), None, ())


def book(at, bid, ask, sequence):
    return DepthSnapshot('BTCUSDC', at, at, sequence,
                         (DepthLevel(D(bid), D(10)),), (DepthLevel(D(ask), D(10)),))


def fixture():
    bars = []
    for i in range(960):
        value = D(100+i//16) if i < 864 else D(160)
        if i in (864, 866, 872):
            value = D(140)
        c = Candle(START+i*BAR, 'BTCUSDC', '15m', value, value+1, value-1, value, D(1), True)
        bars.append(module.ClosedBar(c, c.timestamp+BAR+timedelta(milliseconds=10)))
    snapshots = []
    for j, i in enumerate((866, 874)):
        at = START+i*BAR
        snapshots.extend([
            book(at, '159.9', '160', j*4),
            book(at+timedelta(milliseconds=110), '159.9', '160', j*4+1),
            book(at+timedelta(milliseconds=200), '140', '141', j*4+2),
            book(at+timedelta(milliseconds=300), '139', '140', j*4+3),
        ])
    return bars, snapshots


def buckets(positive):
    model = load_bundle(module.BUNDLE, expected_sha256=module.BUNDLE_SHA)['model']
    # Synthetic reference vectors exercise BOTH sides of the unchanged score.
    direction = 1 if positive else -1
    values = tuple(mean+direction*1e7*coefficient*scale for mean, coefficient, scale in
                   zip(model['means'], model['coefficients'], model['scales'], strict=True))
    assert (score(model, values) > 0) is positive
    return [module.ScoreBucket(START+i*BAR, START+i*BAR-timedelta(milliseconds=1), values)
            for i in (866, 874)]


def run(bars=None, snapshots=None, features=None, **kwargs):
    original_bars, original_books = fixture()
    return module.materialize(bars if bars is not None else original_bars,
                              snapshots if snapshots is not None else original_books,
                              features if features is not None else buckets(True),
                              constraints=CONSTRAINTS, cutoff=START-timedelta(seconds=1),
                              evaluation_start=START+timedelta(days=9),
                              evaluation_end=START+timedelta(days=10), **kwargs)


def test_end_to_end_frozen_generator_and_score_subset_only(tmp_path, monkeypatch):
    calls = []
    original = module.V6.evaluate
    def record(self, context):
        decision = original(self, context)
        calls.append((context, decision))
        return decision
    monkeypatch.setattr(module.V6, 'evaluate', record)
    accepted = run()
    rejected = run(features=buckets(False))
    assert accepted.reference == rejected.reference
    assert len(accepted.reference) == 2
    assert accepted.paired.accepted_count == 2
    assert rejected.paired.accepted_count == 0
    assert accepted.candidates[0].risk_budget == D(5)
    assert accepted.candidates[1].parent_cash_at_signal < D(1000)
    assert accepted.candidates[1].risk_budget < D(5)
    assert accepted.risk['filtered'] == accepted.risk['parent']
    assert rejected.risk['filtered']['final_cash'] == D(1000)
    assert rejected.risk['filtered']['position_seconds'] == 0
    assert rejected.risk['filtered']['max_liquidation_drawdown_usdc'] == 0
    assert any(d.reason_code.value == 'COOLDOWN' for _, d in calls)
    assert all(c.recent_history[-1] == c.current_candle for c, _ in calls)
    path = tmp_path/'ledger.json'
    digest = accepted.write_once(path)
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        accepted.write_once(path)


def test_missing_score_rejects_without_changing_schedule():
    missing = run(features=[])
    reference = run()
    assert missing.reference == reference.reference
    assert missing.paired.accepted_count == 0
    assert all(c.reason == 'MISSING_BUCKET' for c in missing.candidates)


def test_future_candles_do_not_change_earlier_parent_or_id():
    bars, _ = fixture()
    # Only after the final signal; perturb unused future completed/forming 4h inputs.
    for i in range(944, 960):
        bars[i] = replace(bars[i], candle=replace(bars[i].candle, open=D(500), high=D(501), low=D(499), close=D(500)))
    baseline = run()
    changed = run(bars=bars)
    assert baseline.reference == changed.reference
    assert baseline.candidates == changed.candidates
    assert baseline.input_sha256 != changed.input_sha256


def test_execution_price_does_not_define_signal_identity():
    bars, snapshots = fixture()
    snapshots[3] = book(snapshots[3].available_at, '130', '131', 3)
    changed = run(bars, snapshots)
    baseline = run()
    assert changed.candidates[0].candidate_id == baseline.candidates[0].candidate_id
    assert changed.candidates[0].parent_cash_at_signal == D(1000)
    assert changed.candidates[1].parent_cash_at_signal < baseline.candidates[1].parent_cash_at_signal


def test_completed_4h_receive_time_may_follow_close_but_not_decision():
    at = START+BAR
    ParentCandidate('valid', START, at, at+timedelta(milliseconds=10), True,
                    D(1), at+timedelta(milliseconds=5), D(1))
    with pytest.raises(L2IntegrityError, match='future parent ATR'):
        ParentCandidate('bad', START, at, at+timedelta(milliseconds=10), True,
                        D(1), at+timedelta(milliseconds=11), D(1))


@pytest.mark.parametrize('issue', ['gap', 'forming', 'early_receipt', 'late_receipt', 'future_score', 'order', 'duplicate_score'])
def test_causal_input_failures(issue):
    bars, snapshots = fixture()
    features = buckets(True)
    with pytest.raises(L2IntegrityError):
        if issue == 'gap':
            bars.pop(100)
        elif issue == 'forming':
            bars[900] = replace(bars[900], candle=replace(bars[900].candle, is_closed=False))
        elif issue == 'early_receipt':
            bars[0] = replace(bars[0], received_at=START)
        elif issue == 'late_receipt':
            bars[0] = replace(bars[0], received_at=START+BAR+timedelta(seconds=2))
        elif issue == 'future_score':
            features[0] = replace(features[0], available_at=features[0].close+timedelta(microseconds=1))
        elif issue == 'order':
            features[0] = replace(features[0], feature_order=tuple(reversed(features[0].feature_order)))
        else:
            features = features*2
        run(bars, snapshots, features)


def test_cutoff_and_warmup_enforced():
    bars, snapshots = fixture()
    with pytest.raises(L2IntegrityError, match='post-cutoff'):
        module.materialize(bars, snapshots, [], constraints=CONSTRAINTS,
                           cutoff=START, evaluation_start=START+timedelta(days=9),
                           evaluation_end=START+timedelta(days=10))


def test_generator_binding_refuses_changed_source(tmp_path):
    for name in module.V6_BINDINGS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('changed')
    with pytest.raises(L2IntegrityError, match='binding'):
        module.engineering_binding(tmp_path)


def test_risk_sizing_and_cash_caps():
    snapshot = book(START, '99', '100', 1)
    q = module.sized_quantity(snapshot, CONSTRAINTS, risk_budget=D(5), atr=D(2), max_quote=D(50))
    assert q == D('.4999')
    assert q*D('100.02') <= D(50)
    q = module.sized_quantity(snapshot, CONSTRAINTS, risk_budget=D('.1'), atr=D(2), max_quote=D(50))
    assert q == D('.05')
    shallow = replace(snapshot, asks=(DepthLevel(D(100), D('.01')),))
    with pytest.raises(L2IntegrityError, match='no liquidity downsizing'):
        module.sized_quantity(shallow, CONSTRAINTS, risk_budget=D(5), atr=D(2), max_quote=D(50))


def test_external_decimal_context_cannot_change_materialization():
    bars, snapshots = fixture()
    features = buckets(True)
    with localcontext() as ctx:
        ctx.prec = 12
        low = run(bars, snapshots, features)
    with localcontext() as ctx:
        ctx.prec = 42
        high = run(bars, snapshots, features)
    assert low == high


def test_rejected_open_parent_still_blocks_later_entries():
    bars, snapshots = fixture()
    entry_close = START+866*BAR
    # First parent remains open through the next two reclaim signals.
    snapshots = snapshots[:2]
    for second in range(1, 10800):
        snapshots.append(book(entry_close+timedelta(seconds=second), '159.9', '160', second+1))
    snapshots += [book(entry_close+timedelta(seconds=10800), '140', '141', 10801),
                  book(entry_close+timedelta(seconds=10800, milliseconds=100), '139', '140', 10802)]
    rejected = run(bars, snapshots, buckets(False))
    assert len(rejected.reference) == 1
    assert rejected.paired.accepted_count == 0
    assert rejected.reference[0].trade.holding_time > timedelta(hours=2)


def test_liquidation_risk_report_forbids_borrowing(monkeypatch):
    ledger = run()
    refs = ledger.reference
    # Force a purely synthetic reporting account below the fixed entry cost.
    monkeypatch.setattr(module, 'INITIAL_CASH', D(1))
    _, snapshots = fixture()
    with pytest.raises(L2IntegrityError, match='no borrowing'):
        module._risk_report(refs, [row.decision for row in ledger.paired.rows], snapshots)
