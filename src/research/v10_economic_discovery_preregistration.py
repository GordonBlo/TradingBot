"""One-time discovery protocol and read-only binding checks; no diagnostic executor."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from src.cli.validate_v10_overnight import PREREGISTRATION_ID, PREREGISTRATION_SHA256, protocol
from src.microstructure.v10 import session_paths, sha256_file
from src.research.v10_collection_preregistration import WORKSPACE, canonical, strict_json, timestamp, utc
from src.research.v10_collection_readiness import ARTIFACTS, require


ROOT = WORKSPACE / "research/v10_microstructure_economic_discovery"
DATA_ROOT = Path("data/microstructure/v10")
FILES = (*ARTIFACTS, "acquisition.validation.json")
REFERENCE_SOURCES = (
    "src/microstructure/v10.py", "src/orderbook/book.py", "src/orderbook/models.py",
    "src/orderbook/features.py", "src/orderflow/models.py",
    "src/backtest/v10_l2.py", "src/backtest/execution.py",
    "src/diagnostics/v8_derivatives_discovery.py",
    "src/research/v10_collection_readiness.py",
)
L2_FEATURES = (
    "spread_bps", "microprice_minus_mid_bps",
    "depth_imbalance_1", "depth_imbalance_5", "depth_imbalance_10", "depth_imbalance_20",
    "bid_depth_concentration_top1_over_top20", "ask_depth_concentration_top1_over_top20",
    "bid_depth_added_1s", "bid_depth_removed_1s", "ask_depth_added_1s", "ask_depth_removed_1s",
    "update_intensity_1s",
)
WINDOWS = (1, 5, 30)
FLOW_FEATURES = tuple(f"{name}_{window}s" for window in WINDOWS for name in (
    "signed_quantity", "buy_count", "sell_count", "buy_quantity", "sell_quantity",
    "signed_notional", "quantity_flow_imbalance",
))
INTERACTIONS = tuple(f"{name}_x_flow_imbalance_{window}s" for window in WINDOWS
                     for name in ("depth_imbalance_20", "microprice_minus_mid_bps"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def reference_hashes(workspace: Path = WORKSPACE) -> dict:
    return {name: hashlib.sha256((workspace / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in REFERENCE_SOURCES}


def validate_dataset(dataset: dict) -> None:
    require(dataset["root"] == DATA_ROOT.as_posix(), "noncanonical discovery data root")
    report = dataset["readiness"]
    require(digest(report) == dataset["readiness_sha256"], "readiness digest mismatch")
    require(report["preregistration_id"] == PREREGISTRATION_ID and report["status"] == "READY",
            "frozen acquisition READY required")
    require(report["manifest_sha256"] == PREREGISTRATION_SHA256, "acquisition manifest hash mismatch")
    require(report["preregistration_definition_sha256"] ==
            "8326791b411c27b5bad9631ad1cae58d22ed6fb6154977e0256f012840adc1b0"
            and report["prospective_cutoff_utc"] == "2026-09-12T08:30:00Z",
            "acquisition definition/cutoff mismatch")
    require(report["checks"] == {"eligible_sessions": True, "eligible_hours": True, "utc_dates": True},
            "acquisition gates failed")
    require(report["eligible_sessions"] == 8 and report["eligible_hours"] == "24"
            and len(report["sessions"]) == 8, "exactly eight eligible sessions and 24 hours required")
    require(report["predictive_outcomes_evaluated"] is False
            and report["independent_confirmatory_strategy_evidence"] is False
            and report["data_sufficiency_only"] is True and report["blind_holdout"] == "LOCKED",
            "invalid discovery evidence role")
    checked = utc(dataset["bound_at_utc"])
    sessions = report["sessions"]
    ids = [row["session_id"] for row in sessions]
    require(len(set(ids)) == 8 and ids == report["eligible_session_ids"], "session identity/order mismatch")
    require(sessions == sorted(sessions, key=lambda row: (row["started_at_utc"], row["session_id"])),
            "nonchronological dataset")
    previous_end = None
    for row in sessions:
        start, end = utc(row["started_at_utc"]), utc(row["ended_at_utc"])
        require(utc(report["prospective_cutoff_utc"]) < start < end <= checked, "invalid prospective interval")
        require(previous_end is None or previous_end <= start, "overlapping dataset sessions")
        previous_end = end
        relative = session_paths(DATA_ROOT, started_at=start).directory
        require(row["session_id"] == relative.name and row["manifest_path"] == (relative / "session.manifest.json").as_posix(),
                "noncanonical session identity/path")
        require(row["classification"] == "ELIGIBLE" and row["eligible_microseconds"] == 10_800_000_000,
                "ineligible or incomplete session")
        require((end - start).total_seconds() >= 10800, "closed interval shorter than eligible duration")
        require(set(row["artifact_sha256"]) == set(ARTIFACTS), "incomplete session hash binding")
        hashes = [*row["artifact_sha256"].values(), row["synchronized_timeline_sha256"],
                  row["causal_contexts_sha256"], dataset["validation_sha256"][row["session_id"]]]
        require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes),
                "invalid artifact/replay hash")
        require(re.fullmatch(r"[0-9a-f]{40}", row["source_commit_sha"]) is not None, "invalid source commit")
    require(set(dataset["validation_sha256"]) == set(ids), "validation identity set mismatch")
    dates = sorted({utc(row["started_at_utc"]).date().isoformat() for row in sessions})
    require(report["eligible_utc_dates"] == dates and len(dates) >= 3, "UTC date accounting mismatch")


def bind_ready_dataset(report: dict, *, bound_at: datetime, workspace: Path = WORKSPACE) -> dict:
    """Bind an already completed acquisition scan, rehashing bytes without reading outcomes."""
    _, acquisition = protocol()
    require(report["preregistration_definition_sha256"] == acquisition["definition_sha256"]
            and report["prospective_cutoff_utc"] == acquisition["definition"]["prospective_cutoff_utc"],
            "readiness acquisition identity mismatch")
    portable = json.loads(json.dumps(report))
    validation = {}
    expected_paths = set()
    for row in portable["sessions"]:
        directory = session_paths(workspace / DATA_ROOT, started_at=utc(row["started_at_utc"])).directory
        require(Path(row["manifest_path"]).resolve() == (directory / "session.manifest.json").resolve(),
                "readiness path differs from canonical dataset")
        require(directory.resolve().is_relative_to((workspace / DATA_ROOT).resolve()), "redirected session path")
        expected_paths.add(directory / "session.manifest.json")
        for name in FILES:
            path = directory / name
            require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), "redirected artifact")
            actual = sha256_file(path)
            if name in ARTIFACTS:
                require(actual == row["artifact_sha256"][name], f"artifact changed since READINESS: {row['session_id']}/{name}")
            else:
                validation[row["session_id"]] = actual
        metadata = strict_json((directory / "acquisition.validation.json").read_text(encoding="utf-8"))
        require(metadata["preregistration_id"] == PREREGISTRATION_ID
                and metadata["preregistration_sha256"] == PREREGISTRATION_SHA256
                and metadata["result"]["session_id"] == row["session_id"]
                and metadata["result"]["research_eligibility"] == "ELIGIBLE"
                and metadata["result"]["status"] == "PASS", "acquisition validation identity/eligibility mismatch")
        require(metadata["result"]["artifact_hashes"] == row["artifact_sha256"]
                and metadata["result"]["synchronized_timeline_sha256"] == row["synchronized_timeline_sha256"]
                and metadata["result"]["causal_contexts_sha256"] == row["causal_contexts_sha256"]
                and metadata["result"]["deterministic_dual_stream_replay"] == "PASS"
                and metadata["result"]["causal_trade_l2_merge"] == "PASS"
                and metadata["result"]["predictive_outcomes_evaluated"] is False,
                "acquisition validation differs from fresh READINESS")
        row["manifest_path"] = (directory / "session.manifest.json").relative_to(workspace).as_posix()
    require(set((workspace / DATA_ROOT).rglob("session.manifest.json")) == expected_paths,
            "session set changed since READINESS")
    dataset = {"root": DATA_ROOT.as_posix(), "bound_at_utc": timestamp(bound_at),
               "readiness": portable, "readiness_sha256": digest(portable), "validation_sha256": validation}
    validate_dataset(dataset)
    return dataset


def frozen_rules() -> dict:
    return {
        "evidence": {
            "role": "DISCOVERY_ONLY", "independent_confirmatory_strategy_validation": False,
            "outcomes_examined_for_this_protocol": False,
            "consumption": "All eight sessions become CONSUMED discovery evidence when authorized outcome analysis starts; never fresh strategy confirmation.",
            "blind_holdout": "LOCKED; no access", "consumed_v9": "EXCLUDED",
            "scope": "Information and quote-price economic magnitude; NOT a strategy, profitability or deployability test.",
        },
        "authorization": {
            "state": "PREREGISTERED_NOT_AUTHORIZED_TO_EXECUTE", "executions_permitted_now": 0,
            "maximum_future_authorized_diagnostic_runs": 1, "available_modes": ["PLAN", "VERIFY"],
            "future_requirement": "Separate explicit user authorization naming this ID; implement and synthetic-test the exact protocol before opening outcomes.",
            "one_time_seal": "Future executor must exclusively reserve a persistent run ledger before target construction; an existing reservation/result forbids another outcome run, including after interruption.",
            "pre_execution": "Require frozen manifest, reference sources, exact five-file hashes for all eight IDs and fresh acquisition READY; no added/dropped/substituted sessions.",
            "post_execution": "Recheck binding; seal input/source/runtime/report hashes and consume dataset. Integrity failure produces no research classification and no silent retry.",
        },
        "sampling": {
            "frequency_seconds": 1, "grid": "integer UTC seconds T; chronological within session",
            "merge": ["receive/availability timestamp", "depth before aggTrade", "source record index"],
            "asof": "latest valid L2 available <= query time; no future nearest lookup, interpolation or repair",
            "maximum_state_age_ms": 1000,
            "freshness": "Both availability age and last applied diff exchange-event age <= 1000ms at each query; preserve both times. No diff provenance means missing, never infer source time from a snapshot.",
            "warmup_seconds": 30,
            "warmup": "T-30s >= max(session start, first accepted aggTrade receive time, first valid synchronized L2 availability). All windows are (T-w,T]. Initial unobserved stream coverage is missing, not zero. Restart windows at each session/resync; no carry across boundaries.",
            "end": "min(closure end, session start + requested 10800s); discard closure overrun",
            "row_mask": "One common mask for both models and all four horizons: valid fresh book at T, at every T+h, T+100ms and T+h+100ms, all in the same session. Missing target endpoints exclude the row, with reason/count; never impute outcomes.",
            "features": "Only sources available <= T. Missing feature values remain missing until train-only imputation; no target-derived filters or winsorization.",
            "minimum_common_rows_per_session": 1202,
            "failure": "Insufficient rows in any session, missing required provenance, invalid book, nonfinite computation or undefined required primary statistic => INTEGRITY_FAILURE; no session/fold dropping.",
        },
        "features": {
            "l2_only_order": list(L2_FEATURES),
            "combined_order": [*L2_FEATURES, *FLOW_FEATURES, *INTERACTIONS],
            "levels": [1, 5, 10, 20], "flow_windows_seconds": list(WINDOWS),
            "l2_state": "Reuse orderbook.features.state_features at latest causal state: spread=10000*(ask-bid)/mid; microprice=(ask*bid_qty+bid*ask_qty)/(bid_qty+ask_qty); displacement=10000*(microprice-mid)/mid; imbalance_N=(bid_depth_N-ask_depth_N)/(bid_depth_N+ask_depth_N); concentration=top1/top20. Depth is base quantity.",
            "depth_flow": "Reuse event_flow_features before each successfully applied diff. Sum bid/ask quantity added/removed over (T-1s,T]; update_intensity=sum(len(b)+len(a)) per one second. Ignore stale diffs; snapshot contents are not added flow; buffered effects become available only at bridge receipt.",
            "depth_flow_limits": "Changes describe the retained reconstructed book, not identified cancellations or executed volume; no execution/cancel attribution inferred.",
            "trade_sign": "Exact boolean buyer_is_maker m=false => taker BUY sign +1; m=true => taker SELL sign -1. No sign reversal.",
            "trade_count": "Count accepted unique aggTrade records, not underlying constituent trades; deterministic duplicate suppression from acquisition replay.",
            "flow_formulas": "Per receive-time window: buy/sell counts and sum(q); signed_quantity=buy_quantity-sell_quantity; signed_notional=sum(sign*p*q); quantity_flow_imbalance=signed_quantity/(buy_quantity+sell_quantity).",
            "empty_window": "A complete observed window with zero trades gives all seven flow features zero; incomplete/unavailable windows are missing, not zero.",
            "interaction": "Exactly six products: depth_imbalance_20(T) or microprice_minus_mid_bps(T), each times quantity_flow_imbalance_w(T), w=1,5,30. Missing operand => missing product.",
            "forbidden": "No standalone absolute book-depth predictor (depth sums are intermediate inputs to imbalance/concentration only), lagged return, calendar/session-ID, volatility, candle, derivatives, extra flow window, unlisted nonlinear transform, feature selection or post-outcome family.",
        },
        "targets": {
            "horizons_seconds": [5, 30, 60, 300], "primary_seconds": 300,
            "model_target_bps": "10000*(mid(T+h)/mid(T)-1); simple return, not log return",
            "supporting_seconds": [5, 30, 60], "supporting_can_change_classification": False,
            "quote_price_target": "LONG buy ask at T+100ms, sell bid at T+h+100ms; latest causal fresh state at each endpoint, same session only.",
        },
        "economics": {
            "decision_to_quote_latency_ms": 100,
            "scenarios": {"base": {"fee_bps_per_side": "10", "additional_adverse_slippage_bps_per_side": "2"},
                          "stress": {"fee_bps_per_side": "20", "additional_adverse_slippage_bps_per_side": "4"}},
            "reference_round_trip_bps": {"base": "24", "stress": "48"},
            "arithmetic": "Decimal, isolated precision 50, ROUND_HALF_EVEN for financial features/returns/costs; convert once to binary64 at model/statistics boundary.",
            "prices": "At entry/exit quote times let a=ask_entry, b=bid_exit, me=entry_mid, mx=exit_mid; s=slippage_bps/10000, f=fee_bps/10000; A=a*(1+s), B=b*(1-s). No tick rounding.",
            "returns_bps": "mid_move=10000*(mx-me)/A; executable_gross=10000*(b-a)/A; spread=10000*((a-me)+(mx-b))/A; adverse=10000*((A-a)+(b-B))/A; fees=10000*f*(A+B)/A; net=10000*(B-A-f*(A+B))/A = mid_move-spread-adverse-fees.",
            "unadjusted_quote_return_bps": "10000*(b/a-1), separately labelled; do not mix denominators with cost decomposition.",
            "no_double_counting": "Spread is embedded in bid/ask gross; 24/48 bps are cost context, not extra deductions after exact accounting.",
            "capacity": "No order quantity, position schedule or fill simulation. Quote-side unit-price diagnostic is an optimistic infinitesimal-liquidity bound, not a top-of-book full-fill assumption. Historical exchangeInfo/causal admission inputs are absent; do not invent filters or retrospectively fetch replacements.",
            "depth_slippage": "NOT_ESTIMATED, never zero-filled as proven capacity. Future separately preregistered sized research must use depth sweeps, quantity/notional constraints and FULL_FILL_OR_REJECT with causal provenance.",
        },
        "model": {
            "family": "Ridge", "alpha": "1.0", "fit_intercept": True,
            "objective": "sum((y-intercept-X*beta)^2) + 1.0*sum(beta^2); intercept not penalized",
            "preprocessing": "Each fold/model: training-only median per feature; fail all-missing training feature; impute held-out from same medians; train-only means/population standard deviations after imputation. Zero training scale => transformed column zero for train and test.",
            "reference": "Use pure prepare_fold/fit_prepared_fold primitives in src/diagnostics/v8_derivatives_discovery.py, with explicit all-missing/nonfinite guards. Reusing statistical primitives does not reopen V8 research or load its data.",
            "baseline": "constant mean of training targets, on identical held-out rows",
            "folds": [{"train_session_indices": list(range(1, index)), "test_session_index": index}
                      for index in range(2, 9)],
            "ordering": "Frozen chronological sessions 1..8; within-session T ascending; expanding training, each later whole session tested once, no random row split.",
            "paired": "Both models use identical folds, common row mask, targets and baseline; fit separately for each of the four frozen horizons. Only 300s classifies.",
            "weights": "unweighted training rows; report pooled and equal-weight session metrics; no reweighting or sample balancing",
            "forbidden": "No model/hyperparameter search, sign reversal, clipping, feature selection, early stopping, secondary rescue or neural networks.",
        },
        "statistics": {
            "metrics": ["sample/fold/session counts", "pooled held-out Spearman", "per-session Spearman",
                        "equal-session mean Spearman", "model and constant-baseline MSE in bps^2",
                        "per-session MSE", "relative MSE improvement 1-model_MSE/baseline_MSE",
                        "combined-minus-L2 Spearman", "L2-minus-combined MSE", "positive session fractions"],
            "spearman": "Pearson of pooled average ranks (ties averaged); per-session ranks calculated within that session; no absolute values/sign selection.",
            "mse": "mean((actual-predicted)^2) across all held-out rows, not an unweighted mean of fold MSE; baseline predictions are their respective fold's training-target mean. Relative MSE improvement is null if baseline MSE is zero.",
            "null": {
                "primary_only": True, "valid_permutations": 1000, "seed": 20260916,
                "generator": "Python 3.12 random.Random(seed), MT19937; one generator, permutation index then chronological session order",
                "offset": "For each session of n common rows draw randrange(601,n-600); require min(offset,n-offset)>600. Rotate y'[j]=y[(j+offset)%n]. Same shifted targets for L2 and combined; features/time indices stay fixed.",
                "refit": "Refit both models in all seven folds for every permutation, including shifted training-target means. Cached feature-only train preprocessing/Gram matrices allowed, no reused fitted coefficients.",
                "statistics": ["combined pooled Spearman", "combined pooled Spearman minus L2 pooled Spearman"],
                "p_value": "(1 + count(null_stat >= observed_stat))/1001, one-sided positive; exactly 1000 finite statistics for each, no discarded/redrawn permutations",
                "limitations": "Overlapping targets are not IID. Circular shifts preserve ordered dependence except wrap boundary and irregular missing-row spacing; no IID t-test, row bootstrap or profitability confidence claim.",
            },
        },
        "response": {
            "quantiles": ["0.50", "0.75", "0.90", "0.95", "0.975", "0.99"],
            "estimator": "sorted values, linear interpolation at (n-1)*q (type 7); report pooled OOS prediction quantiles as descriptive distribution only",
            "bucket_thresholds": "For each observed fold/model/horizon derive those quantiles from fitted TRAINING predictions only; freeze before test. Seven disjoint buckets (-inf,q50],(q50,q75],...,(q99,+inf); equal edges produce empty bins, never merge adaptively.",
            "positive_tails": "For each predefined training quantile q report test rows prediction>max(0,q); no fitted or held-out profit optimization.",
            "metrics_each_bucket_or_tail": ["count", "session counts", "mean prediction bps", "mean realized mid bps",
                                           "median realized mid bps", "mid win rate y>0", "mean executable gross bps",
                                           "mean and median base net bps", "mean and median stress net bps",
                                           "base/stress positive-net fractions", "spread/fee/slippage decomposition"],
            "monotonicity": "Pooled realized mid means across the seven ordered fold-local bins: every adjacent difference>=0 and at least one>0. Also report Spearman(bin_index,bin_mean), per-session means and counts. Empty bin => UNDEFINED; never collapse bins. Descriptive only.",
            "economic_thresholds_bps": ["12", "24", "36", "48"],
            "exceedance": "Report count/fraction prediction>each fixed threshold for each horizon/model; no other economic thresholds.",
            "persistence": "For each fixed threshold and positive quantile tail, maximal consecutive one-second true runs within a test session; missing rows/session boundaries break runs. Duration is count of true observations seconds (singleton=1s). Report run count, mean/median/max duration, inter-onset timestamp gaps within session, and runs per eligible test hour.",
            "clustering": "Use half-open five-minute bins [session_start+300k,session_start+300(k+1)); include zero-count bins wholly within requested session coverage. Report threshold run-onset counts and population variance/mean (Fano), null if mean zero. No adaptive clustering.",
            "rate_denominator": "Eligible test hours for event/turnover rates = common held-out row count/3600; also report nominal held-out session hours separately.",
            "turnover": "Demand proxy only: every qualifying second => count signals and 2*count quote-side operations, per eligible test hour; maximum simultaneous h-second hypothetical intervals via exact timestamps. No compounded PnL, portfolio, realized trades or invented capital/quantity.",
            "economic_audit_anchors": "Only combined 300s positive q95 tail: take each run's first T, greedily retain in time order when T>=previous_retained_T+300s+100ms; reset per session. This de-overlaps diagnostic intervals, not an executable strategy schedule; no optimized rearm/cooldown/holding rule.",
            "horizon_comparison": "On common OOS rows report 300s minus 5/30/60s Spearman, mean/quantile prediction scale and frozen-tail net response. Material economic-scale increase descriptive flag iff 300s q95 mean base-net minus 30s q95 mean base-net >=12bps (each horizon's own frozen train thresholds). No secondary classification rescue.",
        },
        "classification": {
            "primary_horizon_seconds": 300,
            "information_gate_all_required": {
                "combined_pooled_spearman": ">0", "combined_minus_l2_pooled_spearman": ">0",
                "combined_MSE": "< L2_MSE AND < constant_baseline_MSE",
                "positive_combined_session_spearman": ">=5 of 7",
                "positive_combined_minus_l2_session_spearman": ">=5 of 7",
                "combined_shift_p": "<0.05", "incremental_shift_p": "<0.05",
            },
            "economic_gate_all_required": {
                "only_region": "combined primary prediction > max(0, training q95)",
                "tail_coverage": ">=100 total held-out rows; >=20 rows in each of >=5 held-out sessions",
                "tail_base_net": "pooled mean >0 AND mean >0 in >=5 of 7 held-out sessions; missing session tail counts as nonpositive",
                "anchor_coverage": ">=20 non-overlapping audit anchors across >=4 held-out sessions",
                "anchor_base_net": "pooled mean >0",
                "stress": "Mandatory descriptive report, NOT a classification gate; never substitute stress/base or select another tail after results.",
            },
            "ordered_decision": [
                "Any integrity/provenance/required-statistic failure => INTEGRITY_FAILURE, no research classification.",
                "Information gate false => NO_STABLE_COMBINED_SIGNAL.",
                "Information gate true and economic gate false => INFORMATION_PRESENT_BUT_TOO_SMALL.",
                "Both gates true => ECONOMIC_SIGNAL_PRESENT.",
            ],
            "interpretation": "Exactly one discovery label after authorization, never independent strategy validation. TOO_SMALL means insufficient prespecified after-cost evidence, including insufficient tail/anchor coverage; report which gates fail. No best-bin or shorter-horizon promotion.",
        },
        "report": {
            "required": "Every horizon/model/fold and every frozen bin/threshold, including negative/empty results; exclusions; gate booleans; one primary label; causal endpoint timestamps; source/runtime versions; manifest/input/prediction/null-statistic/report hashes.",
            "precision": "Apply gates to unrounded computed values, never displayed rounding. Empty descriptive buckets/tails have count zero and null means/medians/win rates; insufficient economic coverage fails that gate, not integrity. Undefined required primary correlations or null statistics fail integrity.",
            "location": "ignored reports/v10_microstructure_economic_discovery/<preregistration_id>/; immutable exclusive-create ledger and report, never overwrite historical evidence",
            "trading_limitations": "No depth capacity, live exchange admission, queue priority, passive/maker fill, inventory/risk, capital, impact, fee-asset discount, or trading profitability claim. Later sized trading research requires new preregistration and genuinely new prospective sessions.",
        },
    }


def build_manifest(created_at: datetime, source_commit: str, sources: dict, dataset: dict) -> dict:
    validate_dataset(dataset)
    require(utc(dataset["bound_at_utc"]) <= utc(timestamp(created_at)), "creation predates dataset binding")
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "invalid creation commit")
    require(set(sources) == set(REFERENCE_SOURCES)
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values()), "invalid reference source hashes")
    definition = {
        "protocol_version": "V10_MICROSTRUCTURE_ECONOMIC_DISCOVERY_V1",
        "created_at_utc": timestamp(created_at), "creation_source_commit_sha": source_commit,
        "reference_sources_sha256_lf_normalized": sources, "dataset": dataset, "rules": frozen_rules(),
    }
    value = digest(definition)
    return {"preregistration_id": value[:16], "definition_sha256": value, "definition": definition}


def verify_manifest(manifest: dict) -> None:
    definition = manifest["definition"]
    expected = build_manifest(utc(definition["created_at_utc"]), definition["creation_source_commit_sha"],
                              definition["reference_sources_sha256_lf_normalized"], definition["dataset"])
    require(canonical(manifest) == canonical(expected), "discovery preregistration mismatch")


def load_manifest(root: Path = ROOT) -> tuple[Path, dict]:
    paths = sorted(root.glob("*/manifest.json"))
    require(len(paths) == 1, "require exactly one frozen discovery manifest; never auto-create")
    manifest = strict_json(paths[0].read_text(encoding="utf-8"))
    verify_manifest(manifest)
    require(paths[0].parent.name == manifest["preregistration_id"], "discovery path/ID mismatch")
    return paths[0], manifest


def verify_dataset_binding(manifest: dict, *, workspace: Path = WORKSPACE) -> None:
    """Hash verification only. Does not reconstruct features, targets, fits or predictions."""
    verify_manifest(manifest)
    protocol()
    require(reference_hashes(workspace) == manifest["definition"]["reference_sources_sha256_lf_normalized"],
            "discovery reference source changed")
    dataset = manifest["definition"]["dataset"]
    restored = json.loads(json.dumps(dataset["readiness"]))
    for row in restored["sessions"]:
        row["manifest_path"] = str(workspace / row["manifest_path"])
    rebound = bind_ready_dataset(restored, bound_at=utc(dataset["bound_at_utc"]), workspace=workspace)
    require(canonical(rebound) == canonical(dataset), "frozen discovery dataset changed")
