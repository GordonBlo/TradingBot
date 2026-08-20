# Trading Bot V3.2.1 — Multi-Regime Research Expansion

A production-oriented Python foundation for observing BTC/USDC Spot markets and
replaying validated historical data through a deterministic, offline simulated
execution engine and a causal strategy-research workflow. V3 preserves the
V1.1 real-public-market/Testnet boundary and the V2 execution model.
V3.1 added offline post-trade diagnosis without changing the frozen baseline.
V3.2 adds five pre-registered, one-change-at-a-time experiments (H0-H4),
consumed-data labeling, and an irreversible blind-holdout workflow. It performs
no parameter search and does not change production application behavior.
V3.2.1 keeps H0-H4 frozen while expanding BTCUSDC history across fixed 90-day
chronological windows and causal, diagnostic-only market regimes.

> **Safety warning:** V3.2 does not place trades. It contains no Binance order
> creation, withdrawal, Futures, margin, leverage, shorting, or production
> execution path. `ENTER_LONG`/`EXIT_LONG` decisions reach only the local V2
> simulator.

Indicators and backtests describe historical behavior. A profitable backtest
does **not** guarantee future profitability.

## Current capabilities

- Real Binance public REST price, exchange information, and historical OHLCV
- Real Binance public WebSocket 15-minute candles with reconnection
- Closed-candle UTC semantics and REST/WebSocket deduplication
- SMA 20/50, EMA 20/50, RSI 14, ATR 14, volume SMA 20 and volume ratio
- Paginated public historical downloads for arbitrary UTC date ranges
- Incremental CSV cache with separate JSON provenance metadata
- Strict gap, ordering, timestamp, closed-candle, and OHLCV validation
- Offline chronological replay with current/past-only decision context
- Next-bar-open fills preventing close-price look-ahead
- Long-only USDC/BTC simulated account with fees and adverse slippage
- Conservative stop-loss/take-profit handling (`STOP_FIRST` ambiguity policy)
- Trade journal, equity curve, deterministic run ID, configuration and metrics
- Typed, past-only strategy contexts and structured decisions
- One fixed `TrendMomentumBaselineStrategy` research instrument
- Risk-based sizing, fill-time ATR brackets, trend/time exits, and cooldown
- Independent development, validation, and out-of-sample backtests
- Same-period buy-and-hold comparison and optional 2× trading-cost stress
- Signal journals, sample warnings, stability diagnostics, reproducible reports
- Post-trade MFE/MAE, R-multiple, cost, exit, entry, regime and holding diagnostics
- Exact H0 frozen-baseline reproduction before any V3.2 comparison is accepted
- Causal H1 ATR% and H2 EMA-spread guards using only 100 prior observations
- Stateful H3 eight-bar pullback confirmation and H4 fixed 1R target
- Deterministic H0-H4 deltas, signal alignment, journals and 2× cost stress
- Persistent consumed-data and forward-only blind-holdout research manifest
- Fixed, non-overlapping 90-day research windows with independent accounts
- Gap-safe local warm-up that cannot read the locked blind holdout
- Causal EMA200/EMA50 trend and prior-200-observation ATR% regime diagnostics
- Window-consistency, distribution, worst/best-window, and exact support-gate reports
- No automatic orders or real-money execution

Financial values and simulated accounting use Python `Decimal`.

## Environment boundary

Analytics and historical downloads use the market-data-only endpoints:

```text
REST       https://data-api.binance.vision
WebSocket  wss://data-stream.binance.vision
```

They require no API key, secret, signature, or authenticated header. The
separate private client remains fixed to Binance Spot Testnet at
`https://testnet.binance.vision`, is reserved for future V4 work, and is not
initialized by the downloader or backtester. Live execution configuration is
rejected.

```text
REAL Binance PUBLIC Spot                  Binance Spot TESTNET
price / OHLCV / analytics                 future private boundary
             │                                      │
             v                                      v
historical cache + offline replay          execution DISABLED in V3
```

## V2 architecture

```text
Binance PUBLIC historical REST (download/update only)
                         │
                         v
          CSV candles + JSON provenance metadata
                         │
                         v
        strict DatasetValidator (closed, UTC, gap-free)
                         │
                         v
             OFFLINE BacktestEngine replay
                         │
          current/past-only decision context
                         │
                 next-bar-open intent
                         v
       simulated Spot account + fees + slippage
                         │
                         v
       trades.csv / equity.csv / summary.json / config.json
```

Once a dataset is loaded, `BacktestEngine` has no exchange client and performs
no network operations. A decision made after candle `N` closes can fill only at
candle `N+1` open. A final-candle decision is not executed. Warm-up candles can
be excluded from actionable periods without hiding them from later context.

V2 is strategy-agnostic. Its CLI uses a clearly labeled buy-and-hold accounting
benchmark solely to verify fills and reporting. It is not a candidate strategy
and performs no parameter search or optimization.

## V3 strategy and research architecture

`StrategyContext` exposes the current closed candle, a bounded history ending at
that candle, current/previous causal indicator snapshots, and read-only simulated
account state. It never exposes the next candle. `StrategyDecision` supports only
`ENTER_LONG`, `EXIT_LONG`, and `HOLD`. The V3 adapter is the only bridge to the
V2 `BacktestEngine`; it translates decisions into local simulated intents and
records non-HOLD signals.

```text
validated local candles
        │
        ├── development [start,end) ─┐
        ├── validation  [start,end) ─┼─ fresh account + same frozen strategy
        └── OOS         [start,end) ─┘
                                      │
                 V2 simulator + buy-and-hold benchmark
                                      │
        comparison.csv / signals.csv / period reports
```

Each split is chronological, contiguous, non-overlapping, and independently
funded with the same initial capital. Positions and equity never carry between
periods. Development is for initial inspection, validation is for a separate
check, and out-of-sample is protected final evaluation. Changing the strategy
after inspecting OOS invalidates that data's OOS status.

That invalidation has now occurred: V3.1 diagnostics were inspected and used to
form V3.2 hypotheses. Therefore all three old ranges—including the former V3
OOS range—are now explicitly `CONSUMED_RESEARCH_DATA`. Original names remain in
metadata only for traceability. No V3.2 result from those ranges is fresh OOS
evidence, and reports must never describe it that way.

The baseline is deliberately simple and unoptimized. On a closed candle it
enters only on an EMA20-over-EMA50 crossover when close is above EMA50, RSI14 is
within 52–68, volume ratio is at least 0.80, and every required indicator is
available. It risks an approximate 0.50% of current simulated equity, capped by
available fee-aware Spot cash and `MAX_CAPITAL_USDC`. At the actual next-bar
fill, the fixed stop distance is signal-time ATR14 × 2 and the target is 2R. It
exits through the V2 stop/target logic, EMA20 below EMA50, 96 completed position
bars, or final liquidation. A four-bar cooldown follows each completed trade.
There is no trailing stop, leverage, shorting, or pyramiding.

Fees, slippage, and stop gaps mean realized loss can differ from the ideal risk
budget. Entry sizing reserves the simulated entry fee and never borrows cash.

The baseline strategy is a research instrument. It is **not proven profitable**.
Backtest results do **not** guarantee future performance. V3 intentionally does
not search EMA periods, RSI levels, ATR multipliers, reward/risk ratios, or
volume filters: repeatedly tuning historical results creates backtest
overfitting.

## V3.1 post-trade diagnostics

V3.1 consumes an existing V3 report, its trade/signal journals, and the exact
checksum-verified local candle cache. It does not rerun the strategy, download
data, or initialize an exchange client.

```text
frozen V3 report + validated local OHLC candles
                         │
                         v
       per-trade post-trade diagnostic construction
                         │
         ┌───────────────┼────────────────┐
         v               v                v
     MFE / MAE      cost attribution    entry features
     R multiples    exits / holding     causal regimes
         └───────────────┼────────────────┘
                         v
        isolated Development / Validation / OOS reports
```

For each completed long trade, MFE is the greatest observed high above the
entry fill and MAE is the greatest observed low below it over the V2
`bars_held` candles. Signal-time ATR × the unchanged baseline multiplier defines
initial 1R. If signal ATR is absent or invalid, R-normalized fields remain
`null`; diagnostics never invent them.

OHLC data does not reveal intrabar ordering. An exit candle can show both a high
and low without proving which occurred first, and its recorded extremes may
include movement after the protective trigger. V3.1 reports the observed-candle
excursions and explicitly documents this limitation.

Cost attribution reconstructs frictionless entry/exit reference prices from the
configured V2 adverse-slippage equations, then reports:

```text
frictionless market-movement PnL
+ signed slippage drag
- recorded entry/exit fees
= recorded net PnL
```

Entry diagnostics use only values stored at entry signal time: RSI, ATR and
ATR%, volume ratio, EMA values/spread, close versus slow EMA, UTC time, and a
causal volatility regime based on the preceding/current 100-bar ATR%
distribution. Feature quartiles are descriptive report bins, not thresholds.

MFE, MAE, their timing, early-failure measures, and any other future path
information exist only in `src/diagnostics/`. They are never added to
`StrategyContext`, `StrategyDecision`, sizing, entries, or exits.

V3.1 does **not** optimize the strategy. Diagnostic correlations are not
automatically trading rules. Post-trade information must never be used by
strategy decisions. No subgroup is promoted into a strategy filter.

```text
src/
├── analysis/     # Existing indicators and snapshot formatting
├── backtest/     # Offline replay, account, fills, metrics and reporting
├── cli/          # Historical download, benchmark and research entry points
├── config/       # Environment and simulation-assumption validation
├── diagnostics/  # Offline post-trade evidence and deterministic reports
├── exchange/     # Public market data plus isolated Testnet-private client
├── historical/   # Pagination, validation and CSV/JSON cache management
├── hypotheses/   # V3.2 registry, candidates, manifest, runner and reports
├── market/       # Candle history and interval arithmetic
├── models/       # Candle and indicator domain models
├── risk/         # Existing risk-calculation foundation
├── research/     # Splits, evaluation, comparison and research reports
├── strategy/     # Typed strategy contract and the single baseline
└── main.py       # Existing live read-only observer
```

## Execution assumptions

- Long BTC/USDC Spot only; one position; no pyramiding or negative BTC.
- `quote_amount` is entry notional; its entry fee is charged in addition.
- Fees apply to both entry and exit notionals.
- Buy slippage increases fill price; sell slippage decreases fill price.
- Stop gaps use the worse candle open before sell slippage.
- If stop and target are both touched, `STOP_FIRST` is used conservatively.
- A remaining position closes at the final closed candle price with sell costs
  and `END_OF_BACKTEST` as its exit reason.
- Trade return is net PnL divided by entry notional.
- Maximum drawdown is calculated from the candle-by-candle equity curve.

Fee and slippage settings are simulation assumptions, not guaranteed or
account-specific Binance rates.

## Installation

Python 3.12 or newer is required.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Configuration

```env
MARKET_DATA_SOURCE=binance_public
EXECUTION_ENV=testnet

TRADING_SYMBOL=BTCUSDC
CANDLE_INTERVAL=15m
HISTORICAL_CANDLE_LIMIT=300

BINANCE_API_KEY=
BINANCE_SECRET_KEY=

BACKTEST_INITIAL_CAPITAL_USDC=1000
BACKTEST_FEE_BPS=10
BACKTEST_SLIPPAGE_BPS=2
BACKTEST_AMBIGUOUS_BAR_POLICY=STOP_FIRST

MAX_CAPITAL_USDC=50
MAX_RISK_PER_TRADE_PERCENT=0.5
MAX_DAILY_LOSS_PERCENT=2.0

BASELINE_FAST_EMA_PERIOD=20
BASELINE_SLOW_EMA_PERIOD=50
BASELINE_RSI_PERIOD=14
BASELINE_RSI_MIN=52
BASELINE_RSI_MAX=68
BASELINE_VOLUME_SMA_PERIOD=20
BASELINE_MIN_VOLUME_RATIO=0.80
BASELINE_ATR_PERIOD=14
BASELINE_ATR_STOP_MULTIPLIER=2.0
BASELINE_REWARD_RISK_RATIO=2.0
BASELINE_RISK_PER_TRADE_PERCENT=0.50
BASELINE_MAX_BARS_IN_POSITION=96
BASELINE_COOLDOWN_BARS=4
RESEARCH_MIN_TRADES_WARNING=30
H1_ATR_PERCENTILE_LOOKBACK=100
H1_MAX_PERCENTILE=75
H2_EMA_SPREAD_LOOKBACK=100
H2_MAX_PERCENTILE=75
H3_CONFIRMATION_WINDOW_BARS=8
H4_REWARD_RISK_RATIO=1.0
HYPOTHESIS_TRADE_COUNT_RATIO_WARNING=0.40
MULTIREGIME_WINDOW_DAYS=90
MULTIREGIME_MIN_WINDOW_DAYS=60
MULTIREGIME_VOL_LOOKBACK=200
MULTIREGIME_VOL_LOW_PERCENTILE=33
MULTIREGIME_VOL_HIGH_PERCENTILE=67
```

`MARKET_DATA_SOURCE` supports `binance_public` only. `EXECUTION_ENV` supports
`testnet` only. The former `BINANCE_TESTNET` flag cannot redirect analytics.
Empty credentials are sufficient for all V3.2 data, backtest, and research operations.

## Historical data download

Ranges use an inclusive start and exclusive end. The downloader requests at
most 1,000 candles per page, advances from the last open by one exact interval,
deduplicates pages, uses bounded exponential retry, and honors numeric
`Retry-After` after HTTP 429. It never falls back to Testnet.

```powershell
.\.venv\Scripts\python.exe -m src.cli.download_history `
    --symbol BTCUSDC `
    --interval 15m `
    --start 2024-01-01 `
    --end 2024-01-03
```

The default cache is:

```text
data/historical/BTCUSDC/15m/candles.csv
data/historical/BTCUSDC/15m/metadata.json
```

Existing data is validated before use. Only missing leading or trailing ranges
are downloaded, then the merged cache is sorted, conflict-checked, validated,
and persisted. Missing candles are never synthesized.

## Offline accounting benchmark

Download data first, then run:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_backtest `
    --symbol BTCUSDC `
    --interval 15m `
    --start 2024-01-01 `
    --end 2024-01-03
```

Reports are written under `reports/backtests/<deterministic_run_id>/`:

```text
trades.csv
equity.csv
summary.json
config.json
```

Metrics include initial/final capital, net return, trade counts, win rate, gross
profit/loss, average and largest wins/losses, payoff ratio, profit factor,
expectancy, maximum drawdown, consecutive wins/losses, total fees and market
exposure. Profit factor/payoff ratio are `null` when their denominator does not
exist rather than an invented large value.

## Offline V3 research

Download the complete research range first. Then specify all three contiguous
periods explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_research `
    --symbol BTCUSDC `
    --interval 15m `
    --development-start 2024-01-01 `
    --development-end 2024-07-01 `
    --validation-start 2024-07-01 `
    --validation-end 2024-10-01 `
    --oos-start 2024-10-01 `
    --oos-end 2025-01-01 `
    --cost-stress
```

The research command never downloads data or initializes a Binance client. If
cache coverage is missing, it fails with a download instruction. Reports live
at `reports/research/<deterministic_run_id>/` and contain the exact strategy
fingerprint, periods, data source, costs, execution assumptions, comparison,
stability diagnostics, signal journals, baseline results, and same-period
buy-and-hold results. `LOW SAMPLE SIZE` appears below the configured completed
trade threshold (30 by default); it is a warning, not a failure or quality score.

## Offline V3.1 diagnostics

Diagnose an existing research run by ID or directory:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_diagnostics `
    --research-run adeef00722e9704d
```

The command verifies each period's stored dataset SHA-256 against the local
cache before analysis. Outputs are written under
`reports/diagnostics/<deterministic_run_id>/` and include:

```text
diagnostics_summary.json
period_comparison.csv
development/trade_diagnostics.csv
development/exit_analysis.csv
development/feature_buckets.csv
development/regime_analysis.csv
development/holding_analysis.csv
development/time_analysis.csv
development/summary.json
validation/...
out_of_sample/...
combined_descriptive_only/...
```

Development, validation, and the former OOS period remain isolated in the
historical V3.1 artifact. From V3.2 onward they are all consumed research
segments. The combined directory is explicitly descriptive only. Every small
group receives an evidence-strength label; no p-values or claims of statistical
significance are fabricated.

## V3.2 pre-registered hypothesis research

V3.2 runs exactly five experiments. Each receives a clean simulated account,
the same candles, capital, fees, slippage, risk assumptions, next-bar-open
execution and conservative `STOP_FIRST` ambiguity policy.

- H0 — Frozen Baseline: exact `TrendMomentumBaselineStrategy` control.
- H1 — High Volatility Entry Guard: require current ATR14/close × 100 to be at
  or below the Q75 of the previous 100 valid values.
- H2 — EMA Extension Entry Guard: require current (EMA20−EMA50)/EMA50 × 100 to
  be at or below the Q75 of the previous 100 valid values.
- H3 — Crossover Pullback Confirmation: arm after a baseline-valid crossover
  and wait up to eight subsequently closed bars for a causal EMA20
  pullback/reclaim; a confirmation executes only at the next bar open.
- H4 — 1R Profit Target: keep the exact baseline entry and risk model, changing
  only the target from 2R to 1R.

H1 and H2 read their threshold before appending the current observation. If the
full prior 100-value window is unavailable, they do not enter. The fixed values
100, 75, 8 and 1.0R are pre-registered and alternative settings are refused.
There are no combined candidates, automatic thresholds, grids, sweeps,
optimization, machine learning, or automatic historical “winner” selection.

Results are classified conservatively as `SUPPORTED ON CONSUMED DATA`, `MIXED`,
`NOT SUPPORTED`, or `INSUFFICIENT SAMPLE`. Even the first label always remains
`NOT YET VERIFIED ON BLIND HOLDOUT`. “Improved,” “profitable,” and “robust” are
separate claims; V3.2 reports do not treat them as synonyms.

The audit trail is stored in `research/hypothesis_manifest.json`. The exact
inspected ranges are permanently marked consumed. An operator may explicitly
register one later, non-overlapping `BLIND_HOLDOUT`. Registration downloads
nothing and runs no strategy. Revealing requires `--confirm-reveal`; once the
results are viewed, status advances from `LOCKED_BLIND_HOLDOUT` through
`REVEALED` to `CONSUMED`. There is no reset to blind status because human
knowledge cannot be reset.

Run the suite on the existing consumed research run, optionally with the exact
2× fee/slippage stress replay:

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_hypotheses `
    --research-run adeef00722e9704d

.\.venv\Scripts\python.exe -m src.cli.run_hypotheses `
    --research-run adeef00722e9704d `
    --cost-stress
```

Registering a future range does not download or reveal it:

```powershell
.\.venv\Scripts\python.exe -m src.cli.register_holdout `
    --symbol BTCUSDC `
    --interval 15m `
    --start YYYY-MM-DD `
    --end YYYY-MM-DD
```

After registration, download that exact range deliberately if the local cache
does not already cover it. Only then may an operator irreversibly reveal it:

```powershell
.\.venv\Scripts\python.exe -m src.cli.download_history `
    --symbol BTCUSDC `
    --interval 15m `
    --start YYYY-MM-DD `
    --end YYYY-MM-DD

.\.venv\Scripts\python.exe -m src.cli.run_hypothesis_holdout `
    --manifest research/hypothesis_manifest.json `
    --confirm-reveal
```

The reveal command never downloads data automatically. Missing or incomplete
coverage stops with the exact required download command.

## V3.2.1 multi-regime research expansion

V3.2.1 evaluates the unchanged H0-H4 suite in fixed, non-overlapping 90-day
`[start,end)` windows. Every candidate/window replay starts with the same initial
USDC, zero BTC, no position, and identical execution costs. Prior candles from
the same safe research partition may initialize causal indicators, but cannot
generate counted trades or PnL. No window or warm-up crosses a partition gap.

The data lifecycle is explicit:

```text
RESEARCH_EXPANSION       actual BTCUSDC availability → 2025-08-01
LOCKED_BLIND_HOLDOUT     2025-08-01 → 2026-02-01 (not downloaded/evaluated)
CONSUMED_RESEARCH        2026-02-01 → 2026-08-18
```

Expansion data becomes `CONSUMED_RESEARCH_DATA` immediately before its first
strategy replay. The locked range remains separate from both historical cache
regions and may not contribute strategy input, warm-up, regime context, or
window statistics.

EMA200/EMA50 long-term trend labels and ATR14% low/medium/high labels are causal
analytics only. Volatility thresholds use the previous 200 valid ATR% values,
excluding the current observation. These labels never enter any H0-H4 strategy
decision. H1 remains the distinct, frozen prior-100 Q75 entry mechanism.

Stability reporting includes trade-weighted combined metrics, equal-window
consistency counts, window quartiles, adverse/best historical windows, regime
samples, and doubled-cost stress. A candidate receives
`MECHANISM_SUPPORTED_ON_RESEARCH_DATA` only when every preregistered relative
gate passes; `V3.3_ELIGIBLE` additionally requires positive combined
frictionless expectancy. Neither label is blind-holdout verification.

Download only the pre-holdout expansion into its gap-isolated V2 cache:

```powershell
Set-Location 'C:\Coding\TradingBot'

.\.venv\Scripts\python.exe -m src.cli.download_history `
    --symbol BTCUSDC `
    --interval 15m `
    --start 2023-01-01T00:00:00Z `
    --end 2025-08-01T00:00:00Z `
    --data-root data/historical/multiregime_expansion `
    --allow-source-gaps
```

The explicit gap flag keeps the default V2 cache strict. For this expansion it
records genuine Binance source discontinuities in metadata, synthesizes
nothing, and forces the research layer to start a new replay region after each
gap.

Run the offline experiment with the preregistered doubled-cost stress:

```powershell
Set-Location 'C:\Coding\TradingBot'

.\.venv\Scripts\python.exe -m src.cli.run_multiregime `
    --manifest research/hypothesis_manifest.json `
    --cost-stress
```

Reports are written to `reports/multiregime/<run_id>/` as
`window_comparison.csv`, `candidate_stability.csv`, `regime_comparison.csv`,
`summary.json`, `manifest.json`, and `RESEARCH_SUMMARY.md`. Check lifecycle
status without changing it:

```powershell
Set-Location 'C:\Coding\TradingBot'

.\.venv\Scripts\python.exe -m src.cli.show_research_status `
    --manifest research/hypothesis_manifest.json
```

Multi-regime historical consistency does not prove future profitability. The
locked blind holdout has not yet been evaluated. A candidate marked
`V3.3_ELIGIBLE` is only eligible for further controlled research.

## V3.2.2 new mechanism hypotheses

V3.2.2 reuses the exact V3.2.1 partitions, fixed chronological windows, H0
baseline, H1 Q75 guard, account model, and V2 simulated execution. It adds only
three preregistered candidates:

- H5 accepts baseline entries inside the causal prior-100 ATR% Q25-Q75 band.
- H6 delays a baseline-valid crossover for exactly one closed confirmation bar.
- H7 requires the theoretical 4×ATR target to be at least five times configured
  round-trip fee and slippage costs.

No candidates are combined, no parameters are searched, and the command has no
exchange or download path. The run requires `--cost-stress` so the same signals
are evaluated at frozen base costs and at exactly 2× fee/slippage. H7's signal
threshold is calculated from the frozen base-cost assumption; the stress replay
changes actual V2 execution costs, not the strategy definition.

```powershell
Set-Location 'C:\Coding\TradingBot'

.\.venv\Scripts\python.exe -m src.cli.run_mechanisms `
    --manifest research/hypothesis_manifest.json `
    --cost-stress
```

The immutable preregistration is stored under `research/mechanisms/<run_id>/`.
Reports are written under `reports/mechanisms/<run_id>/` and include combined
stability, window-level differences to H0, a combined comparison to frozen H1,
regime diagnostics, and the locked-holdout audit state. `NEXT_STAGE_ELIGIBLE`
means only that all fixed research-data gates passed with positive frictionless
expectancy; it is not validation or evidence of live profitability.

## Live read-only observer

```powershell
.\.venv\Scripts\python.exe run.py
```

This preserves the V1.1 public REST/history/indicator/WebSocket observer. Press
`Ctrl+C` to stop cleanly.

## Test

Ordinary tests are deterministic and require no network, credentials, current
price, or current date. Test artifacts stay in `.test_artifacts/` for portable
Windows execution.

```powershell
.\.venv\Scripts\python.exe -m pytest -ra
```

## Roadmap

- V0 — Foundation + live market data ✅
- V1 — Historical data + indicators ✅
- V1.1 — Real public market-data separation ✅
- V2 — Deterministic backtesting engine ✅
- V3 — Strategy research framework ✅
- V3.1 — Strategy diagnostics ✅
- V3.2 — Hypothesis-based strategy research ✅
- V3.2.1 — Multi-regime expansion ✅
- V3.2.2 — New mechanism hypotheses ✅ **CURRENT**
- V3.3 — Controlled parameter research
- V3.4 — Walk-forward robustness
- V4 — Binance Testnet execution
- V5 — Small-capital live execution
