# TradingBot

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Market](https://img.shields.io/badge/Market-BTCUSDC%20Spot-F0B90B?logo=binance&logoColor=black)
![Mode](https://img.shields.io/badge/Execution-Shadow%20Only-success)
![Tests](https://img.shields.io/badge/Tests-1124%20passed-brightgreen)

Research-grade crypto trading platform focused on **causal backtesting, reproducible quantitative research, market microstructure and safe live-market execution infrastructure**.

**Primary market:** BTCUSDC Spot  
**Timeframe:** 15m  
**Direction:** Long-only

> **Current status:** V9 information discovery completed; V10 acquisition is READY.
> The frozen V10 economic discovery executor is implemented, **NOT AUTHORIZED and NOT executed**.
> Public-market observation and shadow infrastructure exist; no V10 strategy has passed paper/shadow or live validation. Real-money orders remain disabled. No validated profitable strategy is claimed.

---

## Overview

TradingBot is built to answer a harder question than _“can I create a profitable backtest?”_:

**Can a trading signal survive realistic costs, causal validation, independent testing and prospective market data?**

The project deliberately separates data collection, research, simulation and live runtime components so failed hypotheses can be rejected without contaminating future validation.

---

## Core Engineering

### Deterministic Research & Backtesting

- causal, chronological market replay
- next-bar-open execution
- realistic fees and adverse slippage
- conservative `STOP_FIRST` ambiguous-bar handling
- `Decimal`-safe financial accounting
- deterministic manifests, run IDs and reports
- preregistered hypotheses and locked blind holdout
- independent validation and explicit consumed-data tracking

### Binance Market Data

- public BTCUSDC Spot REST and WebSocket data
- historical OHLCV
- Spot aggregate trades
- historical derivatives-context research
- live Level-2 order-book and synchronized aggregate-trade collection
- no credentials required for public-data pipelines

### Level-2 Order Book

- REST snapshot + 100ms WebSocket diff-depth synchronization
- sequence-aware local book reconstruction
- gap detection, reconnect and resynchronization
- append-only raw event persistence
- deterministic offline replay
- spread, depth, imbalance, microprice, concentration and liquidity-flow features

### Live Shadow Runtime

- closed-candle-only strategy evaluation
- runtime states: `STARTING`, `SYNCING`, `READY`, `STALE`, `RECOVERING`, `STOPPED`
- stale-data and gap protection
- duplicate-candle suppression
- restart-safe persistent state
- hash-chained decision journal
- deterministic recovery
- **zero order execution**

---

## Architecture

```text
                 Binance PUBLIC Market Data
                           │
            ┌──────────────┼──────────────┐
            │              │              │
          OHLCV        aggTrades       Spot L2
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                 Validation / Persistence
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     Deterministic Backtesting     L2 Reconstruction
              │                         │
              ▼                         ▼
       Research Framework       Microstructure Features
              │                         │
              └────────────┬────────────┘
                           ▼
                 Preregistered Discovery
                           │
                           ▼
          Future Strategy Validation / NEW Data
                           │
                           ▼
                    Shadow Runtime
                           │
                           ▼
                    ORDERS DISABLED
```

---

## Research Status

| Research branch | Result |
|---|---|
| Breakout | Not supported after realistic costs |
| Mean reversion | Not supported |
| V6 multi-timeframe continuation | Frictionless signal insufficient after realistic costs |
| V7 Spot aggTrades | NOT_SUPPORTED / exhausted |
| V8 derivatives context | Independent validation failed; CLOSED |
| V9 Spot L2 information discovery | Completed: STABLE_L2_INFORMATION; evidence CONSUMED; standalone short-horizon taker economics insufficient |
| V10 execution research foundation | Cost-aware and depth-aware offline simulation; public L2 + aggTrade collector implemented |
| V10 acquisition | READY: 8 eligible sessions / 24 hours / 4 UTC start dates |
| V10 economic discovery | Frozen protocol; executor implemented; NOT AUTHORIZED / NOT executed |

Failed hypotheses remain part of the project history instead of being hidden or post-hoc optimized.

---

## V9 — Completed Prospective L2 Information Discovery

The prospective BTCUSDC Spot order-book information experiment completed with classification **STABLE_L2_INFORMATION**. Its data is **CONSUMED research evidence**, and cannot be reused as fresh independent strategy validation. Standalone short-horizon taker economics were insufficient to justify a trading strategy.

Its protocol was frozen **before eligible prospective data was evaluated**.

The experiment used:

- predefined L2 feature set
- 1-second causal sampling
- +30 second primary forward mid-price target
- chronological session-blocked out-of-sample evaluation
- Ridge regression
- constant baseline comparison
- permutation testing
- predefined stability gates

This information classification does not establish trading profitability.

## V10 — Execution Foundation and Frozen Economic Discovery

The [depth-aware execution foundation](docs/v10_l2_execution.md) provides causal offline depth sweeps, explicit fill policies, exchange constraints and Decimal cost accounting. The [public collector](docs/v10_public_microstructure_collector.md) persists synchronized Spot L2 and aggTrades without authentication or orders.

Acquisition preregistration `8326791b411c27b5` has a READY dataset of **8 eligible closed sessions, 24 hours and 4 UTC start dates**. Economic discovery preregistration `a296e5ed304640d7` binds exactly those sessions. Its definition SHA-256 is `a296e5ed304640d706916cc8e562cedc6febf561b20b0c8a99d59ac30b275a33`.

The [economic discovery protocol](docs/v10_microstructure_economic_discovery.md) is frozen and its [executor](docs/v10_economic_executor.md) is implemented. Real discovery remains **NOT AUTHORIZED and NOT executed**; no authorization, reservation or result exists. The next step is pre-execution verification followed by one separately, explicitly authorized discovery run. A reservation permanently prevents a second outcome run, including after interruption.

The protocol compares frozen L2-only and combined L2/flow models and quote-price economic magnitude. Even `ECONOMIC_SIGNAL_PRESENT` would be **discovery only**, not a profitable strategy or V10 strategy validation. It does not establish sized depth capacity, actual fills, impact, inventory/risk management or deployability; depth slippage is `NOT_ESTIMATED`. Discovery consumes all eight sessions. Any later confirmatory strategy experiment requires a new preregistration and genuinely new prospective data.

Run the read-only audit from `C:\Coding\TradingBot` using **Python 3.12 exactly**:

```powershell
.\.venv\Scripts\python.exe -B -m src.cli.audit_v10_preexecution
```

The audit emits JSON and exits 0 for PASS or 1 for FAIL. It checks the expected repository/branch, a clean Git worktree, frozen manifests and source hashes, all bound artifacts, fresh acquisition replay/readiness, absent authorization/reservation, executor/runtime identity and free disk space. A dirty worktree always fails, while allowing read-only integrity checks to complete; there is no bypass flag. It never constructs predictive targets, fits models, runs permutations, places orders or consumes the dataset. It does not read the blind holdout. Fresh acquisition replay uses the existing scanner with at most four workers and can take substantial time across 24 hours of data; the JSON result is emitted only after completion.

The operational disk minimum is **10 GiB free** for at most 86,400 sample rows, 40 features, four horizons, 1,000 null statistics and report overhead. This conservative reserve is separate from frozen research rules and does not guarantee future storage availability. The report records actual free bytes before and after verification. JSON may be redirected to an ignored `reports/audits/` path, never the real discovery output directory. A PASS is a point-in-time pre-authorization check, not permission to execute; rerun immediately before any future authorization.

Local audit on **2026-09-22**: fresh acquisition **READY (8 sessions / 24 hours / 4 UTC dates)**; frozen manifests, artifact/reference hashes, executor/runtime identity and final integrity checks passed. Free disk space after verification was **373.28 GiB**. Overall result: **FAIL solely because the worktree contains uncommitted changes**. Authorization and discovery reservation/result remain absent. The ignored diagnostic report is `reports/audits/v10_preexecution/audit.json`.

---

## Safety Boundaries

```text
Real orders        DISABLED
Withdrawals        DISABLED
Margin             DISABLED
Leverage           DISABLED
Short selling      DISABLED
Live credentials   NOT REQUIRED
Blind holdout      LOCKED
```

Research and public-market components are intentionally isolated from authenticated execution.

---

## Quick Start

### Windows PowerShell

```powershell
git clone https://github.com/GordonBlo/TradingBot.git
cd TradingBot

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
$run = ".pytest_runtime\run_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
New-Item -ItemType Directory -Force $run | Out-Null
python -m pytest -ra --basetemp="$run\basetemp" -p no:cacheprovider
```

Run a bounded public-market shadow session:

```powershell
python -m src.cli.run_shadow_runtime --smoke-seconds 60
```

The shadow runtime observes real public BTCUSDC market data while remaining physically separated from order execution.

---

## Project Structure

```text
src/
├── analysis/       # indicators and causal analytics
├── backtest/       # deterministic execution simulation
├── diagnostics/    # post-trade and signal diagnostics
├── derivatives/    # historical derivatives context
├── exchange/       # Binance public market-data boundary
├── historical/     # validated historical datasets
├── orderflow/      # Spot aggregate-trade research
├── orderbook/      # L2 reconstruction and features
├── microstructure/ # synchronized public L2 + aggTrade acquisition/replay
├── research/       # research and validation framework
├── runtime/        # live shadow state machine
├── strategy/       # strategy implementations
└── risk/           # risk foundations

research/            # frozen preregistration manifests
tests/               # deterministic automated test suite
```

---

## Research Principles

**No look-ahead. No silent parameter optimization. No future-data leakage.**

**No rewriting failed hypotheses. No pretending consumed data is fresh OOS.**

**No ignoring transaction costs. No live orders before validated evidence.**

The objective is not to manufacture an attractive equity curve — it is to determine whether an observable market edge actually survives rigorous testing.

---

## Security

Sensitive credentials and generated research artifacts are intentionally excluded from version control.

Repository safeguards include:

- `.env` excluded from Git
- API keys and secrets excluded
- private keys excluded
- raw market datasets excluded
- generated reports excluded
- public Binance market-data clients operate without authentication
- full Git history scanning with locally installed Gitleaks

Fresh local history scan on **2026-09-22**, at commit `ea4b96ccdee74539e37602ecf32408cf9058e7c4`:

```text
Gitleaks 8.30.1
git --log-opts="--all --full-history" --redact=100
89 commits scanned (also confirmed by git rev-list --all --count)
No leaks found; exit code 0
```

Redacted generated evidence is saved under ignored `reports/audits/v10_preexecution/` (`gitleaks.json` and `gitleaks.log`). No upload or Git history modification was performed. This scan covers committed history at the stated revision; it is not a guarantee about future changes.

---

## Testing

The project contains an extensive deterministic automated test suite covering:

- execution semantics
- causal strategy evaluation
- historical dataset integrity
- order-book reconstruction
- sequence-gap handling
- deterministic replay
- L2 feature generation
- research preregistration
- session eligibility
- live runtime state transitions
- stale-data protection
- duplicate suppression
- restart recovery
- shadow logging
- zero-order guarantees
- V10 acquisition, frozen economic-discovery binding and synthetic executor checks
- read-only pre-execution readiness and refusal gates

Full suite on **2026-09-22**, Python **3.12.14**: **1,124 passed** in 186.64 seconds. The 26 focused pre-execution audit tests passed first. Both runs used fresh Windows basetemp directories and `-p no:cacheprovider`; the sole warning is the existing `cache_dir` configuration being unknown while that provider is disabled. The generated full-suite log is ignored at `reports/audits/v10_preexecution/full_pytest.log`.

---

## Roadmap

```text
V9 Information Discovery COMPLETED / CONSUMED
        ↓
V10 Acquisition READY / Economic Protocol FROZEN
        ↓
Pre-execution Verification
        ↓
One Explicitly Authorized DISCOVERY Run (NOT AUTHORIZED)
        ↓
Future Preregistered Strategy + NEW Prospective Validation
        ↓
Paper / Shadow Strategy Validation
        ↓
Execution + Risk Engine
        ↓
Controlled Test Execution
        ↓
Small-capital live validation only if justified
```

Real-money execution will not be enabled solely because a historical backtest performs well.
Existing shadow runtime infrastructure does not imply a validated V10 strategy or completed paper trading. Every future stage depends on sufficient evidence and separate authorization for live execution.

---

## Research History

Earlier research iterations include:

- deterministic baseline strategy research
- volatility and trend-regime diagnostics
- breakout research
- mean-reversion research
- multi-timeframe continuation
- Spot aggregate-trade order-flow research
- derivatives-context discovery
- independent historical validation
- prospective Spot L2 microstructure research

Several branches were explicitly closed after failing predefined validation gates.

This is intentional.

TradingBot is designed to reject unsupported hypotheses rather than repeatedly optimize them until a profitable backtest appears.

---

## Disclaimer

TradingBot is a software-engineering and quantitative-research project.

It is **not financial advice**, does not guarantee profitability and currently does not execute real-money trades.

Cryptocurrency trading involves substantial financial risk.
