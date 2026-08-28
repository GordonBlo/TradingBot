# TradingBot

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Market](https://img.shields.io/badge/Market-BTCUSDC%20Spot-F0B90B?logo=binance&logoColor=black)
![Mode](https://img.shields.io/badge/Execution-Shadow%20Only-success)
![Tests](https://img.shields.io/badge/Tests-700%2B-brightgreen)

Research-grade crypto trading platform focused on **causal backtesting, reproducible quantitative research, market microstructure and safe live-market execution infrastructure**.

**Primary market:** BTCUSDC Spot  
**Timeframe:** 15m  
**Direction:** Long-only

> **Current status:** live public-market observation and shadow execution only.  
> Real-money order execution is disabled. No validated profitable trading edge is claimed.

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
- live Level-2 order-book collection
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
                 Prospective Validation
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
| Multi-timeframe continuation | Positive frictionless expectancy, negative after costs |
| Spot aggTrades | Signal exhausted / not supported |
| Derivatives context | Independent validation failed; branch closed |
| **V9 Spot L2 microstructure** | **Prospective research in progress** |

Failed hypotheses remain part of the project history instead of being hidden or post-hoc optimized.

---

## V9 — Prospective L2 Research

The current research branch tests whether BTCUSDC Spot order-book microstructure contains stable information about future price movement.

Its protocol was frozen **before eligible prospective data was evaluated**.

The experiment uses:

- predefined L2 feature set
- 1-second causal sampling
- +30 second primary forward mid-price target
- chronological session-blocked out-of-sample evaluation
- Ridge regression
- constant baseline comparison
- permutation testing
- predefined stability gates

Predictive evaluation is programmatically refused until the preregistered data-readiness and integrity gates pass.

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
python -m pytest -ra
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
- complete Git history scanned with Gitleaks

Latest repository secret scan:

```text
70 commits scanned
No leaks found
```

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

Current project status: **700+ automated tests**.

---

## Roadmap

```text
Prospective L2 Data
        ↓
Information Diagnostic
        ↓
Independent Validation
        ↓
Candidate Strategy
        ↓
Long-running Shadow / Paper Validation
        ↓
Execution + Risk Engine
        ↓
Controlled Test Execution
        ↓
Small-capital live validation only if justified
```

Real-money execution will not be enabled solely because a historical backtest performs well.

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
