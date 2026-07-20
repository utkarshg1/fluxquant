# fluxquant

High-performance quantitative finance library built for speed and memory safety.

Fluxquant treats market data as a continuous, streaming flow — emphasizing speed, native concurrency, and memory safety for Monte Carlo simulations, volatility modeling, and risk analytics.

## Features

- **GBM-GARCH Pipeline** — Geometric Brownian Motion with GARCH(p,q) volatility forecasting
- **Auto GARCH Order Selection** — grid search over `(p,q)` combinations optimized by BIC
- **Parallel Bootstrap** — rayon-powered Monte Carlo path simulation
- **Interactive HTML Dashboard** — self-contained output with Chart.js visualizations and percentile fan chart
- **Risk Analytics** — VaR, CVaR, Sharpe ratio, drawdown, skewness, kurtosis, t-distribution estimation, terminal price percentiles
- **Legacy API** — builder-pattern `SimulationEngine` for quick volatility fitting
- **YAML Configuration** — declarative simulation configs with CLI template generation
- **Config Management** — CLI subcommand to update defaults (ticker, confidence, bootstrap paths) interactively or via flags

## Architecture

```text
fluxquant/
├── Cargo.toml                  # Workspace root
├── LICENSE
└── crates/
    ├── fluxquant-core/         # Core engine (published as 'fluxquant' on crates.io)
    │   ├── Cargo.toml
    │   └── src/lib.rs
    └── fluxquant-cli/          # CLI toolkit (published as 'fluxquant-cli' on crates.io)
        ├── Cargo.toml
        └── src/main.rs
```

## Installation

```bash
cargo install fluxquant-cli
```

Or add the core library to your project:

```toml
[dependencies]
fluxquant = "1.9.0"
```

## Usage

### CLI

```bash
# First-time setup (interactive wizard)
fluxquant-cli init

# Generate a default simulation YAML template
fluxquant-cli gen

# Run a GBM-GARCH simulation (shows settings, asks for confirmation)
fluxquant-cli run --ticker AAPL --years 5

# Specify GARCH order and VaR tail level
fluxquant-cli run --ticker SPY --years 3 --garch-p 1 --garch-q 1 --var-level 0.01

# Use a YAML config file
fluxquant-cli run --config simulation.yaml

# Update default config interactively
fluxquant-cli config

# Quick update a single default
fluxquant-cli config --ticker SPY
```

The CLI prints a branded ASCII art banner on startup, shows a colored settings summary with confirmation prompt, displays results in clean `tabled`-powered Unicode box-drawing tables (Summary Statistics, Risk Metrics, Distribution Percentiles, Price Forecast) with proper column headers and no index columns, and outputs an interactive HTML dashboard with charts and statistics. Running without flags enters an interactive mode where each parameter is prompted individually. Use `config` to manage default settings.

### Library

```rust
use fluxquant::{SimulationConfig, GarchOrder, run_gbm_garch};

let config = SimulationConfig {
    forecast_weeks: 260,  // 5 years
    confidence_level: 0.95,
    garch_order: GarchOrder::Auto { max_p: 3, max_q: 3 },
    n_bootstrap: 10_000,
    seed: Some(42),
    var_level: 0.05,
};

let last_price = 150.0; // last known market price
let result = run_gbm_garch(&weekly_log_returns, &config, last_price)?;

println!("Mean annual return: {:+.2}%", result.summary.mean_annual_return * 100.0);
println!("Annual volatility:  {:.2}%", result.summary.annual_volatility * 100.0);
println!("Sharpe ratio:       {:.3}", result.summary.sharpe_ratio);
println!("Max drawdown:       {:.2}% (worst case)", result.summary.max_drawdown * 100.0);
println!("Median drawdown:    {:.2}%", result.summary.median_drawdown * 100.0);
println!("VaR  (5%):          {:+.2}%", result.summary.var * 100.0);
println!("CVaR (5%):          {:+.2}%", result.summary.cvar * 100.0);
println!("Target price:       {:.2}", result.price_median.last().unwrap());
println!(
    "Price percentiles: 2.5%: {:.2}, 50%: {:.2}, 97.5%: {:.2}",
    result.summary.price_percentiles[0],
    result.summary.price_percentiles[2],
    result.summary.price_percentiles[4]
);
```

### Legacy API

For quick volatility estimation without the full simulation pipeline:

```rust
use fluxquant::SimulationEngine;

let engine = SimulationEngine::builder().paths(1000).build();
let annualized_vol = engine.fit_volatility(&weekly_returns)?;
println!("Annualized volatility: {:.2}%", annualized_vol * 100.0);
```

## Dashboard

The HTML dashboard includes four interactive Chart.js visualizations:

| Chart | Description |
|-------|-------------|
| **Price Forecast** | Historical prices with median forecast and CI band |
| **Volatility Forecast** | GARCH volatility point estimates with CI band |
| **Bootstrap Fan Chart** | Percentile bands (2.5%, 25%, 50%, 75%, 97.5%) with shaded regions |
| **Terminal Price Distribution** | Histogram of final prices across all bootstrap paths |

Plus four data tables:
- **Summary Statistics** — mean return, annual volatility, Sharpe ratio, max/median drawdown, skewness, kurtosis, drift parameter, GARCH order, bootstrap paths
- **Risk Metrics** — VaR and CVaR at the configured tail level
- **Distribution Percentiles** — return, volatility, Sharpe ratio, and terminal price at 2.5%, 25%, 50%, 75%, 97.5%
- **Price Forecast** — target price, confidence interval, dashboard path

## Pipeline

Fluxquant uses a **GBM + GARCH** approach:

1. **Drift estimation** — compute the drift parameter μ from historical log-returns
2. **GARCH fitting** — fit a GARCH(p,q) model on log-returns (auto or manual order selection)
3. **Volatility forecast** — forecast conditional volatility over the simulation horizon
4. **GBM path simulation** — generate Monte Carlo paths using `S_{t+1} = S_t · exp((μ − σ²/2) · Δt + σ · ε)`
5. **Percentile and risk summary** — compute return/volatility/drawdown/price percentiles across all paths

## License

All Rights Reserved. Copyright (c) 2026 Utkarsh Gaikwad.
