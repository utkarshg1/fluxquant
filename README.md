# fluxquant

High-performance quantitative finance library built for speed and memory safety.

Fluxquant treats market data as a continuous, streaming flow — emphasizing speed, native concurrency, and memory safety for Monte Carlo simulations, volatility modeling, and risk analytics.

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
fluxquant = "1.2.0"
```

## Usage

### CLI

```bash
# First-time setup (interactive wizard)
fluxquant-cli init

# Generate a default simulation YAML template
fluxquant-cli gen

# Run a SARIMA-GARCH simulation
fluxquant-cli run --ticker AAPL --years 5 --paths 10000

# Run from a YAML config file
fluxquant-cli run --config simulation.yaml
```

### Library

```rust
use fluxquant::{SimulationConfig, SarimaOrder, GarchOrder, run_sarima_garch};

let config = SimulationConfig {
    forecast_weeks: 260,  // 5 years
    confidence_level: 0.95,
    sarima_order: SarimaOrder::Auto { seasonal_period: 52 },
    garch_order: GarchOrder::Auto { max_p: 3, max_q: 3 },
    n_bootstrap: 10_000,
    seed: Some(42),
};

let result = run_sarima_garch(&weekly_log_returns, &config)?;

println!("Mean annual return: {:.2}%", result.summary.mean_annual_return * 100.0);
println!("Annual volatility:  {:.2}%", result.summary.annual_volatility * 100.0);
println!("Sharpe ratio:       {:.3}", result.summary.sharpe_ratio);
println!("Max drawdown:       {:.2}% (worst case)", result.summary.max_drawdown * 100.0);
println!("Median drawdown:    {:.2}%", result.summary.median_drawdown * 100.0);
println!(
    "Return percentiles: 2.5%: {:+.2}%, 50%: {:+.2}%, 97.5%: {:+.2}%",
    result.summary.return_percentiles[0] * 100.0,
    result.summary.return_percentiles[2] * 100.0,
    result.summary.return_percentiles[4] * 100.0
);
println!(
    "Volatility percentiles: 2.5%: {:.2}%, 50%: {:.2}%, 97.5%: {:.2}%",
    result.summary.volatility_percentiles[0] * 100.0,
    result.summary.volatility_percentiles[2] * 100.0,
    result.summary.volatility_percentiles[4] * 100.0
);
```

## License

All Rights Reserved. Copyright (c) 2026 Utkarsh Gaikwad.
