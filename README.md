# fluxquant

High-performance quantitative finance library built for speed and memory safety.

Fluxquant treats market data as a continuous, streaming flow — emphasizing speed, native concurrency, and memory safety for Monte Carlo simulations, volatility modeling, and risk analytics.

## Architecture

```text
fluxquant/
├── Cargo.toml                  # Workspace root
├── templates/
│   └── simulation.yaml         # Default configuration blueprint
└── crates/
    ├── fluxquant-core/         # Core engine (published as 'fluxquant' on crates.io)
    │   ├── Cargo.toml
    │   └── src/lib.rs
    └── fluxquant-cli/          # CLI toolkit
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
fluxquant = "0.1.0"
```

## Usage

### CLI

```bash
# Generate a default simulation template
fluxquant gen

# Run a simulation from a YAML config
fluxquant run --config simulation.yaml
```

### Library

```rust
use fluxquant::SimulationEngine;

let engine = SimulationEngine::builder()
    .paths(5000)
    .build();

// Run Monte Carlo simulation
engine.run_monte_carlo().unwrap();

// Fit GARCH(1,1) volatility
let returns = vec![0.01, -0.02, 0.015, -0.005, 0.008];
let annualized_vol = engine.fit_volatility(&returns).unwrap();
```

## License

All Rights Reserved. Copyright (c) 2026 Utkarsh Gaikwad.
