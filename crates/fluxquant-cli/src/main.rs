//! # fluxquant-cli
//!
//! Command-line interface for the fluxquant GBM-GARCH quantitative finance engine.
//!
//! Provides three subcommands:
//!
//! - **`init`** — Interactive workspace setup wizard
//! - **`gen`** — Generate a default YAML simulation template
//! - **`run`** — Fetch market data and run a full GBM-GARCH simulation
//!
//! ## Quick Start
//!
//! ```bash
//! fluxquant-cli init                    # First-time setup
//! fluxquant-cli run --ticker AAPL       # Run simulation with defaults
//! fluxquant-cli run --ticker SPY --years 3 --garch-p 1 --garch-q 1
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;

use fluxquant::{GarchOrder, SimulationConfig, generate_dashboard, run_gbm_garch};

const FLUXQUANT_BANNER: &str = r#"
╔═══════════════════════════════════════════════════════════════╗
║  ███████╗██╗████████╗██╗  ██╗     ██████╗ ███████╗███╗   ██╗║
║  ██╔════╝██║╚══██╔══╝██║  ██║    ██╔═══██╗██╔════╝████╗  ██║║
║  █████╗  ██║   ██║   ███████║    ██║   ██║███████╗██╔██╗ ██║║
║  ██╔══╝  ██║   ██║   ██╔══██║    ██║   ██║╚════██║██║╚██╗██║║
║  ██║     ██║   ██║   ██║  ██║    ╚██████╔╝███████║██║ ╚████║║
║  ╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝     ╚═════╝ ╚══════╝╚═╝  ╚═══╝║
║                                                               ║
║            GBM-GARCH Simulation Engine                        ║
╚═══════════════════════════════════════════════════════════════╝
"#;

const DEFAULT_TEMPLATE: &str = r#"# fluxquant GBM-GARCH simulation configuration
simulation:
  ticker: "AAPL"
  forecast_years: 5
  history_years: 3
  confidence_level: 0.95
  n_bootstrap: 10000

garch:
  mode: "optimize"
  max_p: 3
  max_q: 3

output:
  save_dashboard: true
  dashboard_path: "./results/dashboard.html"
"#;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct FluxConfig {
    workspace: PathBuf,
    default_ticker: String,
    default_confidence: f64,
    default_bootstrap_paths: usize,
}

impl Default for FluxConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self {
            workspace: home.join("fluxquant-workspace"),
            default_ticker: "AAPL".to_string(),
            default_confidence: 0.95,
            default_bootstrap_paths: 10000,
        }
    }
}

fn flux_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_default()
        .join("fluxquant")
        .join("config.yaml")
}

fn load_flux() -> Option<FluxConfig> {
    let p = flux_config_path();
    if !p.exists() {
        return None;
    }
    serde_yaml::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

fn save_flux(cfg: &FluxConfig) -> Result<()> {
    let p = flux_config_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&p, serde_yaml::to_string(cfg)?)?;
    Ok(())
}

fn prompt(default: &str) -> Result<String> {
    io::stdout().flush()?;
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    Ok(if buf.trim().is_empty() {
        default.into()
    } else {
        buf.trim().into()
    })
}

fn setup() -> Result<FluxConfig> {
    let dw = dirs::home_dir()
        .unwrap_or_default()
        .join("fluxquant-workspace");
    print!("Workspace [{}]: ", dw.display());
    let w = PathBuf::from(prompt(&dw.to_string_lossy())?);
    let cfg = FluxConfig {
        workspace: w,
        ..Default::default()
    };
    for d in [
        &cfg.workspace.join("configs"),
        &cfg.workspace.join("results"),
    ] {
        std::fs::create_dir_all(d)?;
    }
    save_flux(&cfg)?;
    println!("Config saved");
    Ok(cfg)
}

#[derive(serde::Deserialize)]
struct YSim {
    ticker: Option<String>,
}

#[derive(serde::Deserialize)]
struct YGarch {
    mode: Option<String>,
    max_p: Option<usize>,
    max_q: Option<usize>,
    p: Option<usize>,
    q: Option<usize>,
}

#[derive(serde::Deserialize)]
struct YConfig {
    simulation: Option<YSim>,
    garch: Option<YGarch>,
}

/// GBM-GARCH quantitative finance simulation engine.
///
/// Run Monte Carlo simulations with GARCH volatility modeling on real market data
/// fetched from Yahoo Finance. Outputs an interactive HTML dashboard with charts,
/// summary statistics, and risk analytics.
#[derive(Parser)]
#[command(
    name = "fluxquant-cli",
    about = "GBM-GARCH quantitative finance simulation engine",
    long_about = "FluxQuant CLI — Monte Carlo simulation with GARCH volatility modeling.\n\n\
        Fetches historical market data from Yahoo Finance, fits a GARCH(p,q) model \
        for volatility estimation, and runs GBM bootstrap simulations to produce \
        price forecasts, fan charts, and risk analytics.\n\n\
        Outputs a self-contained HTML dashboard with interactive Chart.js visualizations.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a GBM-GARCH simulation on a ticker.
    Run {
        /// Ticker symbol (e.g. AAPL, SPY, BTC-USD).
        ticker: Option<String>,
        /// Forecast horizon in years (default: 5).
        years: Option<u32>,
        /// Years of historical data to fetch (default: 3).
        history_years: Option<u32>,
        /// Confidence level for intervals (default: 0.95).
        confidence: Option<f64>,
        /// Number of bootstrap paths (default: 10000).
        paths: Option<usize>,
        /// Fixed GARCH p order (disables auto selection).
        garch_p: Option<usize>,
        /// Fixed GARCH q order (requires --garch-p).
        garch_q: Option<usize>,
        /// Output path for the HTML dashboard.
        output: Option<PathBuf>,
        /// Path to a YAML configuration file.
        config: Option<PathBuf>,
    },
    /// Generate a default YAML simulation template.
    Gen {
        /// Output path for the template file.
        output: Option<PathBuf>,
    },
    /// Interactive workspace setup wizard.
    Init,
}

#[tokio::main]
async fn main() -> Result<()> {
    print!("{FLUXQUANT_BANNER}");

    let cli = Cli::parse();
    match cli.command {
        Commands::Init => {
            if load_flux().is_some() {
                print!("Overwrite? [y/N]: ");
                io::stdout().flush()?;
                let mut c = String::new();
                io::stdin().read_line(&mut c)?;
                if !c.trim().eq_ignore_ascii_case("y") {
                    return Ok(());
                }
            }
            setup()?;
        }
        Commands::Gen { output } => {
            let f = load_flux().unwrap_or_default();
            let p = output.unwrap_or_else(|| f.workspace.join("configs/simulation.yaml"));
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&p, DEFAULT_TEMPLATE)?;
            println!("Template: {}", p.display());
        }
        Commands::Run {
            ticker,
            years,
            history_years,
            confidence,
            paths,
            garch_p,
            garch_q,
            output,
            config,
        } => {
            let flux = load_flux().unwrap_or_else(|| {
                println!("No config. Run init.");
                std::process::exit(1)
            });

            let fy = years.unwrap_or(5);
            let hy = history_years.unwrap_or(3);
            let conf = confidence.unwrap_or(0.95);
            let nboot = paths.unwrap_or(10000);

            let mut garch_order = GarchOrder::Auto { max_p: 3, max_q: 3 };
            let mut ticker = ticker.unwrap_or(flux.default_ticker.clone());

            if let Some(ref cp) = config {
                let raw = std::fs::read_to_string(cp)?;
                let parsed: YConfig = serde_yaml::from_str(&raw)?;
                if let Some(ref g) = parsed.garch {
                    garch_order = match g.mode.as_deref() {
                        Some("fixed") => GarchOrder::Manual {
                            p: g.p.unwrap_or(1),
                            q: g.q.unwrap_or(1),
                        },
                        _ => GarchOrder::Auto {
                            max_p: g.max_p.unwrap_or(3),
                            max_q: g.max_q.unwrap_or(3),
                        },
                    };
                }
                if let Some(ref s) = parsed.simulation
                    && let Some(ref t) = s.ticker
                {
                    ticker = t.clone();
                }
            }

            if let Some(p) = garch_p {
                garch_order = GarchOrder::Manual {
                    p,
                    q: garch_q.unwrap_or(1),
                };
            }

            println!("Fetching {ticker}...");

            let client = yfinance_rs::YfClient::default();
            let yf = yfinance_rs::Ticker::new(&client, &ticker);
            let range = match hy {
                1 => yfinance_rs::Range::Y1,
                2 => yfinance_rs::Range::Y2,
                _ => yfinance_rs::Range::Y5,
            };
            let history = yf
                .history(Some(range), Some(yfinance_rs::Interval::W1), false)
                .await
                .map_err(|e| anyhow::anyhow!("Fetch failed: {e}"))?;

            let cp: Vec<f64> = history
                .iter()
                .filter_map(|b| b.ohlc.close.to_string().parse::<f64>().ok())
                .collect();
            if cp.len() < 20 {
                anyhow::bail!("Need >=20 prices, got {}", cp.len());
            }
            let wret: Vec<f64> = cp.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
            println!("  {} returns", wret.len());

            let cfg = SimulationConfig {
                forecast_weeks: fy as usize * 52,
                confidence_level: conf,
                garch_order,
                n_bootstrap: nboot,
                seed: Some(42),
            };
            let res = tokio::task::spawn_blocking(move || run_gbm_garch(&wret, &cfg)).await??;

            println!("\nResults:");
            println!(
                "  Mean ann return: {:+.2}%",
                res.summary.mean_annual_return * 100.0
            );
            println!(
                "  Ann volatility:  {:.2}%",
                res.summary.annual_volatility * 100.0
            );
            println!("  Sharpe:          {:.3}", res.summary.sharpe_ratio);
            println!(
                "  Max drawdown:    {:.2}%",
                res.summary.max_drawdown * 100.0
            );
            println!("  Drift (mu):      {:+.4}", res.mu_drift);
            println!(
                "  GARCH:           ({},{})",
                res.garch_order_selected.0, res.garch_order_selected.1
            );
            println!("  Paths:           {}", nboot);

            let out = output.unwrap_or_else(|| flux.workspace.join("results/dashboard.html"));
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, generate_dashboard(&ticker, &res, &cp)?)?;
            println!("Dashboard: {}", out.display());
        }
    }
    Ok(())
}
