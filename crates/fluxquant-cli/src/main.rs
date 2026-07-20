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
use owo_colors::OwoColorize;
use std::cmp::max;
use std::io::{self, Write};
use std::path::PathBuf;

use fluxquant::{GarchOrder, SimulationConfig, generate_dashboard, run_gbm_garch};

const DEFAULT_TEMPLATE: &str = r#"# fluxquant GBM-GARCH simulation configuration
simulation:
  ticker: "AAPL"
  forecast_years: 5
  history_years: 3
  confidence_level: 0.95
  n_bootstrap: 10000
  var_level: 0.05

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
    var_level: Option<f64>,
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
        /// VaR/CVaR tail probability (default: 0.05).
        var_level: Option<f64>,
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

fn print_corrected_cyan_banner() {
    let ascii_art = vec![
        "██████╗ ██╗     ██╗   ██╗██╗  ██╗ ██████╗  ██╗   ██╗ █████╗ ███╗   ██╗████████╗",
        "██╔═══╝ ██║     ██║   ██║╚██╗██╔╝██╔═══██╗ ██║   ██║██╔══██╗████╗  ██║╚══██╔══╝",
        "█████╗  ██║     ██║   ██║ ╚███╔╝ ██║   ██║ ██║   ██║███████║██╔██╗ ██║   ██║   ",
        "██╔══╝  ██║     ██║   ██║ ██╔██╗ ██║▄▄ ██║ ██║   ██║██╔══██║██║╚██╗██║   ██║   ",
        "██║     ███████╗╚██████╔╝██╔╝ ██╗╚██████╔╝ ╚██████╔╝██║  ██║██║ ╚████║   ██║   ",
        "╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚════▀▀   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ",
    ];

    let subtitle = "Created by Utkarsh Gaikwad";

    let banner_width = ascii_art
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0);
    let max_width = max(banner_width, subtitle.chars().count());

    let top_border = format!("╔{}╗", "═".repeat(max_width + 4));
    let divider = format!("╠{}╣", "═".repeat(max_width + 4));
    let bot_border = format!("╚{}╝", "═".repeat(max_width + 4));

    println!("{}", top_border.cyan().bold());

    for line in &ascii_art {
        let current_len = line.chars().count();
        let padding = max_width - current_len;
        println!(
            "{}  {}{}  {}",
            "║".cyan().bold(),
            line.cyan().bold(),
            " ".repeat(padding),
            "║".cyan().bold()
        );
    }

    println!("{}", divider.cyan().bold());

    let sub_len = subtitle.chars().count();
    let sub_padding = max_width - sub_len;
    println!(
        "{}  {}{}  {}",
        "║".cyan().bold(),
        subtitle.cyan().bold(),
        " ".repeat(sub_padding),
        "║".cyan().bold()
    );

    println!("{}", bot_border.cyan().bold());
    println!();
}

fn print_settings_box(
    ticker: &str,
    fy: u32,
    hy: u32,
    conf: f64,
    nboot: usize,
    garch_order: &GarchOrder,
    var_level: f64,
) {
    let garch_str = match garch_order {
        GarchOrder::Auto { max_p, max_q } => format!("Auto (max {max_p},{max_q})"),
        GarchOrder::Manual { p, q } => format!("Fixed ({p},{q})"),
    };
    let level_pct = (conf * 100.0).round() as usize;
    let var_pct = (var_level * 100.0).round() as usize;
    let w = 40;

    println!("{}", format!("┌{}┐", "─".repeat(w)).cyan().bold());
    println!(
        "{} {}{} {}",
        "║".cyan().bold(),
        "Simulation Settings".cyan().bold(),
        " ".repeat(w - 21),
        "║".cyan().bold()
    );
    println!("{}", format!("├{}┤", "─".repeat(w)).cyan().bold());

    let rows = [
        ("Ticker", ticker.to_string()),
        ("Forecast", format!("{fy} years")),
        ("History", format!("{hy} years")),
        ("Confidence", format!("{level_pct}%")),
        ("Bootstrap", format!("{nboot} paths")),
        ("GARCH", garch_str),
        ("VaR Level", format!("{var_pct}%")),
    ];

    for (label, value) in &rows {
        let label_w = 14;
        let val_w = w - label_w - 3;
        let padding = val_w.saturating_sub(value.len());
        println!(
            "{}  {:<label_w$}{}{}{}",
            "║".cyan().bold(),
            label.white().bold(),
            value,
            " ".repeat(padding),
            "║".cyan().bold()
        );
    }

    println!("{}", format!("└{}┘", "─".repeat(w)).cyan().bold());
}

#[tokio::main]
async fn main() -> Result<()> {
    print_corrected_cyan_banner();

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
            var_level: var_level_flag,
            output,
            config,
        } => {
            let flux = load_flux().unwrap_or_else(|| {
                println!("No config. Run init.");
                std::process::exit(1)
            });

            let mut fy = years.unwrap_or(5);
            let mut hy = history_years.unwrap_or(3);
            let mut conf = confidence.unwrap_or(0.95);
            let mut nboot = paths.unwrap_or(10000);
            let mut var_level = var_level_flag.unwrap_or(0.05);

            let mut garch_order = GarchOrder::Auto { max_p: 3, max_q: 3 };
            let mut sim_ticker = ticker.clone().unwrap_or(flux.default_ticker.clone());

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
                if let Some(ref s) = parsed.simulation {
                    if let Some(ref t) = s.ticker {
                        sim_ticker = t.clone();
                    }
                    if let Some(v) = s.var_level {
                        var_level = v;
                    }
                }
            }

            if let Some(p) = garch_p {
                garch_order = GarchOrder::Manual {
                    p,
                    q: garch_q.unwrap_or(1),
                };
            }

            // Show settings and confirm
            print_settings_box(&sim_ticker, fy, hy, conf, nboot, &garch_order, var_level);
            print!("Run with these settings? [Y/n]: ");
            io::stdout().flush()?;
            let mut confirm = String::new();
            io::stdin().read_line(&mut confirm)?;

            if confirm.trim().eq_ignore_ascii_case("n") {
                // Interactive mode
                println!();

                print!("Ticker [{}]: ", sim_ticker);
                io::stdout().flush()?;
                let input = prompt(&sim_ticker)?;
                sim_ticker = input;

                print!("Forecast years [{fy}]: ");
                io::stdout().flush()?;
                let input = prompt(&fy.to_string())?;
                if let Ok(v) = input.parse::<u32>() {
                    fy = v;
                }

                print!("History years [{hy}]: ");
                io::stdout().flush()?;
                let input = prompt(&hy.to_string())?;
                if let Ok(v) = input.parse::<u32>() {
                    hy = v;
                }

                print!("Confidence level [{conf}]: ");
                io::stdout().flush()?;
                let input = prompt(&conf.to_string())?;
                if let Ok(v) = input.parse::<f64>() {
                    conf = v;
                }

                print!("Bootstrap paths [{nboot}]: ");
                io::stdout().flush()?;
                let input = prompt(&nboot.to_string())?;
                if let Ok(v) = input.parse::<usize>() {
                    nboot = v;
                }

                let default_garch = match &garch_order {
                    GarchOrder::Auto { .. } => "auto",
                    GarchOrder::Manual { .. } => "fixed",
                };
                print!("GARCH mode (auto/fixed) [{default_garch}]: ");
                io::stdout().flush()?;
                let input = prompt(default_garch)?;
                if input.eq_ignore_ascii_case("fixed") {
                    print!("GARCH p [1]: ");
                    io::stdout().flush()?;
                    let input = prompt("1")?;
                    let p: usize = input.parse().unwrap_or(1);
                    print!("GARCH q [1]: ");
                    io::stdout().flush()?;
                    let input = prompt("1")?;
                    let q: usize = input.parse().unwrap_or(1);
                    garch_order = GarchOrder::Manual { p, q };
                } else {
                    print!("Max GARCH p [3]: ");
                    io::stdout().flush()?;
                    let input = prompt("3")?;
                    let max_p: usize = input.parse().unwrap_or(3);
                    print!("Max GARCH q [3]: ");
                    io::stdout().flush()?;
                    let input = prompt("3")?;
                    let max_q: usize = input.parse().unwrap_or(3);
                    garch_order = GarchOrder::Auto { max_p, max_q };
                }

                print!("VaR level [{var_level}]: ");
                io::stdout().flush()?;
                let input = prompt(&var_level.to_string())?;
                if let Ok(v) = input.parse::<f64>() {
                    var_level = v;
                }

                println!();
                print_settings_box(&sim_ticker, fy, hy, conf, nboot, &garch_order, var_level);
                print!("Confirm? [Y/n]: ");
                io::stdout().flush()?;
                let mut final_confirm = String::new();
                io::stdin().read_line(&mut final_confirm)?;
                if final_confirm.trim().eq_ignore_ascii_case("n") {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            println!();
            println!("{}", "Fetching data...".cyan());

            let client = yfinance_rs::YfClient::default();
            let yf = yfinance_rs::Ticker::new(&client, &sim_ticker);
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
            println!(
                "  {} weekly returns fetched",
                wret.len().to_string().green()
            );

            let cfg = SimulationConfig {
                forecast_weeks: fy as usize * 52,
                confidence_level: conf,
                garch_order,
                n_bootstrap: nboot,
                seed: Some(42),
                var_level,
            };
            let last_price = cp.last().copied().unwrap_or(1.0);
            let res = tokio::task::spawn_blocking(move || run_gbm_garch(&wret, &cfg, last_price))
                .await??;

            let level_pct = (conf * 100.0).round() as usize;
            let var_level_pct = (var_level * 100.0).round() as usize;

            println!();
            println!("{}", "── Simulation Results ──".cyan().bold());
            println!();

            let ret = res.summary.mean_annual_return * 100.0;
            let ret_str = format!("{ret:+.2}%");
            if ret >= 0.0 {
                println!("  Mean ann return:  {}", ret_str.green());
            } else {
                println!("  Mean ann return:  {}", ret_str.red());
            }

            println!(
                "  Ann volatility:   {}",
                format!("{:.2}%", res.summary.annual_volatility * 100.0).yellow()
            );

            let sharpe = res.summary.sharpe_ratio;
            let sharpe_str = format!("{sharpe:.3}");
            if sharpe > 1.0 {
                println!("  Sharpe:           {}", sharpe_str.green());
            } else if sharpe >= 0.0 {
                println!("  Sharpe:           {}", sharpe_str.yellow());
            } else {
                println!("  Sharpe:           {}", sharpe_str.red());
            }

            println!(
                "  Max drawdown:     {}",
                format!("{:.2}%", res.summary.max_drawdown * 100.0).red()
            );
            println!("  Drift (mu):       {:+.4}", res.mu_drift);
            println!(
                "  GARCH:            ({},{})",
                res.garch_order_selected.0, res.garch_order_selected.1
            );
            println!("  Bootstrap paths:  {nboot}");

            println!();
            println!(
                "  VaR  ({var_level_pct}%):     {}",
                format!("{:.2}%", res.summary.var * 100.0).red()
            );
            println!(
                "  CVaR ({var_level_pct}%):     {}",
                format!("{:.2}%", res.summary.cvar * 100.0).red()
            );

            let target = *res.price_median.last().unwrap_or(&0.0);
            let ci_lo = *res.price_lower.last().unwrap_or(&0.0);
            let ci_hi = *res.price_upper.last().unwrap_or(&0.0);

            println!();
            println!(
                "  Target price:     {} ({level_pct}% CI: {} to {})",
                format!("{target:.2}").white().bold(),
                format!("{ci_lo:.2}").white().bold(),
                format!("{ci_hi:.2}").white().bold()
            );

            let out = output.unwrap_or_else(|| flux.workspace.join("results/dashboard.html"));
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, generate_dashboard(&sim_ticker, &res, &cp)?)?;
            println!();
            println!(
                "  Dashboard:        {}",
                format!("{}", out.display()).green()
            );
        }
    }
    Ok(())
}
