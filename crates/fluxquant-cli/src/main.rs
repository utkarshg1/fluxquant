use anyhow::Result;
use clap::{Parser, Subcommand};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use fluxquant::{GarchOrder, SarimaOrder, SimulationConfig, generate_dashboard, run_sarima_garch};

const BANNER: &str = r#"
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  ███████╗██╗     ██╗   ██╗██╗  ██╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗      ║
║  ██╔════╝██║     ██║   ██║╚██╗██╔╝██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝      ║
║  █████╗  ██║     ██║   ██║ ╚███╔╝ ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║         ║
║  ██╔══╝  ██║     ██║   ██║ ██╔██╗ ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║         ║
║  ██║     ███████╗╚██████╔╝██╔╝ ██╗╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║         ║
║  ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝         ║
║                                                                                      ║
║                           [ fluxquant by Utkarsh Gaikwad ]                           ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"#;

const DEFAULT_TEMPLATE: &str = r#"# fluxquant SARIMA-GARCH simulation configuration
simulation:
  ticker: "AAPL"
  forecast_years: 5
  history_years: 3
  confidence_level: 0.95
  n_bootstrap: 10000

sarima:
  mode: "auto"
  seasonal_period: 52
  # manual orders (ignored when mode=auto):
  # p: 1
  # d: 1
  # q: 1
  # P: 1
  # D: 1
  # Q: 1

garch:
  mode: "optimize"
  max_p: 3
  max_q: 3
  # when mode=fixed:
  # p: 1
  # q: 1

output:
  save_dashboard: true
  dashboard_path: "./results/dashboard.html"
"#;

// ── Persistent config (stored at OS-native config path) ─────────────
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

/// OS-native config path: ~/.config/fluxquant/ (Linux), AppData\Roaming\fluxquant\ (Win), etc.
fn flux_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("fluxquant")
        .join("config.yaml")
}

fn load_flux_config() -> Option<FluxConfig> {
    let path = flux_config_path();
    if !path.exists() {
        return None;
    }
    let contents = std::fs::read_to_string(&path).ok()?;
    serde_yaml::from_str(&contents).ok()
}

fn save_flux_config(cfg: &FluxConfig) -> Result<()> {
    let path = flux_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let yaml = serde_yaml::to_string(cfg)?;
    std::fs::write(&path, yaml)?;
    Ok(())
}

/// Create workspace subdirectories and copy the default simulation template.
fn ensure_workspace(cfg: &FluxConfig) -> Result<()> {
    let configs_dir = cfg.workspace.join("configs");
    let results_dir = cfg.workspace.join("results");

    println!(
        "\n  {} Creating workspace at {}...",
        style("→").cyan(),
        style(cfg.workspace.display()).cyan().bold()
    );

    std::fs::create_dir_all(&configs_dir)?;
    println!("    {}/", style("configs").green());

    std::fs::create_dir_all(&results_dir)?;
    println!("    {}/", style("results").green());

    let template_path = configs_dir.join("simulation.yaml");
    if !template_path.exists() {
        std::fs::write(&template_path, DEFAULT_TEMPLATE)?;
        println!("    {}", style("simulation.yaml").green());
    }

    Ok(())
}

/// Read a line of input from stdin, returning trimmed string.
fn prompt_input(default: &str) -> Result<String> {
    io::stdout().flush()?;
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    let trimmed = buf.trim().to_string();
    Ok(if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed
    })
}

/// Interactive setup wizard — prompts user and returns a new FluxConfig.
fn setup_wizard() -> Result<FluxConfig> {
    println!("\n{}", style("fluxquant first-time setup").cyan().bold());
    println!("{}", style("═".repeat(50)).dim());
    println!("\n  {} Workspace folder path", style("1/4").cyan().bold());
    println!(
        "    {}",
        style("Where dashboards and configs will be stored.").dim()
    );

    let default_home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let default_workspace = default_home.join("fluxquant-workspace");
    print!("    [{}]: ", style(default_workspace.display()).dim());
    let input = prompt_input(&default_workspace.to_string_lossy())?;
    let workspace = PathBuf::from(&input);

    // ── Ticker ──
    println!("\n  {} Default ticker symbol", style("2/4").cyan().bold());
    print!("    [{}]: ", style("AAPL").dim());
    let input = prompt_input("AAPL")?;
    let ticker = input.to_uppercase();

    // ── Confidence ──
    println!(
        "\n  {} Default confidence level",
        style("3/4").cyan().bold()
    );
    print!("    [{}]: ", style("0.95").dim());
    let input = prompt_input("0.95")?;
    let confidence: f64 = input.parse().unwrap_or(0.95);

    // ── Bootstrap paths ──
    println!("\n  {} Default bootstrap paths", style("4/4").cyan().bold());
    print!("    [{}]: ", style("10000").dim());
    let input = prompt_input("10000")?;
    let bootstrap: usize = input.parse().unwrap_or(10000);

    let cfg = FluxConfig {
        workspace,
        default_ticker: ticker,
        default_confidence: confidence,
        default_bootstrap_paths: bootstrap,
    };

    // Create workspace dirs
    ensure_workspace(&cfg)?;

    // Save config
    save_flux_config(&cfg)?;
    println!(
        "\n  {} Config saved to {}",
        style("✓").green().bold(),
        style(flux_config_path().display()).cyan().underlined()
    );
    println!(
        "\n  {} Setup complete! Run {} to start.\n",
        style("✓").green().bold(),
        style("fluxquant run").cyan().bold()
    );

    Ok(cfg)
}

#[derive(Parser)]
#[command(name = "fluxquant")]
#[command(
    about = "High-performance quantitative finance CLI with SARIMA-GARCH simulation",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run SARIMA-GARCH simulation with bootstrap confidence intervals
    Run {
        /// Ticker symbol (e.g. AAPL, MSFT, TSLA)
        #[arg(short, long)]
        ticker: Option<String>,

        /// Forecast horizon in years
        #[arg(short = 'y', long, default_value = "5")]
        years: u32,

        /// Years of historical data to fetch
        #[arg(short = 'H', long, default_value = "3")]
        history_years: u32,

        /// Confidence level (e.g. 0.95 for 95%)
        #[arg(short, long, default_value = "0.95")]
        confidence: f64,

        /// Number of bootstrap paths
        #[arg(short = 'n', long, default_value = "10000")]
        paths: usize,

        /// Use AutoARIMA (default) or manual SARIMA orders
        #[arg(long, default_value = "auto")]
        sarima: String,

        /// GARCH mode: optimize (grid search + BIC) or fixed
        #[arg(long, default_value = "optimize")]
        garch: String,

        /// Max GARCH p for grid search (when garch=optimize)
        #[arg(long, default_value = "3")]
        garch_max_p: usize,

        /// Max GARCH q for grid search (when garch=optimize)
        #[arg(long, default_value = "3")]
        garch_max_q: usize,

        /// Fixed GARCH p (when garch=fixed)
        #[arg(long)]
        garch_p: Option<usize>,

        /// Fixed GARCH q (when garch=fixed)
        #[arg(long)]
        garch_q: Option<usize>,

        /// Seasonal period for SARIMA (default 52 for weekly)
        #[arg(long, default_value = "52")]
        seasonal_period: usize,

        /// Output path for HTML dashboard (defaults to {workspace}/results/dashboard.html)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// YAML config file (overrides CLI args when present)
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Generate a default simulation YAML template
    Gen {
        #[arg(short, long, default_value = "simulation.yaml")]
        output: PathBuf,
    },

    /// Set up workspace folder and default configuration
    Init,
}

#[derive(serde::Deserialize)]
struct YamlConfig {
    simulation: Option<YamlSimulation>,
    sarima: Option<YamlSarima>,
    garch: Option<YamlGarch>,
}

#[derive(serde::Deserialize)]
struct YamlSimulation {
    ticker: Option<String>,
    forecast_years: Option<u32>,
    history_years: Option<u32>,
    confidence_level: Option<f64>,
    n_bootstrap: Option<usize>,
}

#[derive(serde::Deserialize)]
#[allow(non_snake_case)]
struct YamlSarima {
    mode: Option<String>,
    seasonal_period: Option<usize>,
    p: Option<usize>,
    d: Option<usize>,
    q: Option<usize>,
    P: Option<usize>,
    D: Option<usize>,
    Q: Option<usize>,
}

#[derive(serde::Deserialize)]
struct YamlGarch {
    mode: Option<String>,
    max_p: Option<usize>,
    max_q: Option<usize>,
    p: Option<usize>,
    q: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("{}", style(BANNER).cyan().bold());
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            ticker,
            years,
            history_years,
            confidence,
            paths,
            sarima,
            garch,
            garch_max_p,
            garch_max_q,
            garch_p,
            garch_q,
            seasonal_period,
            output,
            config,
        } => {
            // Load or create persistent config
            let flux_cfg = load_flux_config().unwrap_or_else(|| {
                println!(
                    "\n  {} No configuration found. Starting setup wizard...",
                    style("!").yellow().bold()
                );
                setup_wizard().expect("Setup wizard failed")
            });

            // Load YAML config if provided (overrides CLI args)
            let yaml = if let Some(config_path) = &config {
                let contents = std::fs::read_to_string(config_path)?;
                Some(serde_yaml::from_str::<YamlConfig>(&contents)?)
            } else {
                None
            };

            let ticker = yaml
                .as_ref()
                .and_then(|y| y.simulation.as_ref())
                .and_then(|s| s.ticker.clone())
                .or(ticker)
                .unwrap_or_else(|| flux_cfg.default_ticker.clone());

            let forecast_years = yaml
                .as_ref()
                .and_then(|y| y.simulation.as_ref())
                .and_then(|s| s.forecast_years)
                .unwrap_or(years);

            let hist_years = yaml
                .as_ref()
                .and_then(|y| y.simulation.as_ref())
                .and_then(|s| s.history_years)
                .unwrap_or(history_years);

            let conf_level = yaml
                .as_ref()
                .and_then(|y| y.simulation.as_ref())
                .and_then(|s| s.confidence_level)
                .unwrap_or(confidence);

            let n_bootstrap = yaml
                .as_ref()
                .and_then(|y| y.simulation.as_ref())
                .and_then(|s| s.n_bootstrap)
                .unwrap_or(paths);

            let s_period = yaml
                .as_ref()
                .and_then(|y| y.sarima.as_ref())
                .and_then(|s| s.seasonal_period)
                .unwrap_or(seasonal_period);

            let sarima_order = match yaml
                .as_ref()
                .and_then(|y| y.sarima.as_ref())
                .and_then(|s| s.mode.as_deref())
                .unwrap_or(&sarima)
            {
                "manual" => {
                    let s = yaml.as_ref().and_then(|y| y.sarima.as_ref()).unwrap();
                    SarimaOrder::Manual {
                        p: s.p.unwrap_or(1),
                        d: s.d.unwrap_or(1),
                        q: s.q.unwrap_or(1),
                        P: s.P.unwrap_or(1),
                        D: s.D.unwrap_or(1),
                        Q: s.Q.unwrap_or(1),
                        s: s_period,
                    }
                }
                _ => SarimaOrder::Auto {
                    seasonal_period: s_period,
                },
            };

            let garch_order = match yaml
                .as_ref()
                .and_then(|y| y.garch.as_ref())
                .and_then(|s| s.mode.as_deref())
                .unwrap_or(&garch)
            {
                "fixed" => {
                    let ym = yaml.as_ref().and_then(|y| y.garch.as_ref());
                    let p = ym.and_then(|s| s.p).or(garch_p).unwrap_or(1);
                    let q = ym.and_then(|s| s.q).or(garch_q).unwrap_or(1);
                    GarchOrder::Manual { p, q }
                }
                _ => {
                    let ym = yaml.as_ref().and_then(|y| y.garch.as_ref());
                    GarchOrder::Auto {
                        max_p: ym.and_then(|s| s.max_p).unwrap_or(garch_max_p),
                        max_q: ym.and_then(|s| s.max_q).unwrap_or(garch_max_q),
                    }
                }
            };

            let config = SimulationConfig {
                forecast_weeks: forecast_years as usize * 52,
                confidence_level: conf_level,
                sarima_order,
                garch_order,
                n_bootstrap,
                seed: Some(42),
            };

            // Resolve output path: CLI > workspace default
            let output =
                output.unwrap_or_else(|| flux_cfg.workspace.join("results").join("dashboard.html"));

            // ── Fetch historical data via yfinance-rs ─────────────────────
            println!(
                "\n{}",
                style(format!(
                    "Fetching {} weekly data ({} years)...",
                    ticker, hist_years
                ))
                .bold()
            );

            let spinner = ProgressBar::new_spinner();
            spinner.set_style(
                ProgressStyle::default_spinner()
                    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                    .template("{spinner:.green} {msg}")?,
            );
            spinner.set_message("Connecting to Yahoo Finance...");
            spinner.enable_steady_tick(std::time::Duration::from_millis(80));

            let client = yfinance_rs::YfClient::default();
            let yf_ticker = yfinance_rs::Ticker::new(&client, &ticker);

            let range = match hist_years {
                1 => yfinance_rs::Range::Y1,
                2 => yfinance_rs::Range::Y2,
                3..=4 => yfinance_rs::Range::Y5,
                5 => yfinance_rs::Range::Y5,
                _ => yfinance_rs::Range::Y5,
            };

            let history = yf_ticker
                .history(Some(range), Some(yfinance_rs::Interval::W1), false)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to fetch data: {e}"))?;

            spinner.finish_with_message(
                style(format!("Fetched {} weekly bars", history.len()))
                    .green()
                    .to_string(),
            );

            // Extract closing prices and compute log returns
            let closing_prices: Vec<f64> = history
                .iter()
                .filter_map(|bar| {
                    let s = bar.ohlc.close.to_string();
                    s.parse::<f64>().ok()
                })
                .collect();

            if closing_prices.len() < 20 {
                anyhow::bail!(
                    "Insufficient data: got {} closing prices, need at least 20",
                    closing_prices.len()
                );
            }

            let weekly_returns: Vec<f64> = closing_prices
                .windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();

            println!(
                "  {} weekly returns computed from {} price observations",
                style(weekly_returns.len()).cyan().bold(),
                closing_prices.len()
            );

            // ── Run SARIMA-GARCH simulation ───────────────────────────────
            println!("\n{}", style("Running SARIMA-GARCH simulation...").bold());

            let pb = ProgressBar::new(config.n_bootstrap as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} paths ({eta})",
                    )?
                    .progress_chars("#>-"),
            );

            let forecast_weeks = config.forecast_weeks;
            let n_bootstrap = config.n_bootstrap;

            // Run simulation (bootstrap progress is internal via rayon)
            let result =
                tokio::task::spawn_blocking(move || run_sarima_garch(&weekly_returns, &config))
                    .await??;

            pb.set_position(n_bootstrap as u64);
            pb.finish_with_message(style("Simulation complete!").green().bold().to_string());

            // ── Display Results ───────────────────────────────────────────
            println!("\n{}", style("═".repeat(60)).dim());
            println!(
                "{} {} — SARIMA-GARCH Forecast ({} weeks)",
                style("📊").bold(),
                style(&ticker).cyan().bold(),
                forecast_weeks
            );
            println!("{}", style("═".repeat(60)).dim());

            println!(
                "\n  {} Returns Forecast ({}% CI):",
                style("→").green(),
                (conf_level * 100.0) as usize
            );
            let n = result.returns_forecast.point.len();
            for &w in &[52usize, 104, 156, 208, 260] {
                if w <= n {
                    let i = w - 1;
                    let pt = result.returns_forecast.point[i] * 100.0;
                    let lo = result.returns_forecast.lower[i] * 100.0;
                    let hi = result.returns_forecast.upper[i] * 100.0;
                    println!(
                        "    Week {:3}: {:>6.2}%  [ {:>6.2}%, {:>6.2}% ]",
                        w, pt, lo, hi
                    );
                }
            }

            println!(
                "\n  {} Volatility Forecast ({}% CI):",
                style("→").magenta(),
                (conf_level * 100.0) as usize
            );
            for &w in &[52usize, 104, 156, 208, 260] {
                if w <= n {
                    let i = w - 1;
                    let pt = result.volatility_forecast.point[i] * 100.0;
                    let lo = result.volatility_forecast.lower[i] * 100.0;
                    let hi = result.volatility_forecast.upper[i] * 100.0;
                    println!(
                        "    Week {:3}: {:>6.2}%  [ {:>6.2}%, {:>6.2}% ]",
                        w, pt, lo, hi
                    );
                }
            }

            println!("\n  {} Summary Statistics:", style("→").yellow());
            println!(
                "    Mean Annual Return:  {}{}%{}",
                if result.summary.mean_annual_return >= 0.0 {
                    "+"
                } else {
                    ""
                },
                style(format!("{:.2}", result.summary.mean_annual_return * 100.0)).green(),
                ""
            );
            println!(
                "    Annual Volatility:   {}%",
                style(format!("{:.2}", result.summary.annual_volatility * 100.0)).cyan()
            );
            println!(
                "    Sharpe Ratio:        {}",
                style(format!("{:.3}", result.summary.sharpe_ratio)).cyan()
            );
            println!(
                "    Max Drawdown:        {}%",
                style(format!("{:.2}", result.summary.max_drawdown * 100.0)).red()
            );
            println!(
                "    Skewness:            {}",
                style(format!("{:.4}", result.summary.skewness)).dim()
            );
            println!(
                "    Excess Kurtosis:     {}",
                style(format!("{:.4}", result.summary.kurtosis)).dim()
            );
            println!(
                "    t-df Estimate:       {}",
                style(format!("{:.2}", result.summary.t_df_estimate)).dim()
            );
            println!(
                "    SARIMA Order:        {}",
                style(&result.sarima_order_selected).cyan()
            );
            println!(
                "    GARCH Order:         ({},{})",
                result.garch_order_selected.0, result.garch_order_selected.1
            );
            println!("    Bootstrap Paths:     {}", style(n_bootstrap).cyan());

            // ── Generate HTML Dashboard ───────────────────────────────────
            let html = generate_dashboard(&ticker, &result, &closing_prices)?;
            std::fs::create_dir_all(output.parent().unwrap_or(Path::new(".")))?;
            std::fs::write(&output, &html)?;

            println!(
                "\n  {} Dashboard saved to {}",
                style("✓").green().bold(),
                style(output.display()).cyan().underlined()
            );
            println!();
        }

        Commands::Gen { output } => {
            // Load or create persistent config
            let flux_cfg = load_flux_config().unwrap_or_else(|| {
                println!(
                    "\n  {} No configuration found. Starting setup wizard...",
                    style("!").yellow().bold()
                );
                setup_wizard().expect("Setup wizard failed")
            });

            let output = output
                .to_str()
                .map(|s| s == "simulation.yaml")
                .unwrap_or(false)
                .then(|| flux_cfg.workspace.join("configs").join("simulation.yaml"))
                .unwrap_or(output);

            let spinner = ProgressBar::new_spinner();
            spinner
                .set_style(ProgressStyle::default_spinner().template("{spinner:.yellow} {msg}")?);
            spinner.set_message(format!("Generating template at {}...", output.display()));

            std::thread::sleep(std::time::Duration::from_millis(400));

            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, DEFAULT_TEMPLATE)?;
            spinner.finish_with_message(
                style("Template generated successfully!")
                    .green()
                    .to_string(),
            );
        }

        Commands::Init => {
            let existing = load_flux_config();
            if existing.is_some() {
                println!(
                    "\n  {} Existing configuration found at {}",
                    style("!").yellow().bold(),
                    style(flux_config_path().display()).cyan().underlined()
                );
                print!("    {} Overwrite? [y/N]: ", style("Continue?").dim());
                io::stdout().flush()?;
                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm)?;
                if !confirm.trim().eq_ignore_ascii_case("y") {
                    println!("    {}", style("Aborted.").dim());
                    return Ok(());
                }
            }
            setup_wizard()?;
        }
    }

    Ok(())
}
