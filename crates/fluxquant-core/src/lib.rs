//! # Fluxquant
//!
//! High-performance quantitative finance library built for speed and memory safety.
//!
//! Fluxquant treats market data as a continuous, streaming flow — emphasizing
//! speed, native concurrency, and memory safety for Monte Carlo simulations,
//! volatility modeling, and risk analytics.
//!
//! ## Features
//!
//! - **GBM-GARCH Pipeline** — Geometric Brownian Motion with GARCH volatility forecasting
//! - **Auto GARCH Order Selection** — grid search over `(p,q)` combinations with BIC optimization
//! - **Parallel Bootstrap** — rayon-powered Monte Carlo path simulation
//! - **Interactive Dashboard** — self-contained HTML output with Chart.js visualizations
//! - **Risk Analytics** — Sharpe ratio, drawdown, skewness, kurtosis, t-distribution estimation
//!
//! ## Quick Start
//!
//! ```rust
//! use fluxquant::{SimulationConfig, GarchOrder, run_gbm_garch};
//!
//! let config = SimulationConfig {
//!     forecast_weeks: 260,  // 5 years
//!     confidence_level: 0.95,
//!     garch_order: GarchOrder::Auto { max_p: 3, max_q: 3 },
//!     n_bootstrap: 10_000,
//!     seed: Some(42),
//!     var_level: 0.05,
//! };
//!
//! // With real weekly log-returns:
//! // let result = run_gbm_garch(&weekly_log_returns, &config).unwrap();
//! // println!("Mean annual return: {:+.2}%", result.summary.mean_annual_return * 100.0);
//! ```
//!
//! ## Pipeline
//!
//! 1. **Drift estimation** — compute `μ` from historical log-returns
//! 2. **GARCH fitting** — fit GARCH(p,q) on returns (auto or manual order)
//! 3. **Volatility forecast** — forecast conditional variance over the horizon
//! 4. **GBM path simulation** — `S_{t+1} = S_t · exp((μ − σ²/2) + σ · ε)`
//! 5. **Percentile & risk summary** — return/vol/drawdown distributions across all paths

use rayon::prelude::*;
use thiserror::Error;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during simulation, fitting, or dashboard generation.
#[derive(Error, Debug)]
pub enum FluxError {
    /// The simulation pipeline failed (e.g. insufficient data).
    #[error("Simulation failed: {0}")]
    SimulationError(String),

    /// Volatility model fitting failed.
    #[error("Volatility fitting error: {0}")]
    VolatilityError(String),

    /// GARCH model fitting or forecasting failed.
    #[error("GARCH fitting error: {0}")]
    GARCHError(String),

    /// HTML dashboard generation failed.
    #[error("Dashboard generation error: {0}")]
    DashboardError(String),
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// GARCH order specification.
#[derive(Debug, Clone)]
pub enum GarchOrder {
    /// Grid search over (p,q) combinations, select best by BIC.
    Auto { max_p: usize, max_q: usize },
    /// Fixed GARCH(p,q) order.
    Manual { p: usize, q: usize },
}

/// Configuration for the GBM-GARCH simulation.
///
/// # Example
///
/// ```rust
/// use fluxquant::{SimulationConfig, GarchOrder};
///
/// let config = SimulationConfig {
///     forecast_weeks: 260,
///     confidence_level: 0.95,
///     garch_order: GarchOrder::Auto { max_p: 3, max_q: 3 },
///     n_bootstrap: 10_000,
///     seed: Some(42),
///     var_level: 0.05,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of weeks to forecast.
    pub forecast_weeks: usize,
    /// Confidence level for intervals (e.g. 0.95 for 95%).
    pub confidence_level: f64,
    /// GARCH order specification.
    pub garch_order: GarchOrder,
    /// Number of bootstrap paths for simulation.
    pub n_bootstrap: usize,
    /// Random seed for reproducibility (None = random).
    pub seed: Option<u64>,
    /// VaR/CVaR tail probability level (e.g. 0.05 for 5%).
    pub var_level: f64,
}

// ─── Results ──────────────────────────────────────────────────────────────────

/// Summary statistics for the simulation.
#[derive(Debug, Clone)]
pub struct SummaryStats {
    /// Mean annualized return.
    pub mean_annual_return: f64,
    /// Annualized volatility.
    pub annual_volatility: f64,
    /// Sharpe ratio (assuming 0 risk-free rate).
    pub sharpe_ratio: f64,
    /// Worst-case max drawdown across all bootstrap paths.
    pub max_drawdown: f64,
    /// Median max drawdown across bootstrap paths.
    pub median_drawdown: f64,
    /// Return distribution skewness.
    pub skewness: f64,
    /// Return distribution excess kurtosis.
    pub kurtosis: f64,
    /// Value at Risk at var_level (negative = loss).
    pub var: f64,
    /// Conditional Value at Risk (Expected Shortfall) at var_level.
    pub cvar: f64,
    /// VaR/CVaR tail probability level used.
    pub var_level: f64,
    /// Estimated t-distribution degrees of freedom.
    pub t_df_estimate: f64,
    /// Annual return percentiles: [2.5%, 25%, 50%, 75%, 97.5%]
    pub return_percentiles: [f64; 5],
    /// Annual volatility percentiles: [2.5%, 25%, 50%, 75%, 97.5%]
    pub volatility_percentiles: [f64; 5],
    /// Sharpe ratio percentiles: [2.5%, 25%, 50%, 75%, 97.5%]
    pub sharpe_percentiles: [f64; 5],
    /// Terminal price percentiles (actual $): [2.5%, 25%, 50%, 75%, 97.5%]
    pub price_percentiles: [f64; 5],
}

/// Full result of a GBM-GARCH simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Median price path (scaled to actual prices).
    pub price_median: Vec<f64>,
    /// Lower confidence bound for price path.
    pub price_lower: Vec<f64>,
    /// Upper confidence bound for price path.
    pub price_upper: Vec<f64>,
    /// GARCH volatility forecast point estimates.
    pub vol_forecast: Vec<f64>,
    /// GARCH volatility forecast lower CI.
    pub vol_lower: Vec<f64>,
    /// GARCH volatility forecast upper CI.
    pub vol_upper: Vec<f64>,
    /// Complete bootstrap price paths (scaled to actual prices).
    pub bootstrap_paths: Vec<Vec<f64>>,
    /// Summary statistics.
    pub summary: SummaryStats,
    /// GARCH order selected (p, q).
    pub garch_order_selected: (usize, usize),
    /// Estimated drift (mean log-return).
    pub mu_drift: f64,
    /// Last known historical price used to scale forecast.
    pub last_price: f64,
    /// Confidence level used for intervals.
    pub confidence_level: f64,
}

// ─── GARCH Optimization ───────────────────────────────────────────────────────

/// Compute Gaussian log-likelihood for a GARCH model.
fn garch_log_likelihood(returns: &[f64], conditional_var: &[f64]) -> f64 {
    let n = returns.len().min(conditional_var.len());
    let mut ll = 0.0;
    for i in 0..n {
        let v = conditional_var[i];
        if v > 0.0 && v.is_finite() {
            ll -= 0.5 * (v.ln() + returns[i] * returns[i] / v);
        }
    }
    ll
}

/// Fit a GARCH(p,q) model and return its BIC.
fn fit_garch_and_bic(
    returns: &[f64],
    p: usize,
    q: usize,
) -> Result<(anofox_forecast::models::garch::GARCH, f64), FluxError> {
    use anofox_forecast::core::TimeSeries;
    use anofox_forecast::models::Forecaster;
    use anofox_forecast::models::garch::GARCH;
    use chrono::{TimeZone, Utc};

    let timestamps: Vec<_> = (0..returns.len())
        .map(|i| {
            Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i as i64)
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, returns.to_vec()).map_err(|e| {
        FluxError::GARCHError(format!(
            "Failed to create TimeSeries for GARCH({p},{q}): {e}"
        ))
    })?;

    let mut model = GARCH::builder().p(p).q(q).build();
    model
        .fit(&ts)
        .map_err(|e| FluxError::GARCHError(format!("GARCH({p},{q}) fit failed: {e}")))?;

    let cond_var = model.conditional_variance().ok_or_else(|| {
        FluxError::GARCHError("No conditional variance available after fit".into())
    })?;

    let ll = garch_log_likelihood(returns, cond_var);
    let n = returns.len() as f64;
    let k = (p + q + 1) as f64;
    let bic = k * n.ln() - 2.0 * ll;

    Ok((model, bic))
}

/// Grid search over GARCH(p,q) orders, selecting the best by BIC.
///
/// Tests all combinations of `p` in `1..=max_p` and `q` in `1..=max_q`,
/// fits each model, and returns the one with the lowest BIC.
///
/// # Errors
///
/// Returns [`FluxError::GARCHError`] if fewer than 12 observations are provided
/// or if all model fits fail.
pub fn optimize_garch(
    returns: &[f64],
    max_p: usize,
    max_q: usize,
) -> Result<(anofox_forecast::models::garch::GARCH, usize, usize), FluxError> {
    if returns.len() < 12 {
        return Err(FluxError::GARCHError(
            "Need at least 12 observations for GARCH fitting".into(),
        ));
    }

    let mut best_bic = f64::INFINITY;
    let mut best_model = None;
    let mut best_p = 1;
    let mut best_q = 1;

    for p in 1..=max_p {
        for q in 1..=max_q {
            match fit_garch_and_bic(returns, p, q) {
                Ok((model, bic)) if bic < best_bic => {
                    best_bic = bic;
                    best_model = Some(model);
                    best_p = p;
                    best_q = q;
                }
                Ok(_) => {}
                Err(_) => continue,
            }
        }
    }

    let model = best_model.ok_or_else(|| {
        FluxError::GARCHError(format!(
            "All GARCH models failed for p=1..{max_p}, q=1..{max_q}"
        ))
    })?;

    Ok((model, best_p, best_q))
}

// ─── GBM-GARCH Pipeline ───────────────────────────────────────────────────────

/// Run the full GBM-GARCH simulation pipeline.
///
/// 1. Estimates drift `μ` from historical log-returns
/// 2. Fits GARCH(p,q) on returns (auto or manual order selection)
/// 3. Forecasts conditional volatility with confidence bands
/// 4. Generates Monte Carlo paths via GBM bootstrap
/// 5. Computes percentile bands and summary risk statistics
///
/// `weekly_returns` should be log-returns: `ln(P_t / P_{t-1})`.
///
/// # Errors
///
/// Returns [`FluxError::SimulationError`] if fewer than 20 observations are provided.
/// Returns [`FluxError::GARCHError`] if GARCH fitting or variance forecasting fails.
pub fn run_gbm_garch(
    weekly_returns: &[f64],
    config: &SimulationConfig,
    last_price: f64,
) -> Result<SimulationResult, FluxError> {
    if weekly_returns.len() < 20 {
        return Err(FluxError::SimulationError(
            "Need at least 20 weekly observations".into(),
        ));
    }

    let n = weekly_returns.len();

    let mu_drift: f64 = weekly_returns.iter().sum::<f64>() / n as f64;

    let (garch_model, selected_p, selected_q) = match &config.garch_order {
        GarchOrder::Auto { max_p, max_q } => optimize_garch(weekly_returns, *max_p, *max_q)?,
        GarchOrder::Manual { p, q } => {
            use anofox_forecast::core::TimeSeries;
            use anofox_forecast::models::Forecaster;
            use chrono::{TimeZone, Utc};

            let timestamps: Vec<_> = (0..n)
                .map(|i| {
                    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64)
                })
                .collect();
            let ts = TimeSeries::univariate(timestamps, weekly_returns.to_vec())
                .map_err(|e| FluxError::GARCHError(format!("Failed to create TimeSeries: {e}")))?;
            let mut gm = anofox_forecast::models::GARCH::new(*p, *q);
            gm.fit(&ts)
                .map_err(|e| FluxError::GARCHError(format!("GARCH({p},{q}) fit failed: {e}")))?;
            (gm, *p, *q)
        }
    };

    let variance_forecast = garch_model
        .forecast_variance(config.forecast_weeks)
        .map_err(|e| FluxError::GARCHError(format!("GARCH variance forecast failed: {e}")))?;

    let vol_forecast: Vec<f64> = variance_forecast.iter().map(|v| v.sqrt()).collect();

    let z = normal_inv_cdf((1.0 + config.confidence_level) / 2.0);
    let vol_lower: Vec<f64> = vol_forecast
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let se = v * (2.0 / (i as f64 + 1.0)).sqrt();
            (v - z * se).max(0.0)
        })
        .collect();
    let vol_upper: Vec<f64> = vol_forecast
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let se = v * (2.0 / (i as f64 + 1.0)).sqrt();
            v + z * se
        })
        .collect();

    let res_variance = weekly_returns
        .iter()
        .map(|r| (r - mu_drift).powi(2))
        .sum::<f64>()
        / n as f64;
    let res_std = res_variance.sqrt();
    let standardized_returns: Vec<f64> = if res_std > 0.0 {
        weekly_returns
            .iter()
            .map(|&r| (r - mu_drift) / res_std)
            .collect()
    } else {
        weekly_returns.to_vec()
    };

    let norm_bootstrap_paths = run_gbm_bootstrap(
        &standardized_returns,
        mu_drift,
        &vol_forecast,
        config.forecast_weeks,
        config.n_bootstrap,
        config.seed,
    );

    let (norm_price_median, norm_price_lower, norm_price_upper) =
        compute_price_percentiles(&norm_bootstrap_paths, config.confidence_level);

    let summary = compute_summary(&norm_bootstrap_paths, weekly_returns, config.var_level);

    // Scale bootstrap paths to actual prices
    let bootstrap_paths: Vec<Vec<f64>> = norm_bootstrap_paths
        .iter()
        .map(|path| path.iter().map(|p| p * last_price).collect())
        .collect();
    let price_median: Vec<f64> = norm_price_median.iter().map(|p| p * last_price).collect();
    let price_lower: Vec<f64> = norm_price_lower.iter().map(|p| p * last_price).collect();
    let price_upper: Vec<f64> = norm_price_upper.iter().map(|p| p * last_price).collect();

    // Scale price percentiles to actual dollar terms
    let mut summary = summary;
    summary.price_percentiles = summary.price_percentiles.map(|p| p * last_price);

    Ok(SimulationResult {
        price_median,
        price_lower,
        price_upper,
        vol_forecast,
        vol_lower,
        vol_upper,
        bootstrap_paths,
        summary,
        garch_order_selected: (selected_p, selected_q),
        mu_drift,
        last_price,
        confidence_level: config.confidence_level,
    })
}

// ─── GBM Bootstrap ────────────────────────────────────────────────────────────

/// Run parallel GBM bootstrap simulation using rayon.
///
/// For each path, resamples standardized returns with replacement and applies
/// the GBM equation: `S_{t+1} = S_t · exp((μ − σ²/2) + σ · ε)`.
fn run_gbm_bootstrap(
    standardized_returns: &[f64],
    mu: f64,
    vol_forecast: &[f64],
    forecast_weeks: usize,
    n_paths: usize,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut rng = if let Some(s) = seed {
                rand::rngs::StdRng::seed_from_u64(s.wrapping_add(path_idx as u64))
            } else {
                rand::rngs::StdRng::from_entropy()
            };

            let mut path = Vec::with_capacity(forecast_weeks + 1);
            path.push(1.0); // normalized starting price

            for t in 0..forecast_weeks {
                let epsilon = *standardized_returns.choose(&mut rng).unwrap_or(&0.0);
                let sigma = vol_forecast
                    .get(t)
                    .copied()
                    .unwrap_or(vol_forecast.last().copied().unwrap_or(0.02));
                let prev = *path.last().unwrap_or(&1.0);
                let s_t = prev * (mu - 0.5 * sigma * sigma + sigma * epsilon).exp();
                path.push(s_t);
            }

            path
        })
        .collect()
}

// ─── Percentiles from paths ───────────────────────────────────────────────────

fn compute_price_percentiles(
    paths: &[Vec<f64>],
    confidence_level: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if paths.is_empty() || paths[0].is_empty() {
        return (vec![], vec![], vec![]);
    }
    let n_steps = paths[0].len();
    let lower_q = (1.0 - confidence_level) / 2.0;
    let upper_q = 1.0 - lower_q;

    let mut med = Vec::with_capacity(n_steps);
    let mut low = Vec::with_capacity(n_steps);
    let mut high = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let mut vals: Vec<f64> = paths.iter().map(|p| p[step]).collect();
        let p = compute_percentiles(&mut vals, &[lower_q, 0.5, upper_q]);
        low.push(p[0]);
        med.push(p[1]);
        high.push(p[2]);
    }

    (med, low, high)
}

// ─── Summary Statistics ───────────────────────────────────────────────────────

/// Compute quantile values from a dataset.
/// `quantiles` should be in `[0.0, 1.0]` range.
fn compute_percentiles(sorted_data: &mut [f64], quantiles: &[f64]) -> Vec<f64> {
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted_data.len();
    quantiles
        .iter()
        .map(|&q| {
            let pos = q * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            if lo == hi {
                sorted_data[lo]
            } else {
                let frac = pos - lo as f64;
                sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac
            }
        })
        .collect()
}

/// Compute summary statistics from bootstrap paths and historical returns.
fn compute_summary(
    paths: &[Vec<f64>],
    _historical_returns: &[f64],
    var_level: f64,
) -> SummaryStats {
    let n = paths.len() as f64;
    let quantiles = [0.025, 0.25, 0.50, 0.75, 0.975];

    // Terminal returns across paths
    let terminal_returns: Vec<f64> = paths
        .iter()
        .filter_map(|p| p.last())
        .map(|&p| p.ln())
        .collect();

    let mean = terminal_returns.iter().sum::<f64>() / terminal_returns.len() as f64;
    let variance = terminal_returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (terminal_returns.len() - 1) as f64;
    let std = variance.sqrt();

    // Annualize (assuming weekly data, ~52 weeks/year)
    let forecast_weeks = paths.first().map(|p| p.len() - 1).unwrap_or(52) as f64;
    let forecast_years = forecast_weeks / 52.0;
    let mean_annual_return = mean / forecast_years;
    let annual_volatility = std / forecast_years.sqrt();

    let sharpe_ratio = if annual_volatility > 0.0 {
        mean_annual_return / annual_volatility
    } else {
        0.0
    };

    // Max drawdown: worst across all bootstrap paths
    let drawdowns: Vec<f64> = paths.iter().map(|p| compute_max_drawdown(p)).collect();
    let max_drawdown = drawdowns.iter().cloned().fold(0.0f64, f64::min);
    let mut sorted_dd = drawdowns.clone();
    let median_drawdown = compute_percentiles(&mut sorted_dd, &[0.50])[0];

    // Per-path annualized returns, volatilities, and Sharpe ratios
    let annualization_factor = 52.0 / forecast_weeks;

    let path_annual_returns: Vec<f64> = paths
        .iter()
        .filter_map(|p| p.last().map(|&end| end.ln() * annualization_factor))
        .collect();

    let path_annual_vols: Vec<f64> = paths
        .iter()
        .map(|p| {
            let rets: Vec<f64> = p.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
            let m = rets.iter().sum::<f64>() / rets.len() as f64;
            let v = rets.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
            v.sqrt() * (52.0_f64).sqrt()
        })
        .collect();

    let path_sharpes: Vec<f64> = path_annual_returns
        .iter()
        .zip(path_annual_vols.iter())
        .map(|(&r, &v)| if v > 0.0 { r / v } else { 0.0 })
        .collect();

    let return_percentiles: [f64; 5] =
        compute_percentiles(&mut path_annual_returns.clone(), &quantiles)
            .try_into()
            .unwrap_or([0.0; 5]);
    let volatility_percentiles: [f64; 5] =
        compute_percentiles(&mut path_annual_vols.clone(), &quantiles)
            .try_into()
            .unwrap_or([0.0; 5]);
    let sharpe_percentiles: [f64; 5] = compute_percentiles(&mut path_sharpes.clone(), &quantiles)
        .try_into()
        .unwrap_or([0.0; 5]);

    // Terminal prices across paths (normalized, scaled by caller)
    let terminal_prices: Vec<f64> = paths.iter().filter_map(|p| p.last().copied()).collect();
    let price_percentiles: [f64; 5] = compute_percentiles(&mut terminal_prices.clone(), &quantiles)
        .try_into()
        .unwrap_or([0.0; 5]);

    // Skewness and kurtosis of terminal returns
    let skewness = if std > 0.0 {
        terminal_returns
            .iter()
            .map(|r| ((r - mean) / std).powi(3))
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    let kurtosis = if std > 0.0 {
        let m4 = terminal_returns
            .iter()
            .map(|r| ((r - mean) / std).powi(4))
            .sum::<f64>()
            / n;
        m4 - 3.0 // excess kurtosis
    } else {
        0.0
    };

    // VaR and CVaR from terminal simple returns
    let terminal_simple: Vec<f64> = paths
        .iter()
        .filter_map(|p| p.last().copied())
        .map(|terminal| terminal - 1.0)
        .collect();
    let mut sorted_simple = terminal_simple.clone();
    sorted_simple.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let var_idx = (var_level * sorted_simple.len() as f64).floor() as usize;
    let var = sorted_simple[var_idx.min(sorted_simple.len() - 1)];
    let tail: Vec<f64> = sorted_simple.iter().take(var_idx + 1).copied().collect();
    let cvar = if tail.is_empty() {
        var
    } else {
        tail.iter().sum::<f64>() / tail.len() as f64
    };

    // Estimate t-distribution degrees of freedom from kurtosis
    let t_df_estimate = if kurtosis > 0.1 {
        (6.0 / kurtosis) + 4.0
    } else {
        30.0 // effectively normal
    };

    SummaryStats {
        mean_annual_return,
        annual_volatility,
        sharpe_ratio,
        max_drawdown,
        median_drawdown,
        skewness,
        kurtosis,
        var,
        cvar,
        var_level,
        t_df_estimate,
        return_percentiles,
        volatility_percentiles,
        sharpe_percentiles,
        price_percentiles,
    }
}

/// Compute maximum drawdown from a price path.
fn compute_max_drawdown(path: &[f64]) -> f64 {
    let mut peak = path.first().copied().unwrap_or(1.0);
    let mut max_dd = 0.0;
    for &price in path.iter().skip(1) {
        if price > peak {
            peak = price;
        }
        let dd = (peak - price) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    -max_dd // negative convention
}

// ─── HTML Dashboard Generation ────────────────────────────────────────────────

/// Generate a self-contained HTML dashboard with interactive Chart.js visualizations.
///
/// Returns the full HTML string. The dashboard includes:
/// - Price forecast chart with confidence band
/// - Volatility forecast chart with confidence band
/// - Bootstrap fan chart
/// - Terminal price distribution histogram
/// - Summary statistics table
/// - Distribution percentiles table
///
/// # Errors
///
/// Returns [`FluxError::DashboardError`] if data serialization fails.
pub fn generate_dashboard(
    ticker: &str,
    result: &SimulationResult,
    historical_prices: &[f64],
) -> Result<String, FluxError> {
    let n = result.price_median.len();
    let forecast_labels: Vec<String> = (1..=n).map(|w| format!("W{w}")).collect();

    // Sample bootstrap paths for fan chart (max 50 paths)
    let sample_size = result.bootstrap_paths.len().min(50);
    let step = if result.bootstrap_paths.len() > sample_size {
        result.bootstrap_paths.len() / sample_size
    } else {
        1
    };
    let sampled_paths: Vec<&Vec<f64>> = result.bootstrap_paths.iter().step_by(step).collect();

    // Terminal prices for histogram
    let terminal_prices: Vec<f64> = result
        .bootstrap_paths
        .iter()
        .filter_map(|p| p.last().copied())
        .collect();
    let hist_bins = compute_histogram(&terminal_prices, 30);

    // Historical labels
    let hist_labels: Vec<String> = (0..historical_prices.len())
        .map(|i| format!("H{i}"))
        .collect();

    let mut all_labels = hist_labels.clone();
    all_labels.extend(forecast_labels.iter().cloned());

    let level_pct = (result.confidence_level * 100.0).round() as usize;
    let gp = result.garch_order_selected.0;
    let gq = result.garch_order_selected.1;

    let ret_cls = if result.summary.mean_annual_return >= 0.0 {
        "positive"
    } else {
        "negative"
    };
    let sharpe_cls = if result.summary.sharpe_ratio >= 0.5 {
        "positive"
    } else if result.summary.sharpe_ratio >= 0.0 {
        "neutral"
    } else {
        "negative"
    };
    let skew_cls = if result.summary.skewness.abs() < 0.5 {
        "neutral"
    } else {
        "negative"
    };
    let kurt_cls = if result.summary.kurtosis > 1.0 {
        "negative"
    } else {
        "neutral"
    };

    // Pre-compute all formatted values
    let ret_pct = format!("{:.2}", result.summary.mean_annual_return * 100.0);
    let vol_pct = format!("{:.2}", result.summary.annual_volatility * 100.0);
    let sharpe_val = format!("{:.3}", result.summary.sharpe_ratio);
    let dd_pct = format!("{:.2}", result.summary.max_drawdown * 100.0);
    let md_pct = format!("{:.2}", result.summary.median_drawdown * 100.0);
    let skew_val = format!("{:.4}", result.summary.skewness);
    let kurt_val = format!("{:.4}", result.summary.kurtosis);
    let t_df_val = format!("{:.2}", result.summary.t_df_estimate);
    let mu_drift_val = format!("{:+.4}", result.mu_drift);
    let var_pct = format!("{:.2}", result.summary.var * 100.0);
    let cvar_pct = format!("{:.2}", result.summary.cvar * 100.0);
    let var_level_pct = (result.summary.var_level * 100.0).round() as usize;
    let rp: Vec<String> = result
        .summary
        .return_percentiles
        .iter()
        .map(|v| format!("{:.2}", v * 100.0))
        .collect();
    let vp: Vec<String> = result
        .summary
        .volatility_percentiles
        .iter()
        .map(|v| format!("{:.2}", v * 100.0))
        .collect();
    let sp: Vec<String> = result
        .summary
        .sharpe_percentiles
        .iter()
        .map(|v| format!("{:.3}", v))
        .collect();
    let pp: Vec<String> = result
        .summary
        .price_percentiles
        .iter()
        .map(|v| format!("${:.2}", v))
        .collect();

    let forecast_labels_json = serde_json_vec_str(&forecast_labels);
    let hist_labels_json = serde_json_vec_str(&hist_labels);
    let all_labels_json = serde_json_vec_str(&all_labels);
    let price_median_json = serde_json_vec_f64(&result.price_median);
    let price_lower_json = serde_json_vec_f64(&result.price_lower);
    let price_upper_json = serde_json_vec_f64(&result.price_upper);
    let vol_point_json = serde_json_vec_f64(&result.vol_forecast);
    let vol_lower_json = serde_json_vec_f64(&result.vol_lower);
    let vol_upper_json = serde_json_vec_f64(&result.vol_upper);
    let sampled_paths_json = serde_json_sampled_paths(&sampled_paths);
    let hist_prices_json = serde_json_vec_f64(&terminal_prices);
    let hist_bins_json = serde_json_histogram(&hist_bins);
    let historical_prices_json = serde_json_vec_f64(historical_prices);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fluxquant — {ticker} GBM-GARCH Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card-bg: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e0;
    --text-muted: #888;
    --accent: #00d4aa;
    --accent2: #7c5cfc;
    --red: #ff4757;
    --green: #2ed573;
    --yellow: #ffa502;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    padding: 24px;
    min-height: 100vh;
  }}
  header {{
    text-align: center;
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 8px;
  }}
  header .subtitle {{
    color: var(--text-muted);
    font-size: 14px;
  }}
  header .model-badge {{
    display: inline-block;
    background: var(--accent2);
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
  }}
  .dashboard {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .card.full-width {{
    grid-column: 1 / -1;
  }}
  .card h3 {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  canvas {{ width: 100% !important; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  th, td {{
    padding: 10px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--text-muted);
    font-weight: 500;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  td {{ color: var(--text); }}
  td.value {{
    font-weight: 600;
    font-family: 'SF Mono', 'Fira Code', monospace;
  }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .neutral {{ color: var(--yellow); }}
  footer {{
    text-align: center;
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 12px;
  }}
</style>
</head>
<body>
<header>
  <h1>{ticker} — GBM-GARCH Forecast</h1>
  <div class="subtitle">Generated by fluxquant &middot; {n} week forecast &middot; {level_pct}% confidence</div>
  <div class="model-badge">GBM + GARCH({gp},{gq})</div>
</header>

<div class="dashboard">
  <div class="card">
    <h3>Price Forecast ({level_pct}% CI)</h3>
    <canvas id="priceChart"></canvas>
  </div>
  <div class="card">
    <h3>Volatility Forecast ({level_pct}% CI)</h3>
    <canvas id="volChart"></canvas>
  </div>
  <div class="card">
    <h3>Bootstrap Fan Chart</h3>
    <canvas id="fanChart"></canvas>
  </div>
  <div class="card">
    <h3>Terminal Price Distribution</h3>
    <canvas id="histChart"></canvas>
  </div>
  <div class="card full-width">
    <h3>Summary Statistics</h3>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Mean Annual Return</td><td class="value {ret_cls}">{ret_pct}%</td></tr>
      <tr><td>Annual Volatility</td><td class="value neutral">{vol_pct}%</td></tr>
      <tr><td>Sharpe Ratio</td><td class="value {sharpe_cls}">{sharpe_val}</td></tr>
      <tr><td>Max Drawdown (worst)</td><td class="value negative">{dd_pct}%</td></tr>
      <tr><td>Median Drawdown</td><td class="value negative">{md_pct}%</td></tr>
      <tr><td>Skewness</td><td class="value {skew_cls}">{skew_val}</td></tr>
      <tr><td>Excess Kurtosis</td><td class="value {kurt_cls}">{kurt_val}</td></tr>
      <tr><td>VaR ({var_level_pct}%)</td><td class="value negative">{var_pct}%</td></tr>
      <tr><td>CVaR ({var_level_pct}%)</td><td class="value negative">{cvar_pct}%</td></tr>
      <tr><td>t-df Estimate</td><td class="value">{t_df_val}</td></tr>
      <tr><td>GARCH Order</td><td class="value">({gp},{gq})</td></tr>
      <tr><td>Drift (μ)</td><td class="value">{mu_drift_val}</td></tr>
      <tr><td>Bootstrap Paths</td><td class="value">{n_paths}</td></tr>
    </table>
  </div>
  <div class="card full-width">
    <h3>Distribution Percentiles</h3>
    <table>
      <tr><th>Percentile</th><th>2.5%</th><th>25%</th><th>50%</th><th>75%</th><th>97.5%</th></tr>
      <tr><td>Annual Return</td><td>{rp0}%</td><td>{rp1}%</td><td>{rp2}%</td><td>{rp3}%</td><td>{rp4}%</td></tr>
      <tr><td>Annual Volatility</td><td>{vp0}%</td><td>{vp1}%</td><td>{vp2}%</td><td>{vp3}%</td><td>{vp4}%</td></tr>
      <tr><td>Sharpe Ratio</td><td>{sp0}</td><td>{sp1}</td><td>{sp2}</td><td>{sp3}</td><td>{sp4}</td></tr>
      <tr><td>Terminal Price</td><td>{pp0}</td><td>{pp1}</td><td>{pp2}</td><td>{pp3}</td><td>{pp4}</td></tr>
    </table>
  </div>
</div>

<footer>
  fluxquant by Utkarsh Gaikwad &middot; GBM-GARCH Monte Carlo Simulation
</footer>

<script>
const DATA = {{
  forecastLabels: {forecast_labels_json},
  histLabels: {hist_labels_json},
  allLabels: {all_labels_json},
  priceMedian: {price_median_json},
  priceLower: {price_lower_json},
  priceUpper: {price_upper_json},
  volPoint: {vol_point_json},
  volLower: {vol_lower_json},
  volUpper: {vol_upper_json},
  sampledPaths: {sampled_paths_json},
  histPrices: {hist_prices_json},
  histBins: {hist_bins_json},
  historicalPrices: {historical_prices_json},
  forecastWeeks: {n}
}};

const gridColor = 'rgba(255,255,255,0.06)';
const tickColor = '#666';

function baseOpts() {{
  return {{
    responsive: true,
    maintainAspectRatio: true,
    plugins: {{
      legend: {{ display: true, labels: {{ color: '#aaa', boxWidth: 12, font: {{ size: 11 }} }} }},
      tooltip: {{ mode: 'index', intersect: false }},
      zoom: {{ pan: {{ enabled: true, mode: 'x' }}, zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'x' }} }}
    }},
    scales: {{
      x: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, maxTicksLimit: 12, font: {{ size: 10 }} }} }},
      y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, font: {{ size: 10 }} }} }}
    }}
  }};
}}

// ── Price Forecast Chart ──
new Chart(document.getElementById('priceChart'), {{
  type: 'line',
  data: {{
    labels: DATA.allLabels,
    datasets: [
      {{ label: 'Historical', data: [...DATA.historicalPrices, ...Array(DATA.priceMedian.length).fill(null)], borderColor: '#666', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: 'Upper CI', data: [...Array(DATA.historicalPrices.length).fill(null), ...DATA.priceUpper], borderColor: 'transparent', backgroundColor: 'rgba(0,212,170,0.15)', fill: '+2', pointRadius: 0 }},
      {{ label: 'Median', data: [...Array(DATA.historicalPrices.length).fill(null), ...DATA.priceMedian], borderColor: '#00d4aa', borderWidth: 2, backgroundColor: 'transparent', pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: 'Lower CI', data: [...Array(DATA.historicalPrices.length).fill(null), ...DATA.priceLower], borderColor: 'transparent', backgroundColor: 'rgba(0,212,170,0.15)', fill: '-2', pointRadius: 0 }}
    ]
  }},
  options: baseOpts()
}});

// ── Volatility Chart ──
new Chart(document.getElementById('volChart'), {{
  type: 'line',
  data: {{
    labels: DATA.forecastLabels,
    datasets: [
      {{ label: 'Upper CI', data: DATA.volUpper, borderColor: 'transparent', backgroundColor: 'rgba(124,92,252,0.15)', fill: '+1', pointRadius: 0 }},
      {{ label: 'Volatility', data: DATA.volPoint, borderColor: '#7c5cfc', borderWidth: 2, backgroundColor: 'transparent', pointRadius: 0, tension: 0.3 }},
      {{ label: 'Lower CI', data: DATA.volLower, borderColor: 'transparent', backgroundColor: 'rgba(124,92,252,0.15)', fill: false, pointRadius: 0 }}
    ]
  }},
  options: baseOpts()
}});

// ── Fan Chart ──
const fanDatasets = DATA.sampledPaths.map((path, i) => ({{
  data: path,
  borderColor: `rgba(0,212,170,${{0.05 + 0.02 * (i % 10)}})`,
  borderWidth: 1,
  pointRadius: 0,
  tension: 0.3
}}));
new Chart(document.getElementById('fanChart'), {{
  type: 'line',
  data: {{
    labels: DATA.forecastLabels,
    datasets: fanDatasets
  }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }} }} }}
}});

// ── Histogram ──
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.histBins.labels,
    datasets: [{{
      label: 'Frequency',
      data: DATA.histBins.counts,
      backgroundColor: 'rgba(0,212,170,0.6)',
      borderColor: '#00d4aa',
      borderWidth: 1
    }}]
  }},
  options: {{
    ...baseOpts(),
    plugins: {{ ...baseOpts().plugins, legend: {{ display: false }} }},
    scales: {{
      ...baseOpts().scales,
      x: {{ ...baseOpts().scales.x, title: {{ display: true, text: 'Terminal Price', color: '#888' }} }},
      y: {{ ...baseOpts().scales.y, title: {{ display: true, text: 'Frequency', color: '#888' }} }}
    }}
  }}
}});
</script>
</body>
</html>"#,
        ticker = ticker,
        n = n,
        level_pct = level_pct,
        gp = gp,
        gq = gq,
        n_paths = result.bootstrap_paths.len(),
        ret_pct = ret_pct,
        vol_pct = vol_pct,
        sharpe_val = sharpe_val,
        dd_pct = dd_pct,
        md_pct = md_pct,
        skew_val = skew_val,
        kurt_val = kurt_val,
        t_df_val = t_df_val,
        mu_drift_val = mu_drift_val,
        ret_cls = ret_cls,
        sharpe_cls = sharpe_cls,
        skew_cls = skew_cls,
        kurt_cls = kurt_cls,
        var_pct = var_pct,
        cvar_pct = cvar_pct,
        var_level_pct = var_level_pct,
        rp0 = rp[0],
        rp1 = rp[1],
        rp2 = rp[2],
        rp3 = rp[3],
        rp4 = rp[4],
        vp0 = vp[0],
        vp1 = vp[1],
        vp2 = vp[2],
        vp3 = vp[3],
        vp4 = vp[4],
        sp0 = sp[0],
        sp1 = sp[1],
        sp2 = sp[2],
        sp3 = sp[3],
        sp4 = sp[4],
        pp0 = pp[0],
        pp1 = pp[1],
        pp2 = pp[2],
        pp3 = pp[3],
        pp4 = pp[4],
        forecast_labels_json = forecast_labels_json,
        hist_labels_json = hist_labels_json,
        all_labels_json = all_labels_json,
        price_median_json = price_median_json,
        price_lower_json = price_lower_json,
        price_upper_json = price_upper_json,
        vol_point_json = vol_point_json,
        vol_lower_json = vol_lower_json,
        vol_upper_json = vol_upper_json,
        sampled_paths_json = sampled_paths_json,
        hist_prices_json = hist_prices_json,
        hist_bins_json = hist_bins_json,
        historical_prices_json = historical_prices_json,
    );

    Ok(html)
}

// ─── Histogram Computation ────────────────────────────────────────────────────

struct Histogram {
    labels: Vec<String>,
    counts: Vec<usize>,
}

fn compute_histogram(data: &[f64], n_bins: usize) -> Histogram {
    if data.is_empty() {
        return Histogram {
            labels: vec![],
            counts: vec![],
        };
    }
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-10 {
        return Histogram {
            labels: vec![format!("{:.2}", min)],
            counts: vec![data.len()],
        };
    }
    let bin_width = (max - min) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &v in data {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }
    let labels = (0..n_bins)
        .map(|i| format!("{:.2}", min + (i as f64 + 0.5) * bin_width))
        .collect();
    Histogram { labels, counts }
}

// ─── JSON Helpers (no serde_json dependency) ──────────────────────────────────

fn serde_json_vec_f64(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", inner.join(","))
}

fn serde_json_vec_str(v: &[String]) -> String {
    let inner: Vec<String> = v.iter().map(|s| format!("\"{s}\"")).collect();
    format!("[{}]", inner.join(","))
}

fn serde_json_sampled_paths(paths: &[&Vec<f64>]) -> String {
    let inner: Vec<String> = paths.iter().map(|p| serde_json_vec_f64(p)).collect();
    format!("[{}]", inner.join(","))
}

fn serde_json_histogram(h: &Histogram) -> String {
    format!(
        "{{\"labels\":{},\"counts\":{}}}",
        serde_json_vec_str(&h.labels),
        {
            let inner: Vec<String> = h.counts.iter().map(|c| c.to_string()).collect();
            format!("[{}]", inner.join(","))
        }
    )
}

// ─── Utility Functions ────────────────────────────────────────────────────────

fn normal_inv_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-10 {
        return 0.0;
    }
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -result } else { result }
}

// ─── Legacy API (backward compatible) ─────────────────────────────────────────

/// High-performance simulation engine for quantitative finance.
///
/// Provides a builder-pattern interface for quick simulations and
/// direct volatility fitting via GARCH.
pub struct SimulationEngine {
    /// Number of Monte Carlo paths to simulate.
    pub paths: usize,
}

impl SimulationEngine {
    /// Create a new engine with default settings (1000 paths).
    pub fn new() -> Self {
        Self { paths: 1000 }
    }

    /// Create a builder for configuring the engine.
    pub fn builder() -> SimulationEngineBuilder {
        SimulationEngineBuilder { paths: 1000 }
    }

    /// Run a Monte Carlo simulation with the configured path count.
    ///
    /// # Errors
    ///
    /// Returns [`FluxError::SimulationError`] if path count is zero.
    pub fn run_monte_carlo(&self) -> Result<(), FluxError> {
        if self.paths == 0 {
            return Err(FluxError::SimulationError(
                "Path count must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    /// Fit a GARCH(1,1) model and return annualized volatility.
    ///
    /// # Errors
    ///
    /// Returns [`FluxError::VolatilityError`] if data is empty or fitting fails.
    pub fn fit_volatility(&self, data: &[f64]) -> Result<f64, FluxError> {
        if data.is_empty() {
            return Err(FluxError::VolatilityError("Dataset is empty".into()));
        }

        use anofox_forecast::core::TimeSeries;
        use anofox_forecast::models::Forecaster;
        use anofox_forecast::models::garch::GARCH;
        use chrono::{TimeZone, Utc};

        let timestamps: Vec<_> = (0..data.len())
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect();

        let ts = TimeSeries::univariate(timestamps, data.to_vec())
            .map_err(|e| FluxError::VolatilityError(format!("Failed to create TimeSeries: {e}")))?;

        let mut model = GARCH::new(1, 1);
        model
            .fit(&ts)
            .map_err(|e| FluxError::VolatilityError(format!("GARCH fit failed: {e}")))?;

        let variance = model
            .forecast_variance(1)
            .map_err(|e| FluxError::VolatilityError(format!("Variance forecast failed: {e}")))?;

        let daily_vol = variance.first().copied().unwrap_or(0.0).sqrt();
        let annualized_vol = daily_vol * (252.0_f64).sqrt();

        Ok(annualized_vol)
    }
}

impl Default for SimulationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring a [`SimulationEngine`].
pub struct SimulationEngineBuilder {
    paths: usize,
}

impl SimulationEngineBuilder {
    /// Set the number of Monte Carlo paths.
    pub fn paths(mut self, paths: usize) -> Self {
        self.paths = paths;
        self
    }

    /// Build the configured [`SimulationEngine`].
    pub fn build(self) -> SimulationEngine {
        SimulationEngine { paths: self.paths }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_engine_has_1000_paths() {
        let engine = SimulationEngine::new();
        assert_eq!(engine.paths, 1000);
    }

    #[test]
    fn builder_sets_paths() {
        let engine = SimulationEngine::builder().paths(500).build();
        assert_eq!(engine.paths, 500);
    }

    #[test]
    fn monte_carlo_succeeds() {
        let engine = SimulationEngine::new();
        assert!(engine.run_monte_carlo().is_ok());
    }

    #[test]
    fn monte_carlo_zero_paths_fails() {
        let engine = SimulationEngine { paths: 0 };
        assert!(engine.run_monte_carlo().is_err());
    }

    #[test]
    fn fit_volatility_empty_data_fails() {
        let engine = SimulationEngine::new();
        assert!(engine.fit_volatility(&[]).is_err());
    }

    #[test]
    fn fit_volatility_returns_positive() {
        let engine = SimulationEngine::new();
        let returns: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin() * 0.02).collect();
        let vol = engine.fit_volatility(&returns).unwrap();
        assert!(vol > 0.0, "Volatility should be positive, got {vol}");
    }

    #[test]
    fn garch_optimization_basic() {
        let returns: Vec<f64> = (0..100)
            .map(|i| {
                let i = i as f64;
                (i * 0.3).sin() * 0.02 + (i * 0.1).cos() * 0.01
            })
            .collect();
        let result = optimize_garch(&returns, 2, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn garch_optimization_too_few() {
        let returns = vec![0.01; 10];
        assert!(optimize_garch(&returns, 1, 2).is_err());
    }

    #[test]
    fn normal_inv_cdf_sanity() {
        let p50 = normal_inv_cdf(0.5);
        assert!(p50.abs() < 0.001);
        let p975 = normal_inv_cdf(0.975);
        assert!((p975 - 1.96).abs() < 0.01);
        let p025 = normal_inv_cdf(0.025);
        assert!((p025 + 1.96).abs() < 0.01);
    }

    #[test]
    fn compute_percentiles_works() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let p = compute_percentiles(&mut data, &[0.5]);
        assert!((p[0] - 2.5).abs() < 0.001);
    }

    #[test]
    fn max_drawdown_calculation() {
        let path = vec![1.0, 1.2, 1.1, 1.3, 0.9, 1.0];
        let dd = compute_max_drawdown(&path);
        assert!(dd < 0.0);
        assert!((dd - (-0.3077)).abs() < 0.01);
    }

    #[test]
    fn histogram_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let h = compute_histogram(&data, 5);
        assert_eq!(h.labels.len(), 5);
        assert_eq!(h.counts.iter().sum::<usize>(), 5);
    }

    #[test]
    fn bootstrap_produces_paths() {
        let standardized = vec![0.1, -0.1, 0.05, -0.05, 0.15, -0.15, 0.08, -0.08];
        let vol_forecast = vec![0.02; 10];
        let paths = run_gbm_bootstrap(&standardized, 0.001, &vol_forecast, 10, 100, Some(42));
        assert_eq!(paths.len(), 100);
        assert_eq!(paths[0].len(), 11); // 10 weeks + initial price
        assert!((paths[0][0] - 1.0).abs() < 1e-10); // starts at 1.0
    }

    #[test]
    fn dashboard_generation() {
        let result = SimulationResult {
            price_median: vec![100.0, 101.0, 102.0],
            price_lower: vec![98.0, 97.0, 96.0],
            price_upper: vec![102.0, 104.0, 106.0],
            vol_forecast: vec![0.02, 0.02, 0.02],
            vol_lower: vec![0.015, 0.015, 0.015],
            vol_upper: vec![0.025, 0.025, 0.025],
            bootstrap_paths: vec![vec![100.0, 101.0, 102.0], vec![100.0, 99.0, 100.0]],
            summary: SummaryStats {
                mean_annual_return: 0.08,
                annual_volatility: 0.18,
                sharpe_ratio: 0.44,
                max_drawdown: -0.12,
                median_drawdown: -0.06,
                skewness: -0.3,
                kurtosis: 0.5,
                var: -0.15,
                cvar: -0.20,
                var_level: 0.05,
                t_df_estimate: 16.0,
                return_percentiles: [-0.15, -0.02, 0.06, 0.12, 0.29],
                volatility_percentiles: [0.10, 0.14, 0.17, 0.21, 0.28],
                sharpe_percentiles: [-0.3, 0.1, 0.4, 0.7, 1.2],
                price_percentiles: [95.0, 99.0, 101.0, 103.0, 108.0],
            },
            garch_order_selected: (1, 1),
            mu_drift: 0.001,
            last_price: 100.0,
            confidence_level: 0.95,
        };
        let html = generate_dashboard("AAPL", &result, &[100.0, 101.0, 102.0]).unwrap();
        assert!(html.contains("AAPL"));
        assert!(html.contains("chart.js"));
        assert!(html.contains("priceChart"));
        assert!(html.contains("volChart"));
        assert!(html.contains("fanChart"));
        assert!(html.contains("histChart"));
        assert!(html.contains("Summary Statistics"));
        assert!(html.contains("Distribution Percentiles"));
        assert!(html.contains("VaR"));
        assert!(html.contains("CVaR"));
    }
}
