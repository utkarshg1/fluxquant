//! # Fluxquant
//!
//! High-performance quantitative finance library built for speed and memory safety.
//!
//! Fluxquant treats market data as a continuous, streaming flow — emphasizing
//! speed, native concurrency, and memory safety for Monte Carlo simulations,
//! volatility modeling, and risk analytics.
//!
//! ## Example
//!
//! ```rust
//! use fluxquant::{SimulationConfig, SarimaOrder, GarchOrder, run_sarima_garch};
//!
//! let config = SimulationConfig {
//!     forecast_weeks: 260,
//!     confidence_level: 0.95,
//!     sarima_order: SarimaOrder::Auto { seasonal_period: 52 },
//!     garch_order: GarchOrder::Auto { max_p: 3, max_q: 3 },
//!     n_bootstrap: 1000,
//!     seed: Some(42),
//! };
//!
//! // With real weekly returns data:
//! // let result = run_sarima_garch(&weekly_returns, &config).unwrap();
//! ```

use rayon::prelude::*;
use thiserror::Error;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during fluxquant operations.
#[derive(Error, Debug)]
pub enum FluxError {
    /// Raised when a simulation fails to converge or encounters invalid state.
    #[error("Simulation failed: {0}")]
    SimulationError(String),

    /// Raised when volatility model fitting fails.
    #[error("Volatility fitting error: {0}")]
    VolatilityError(String),

    /// Raised when SARIMA model fitting fails.
    #[error("SARIMA fitting error: {0}")]
    SARIMAError(String),

    /// Raised when GARCH model fitting fails.
    #[error("GARCH fitting error: {0}")]
    GARCHError(String),

    /// Raised when dashboard generation fails.
    #[error("Dashboard generation error: {0}")]
    DashboardError(String),
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// SARIMA order specification.
#[derive(Debug, Clone)]
pub enum SarimaOrder {
    /// Automatically select best SARIMA order using AutoARIMA.
    Auto {
        /// Seasonal period (e.g. 52 for weekly data with yearly seasonality).
        seasonal_period: usize,
    },
    /// Manually specify SARIMA(p,d,q)(P,D,Q)[s] orders.
    #[allow(non_snake_case)]
    Manual {
        p: usize,
        d: usize,
        q: usize,
        P: usize,
        D: usize,
        Q: usize,
        s: usize,
    },
}

/// GARCH order specification.
#[derive(Debug, Clone)]
pub enum GarchOrder {
    /// Grid search over (p,q) combinations, select best by BIC.
    Auto {
        /// Maximum GARCH order p to test.
        max_p: usize,
        /// Maximum GARCH order q to test.
        max_q: usize,
    },
    /// Fixed GARCH(p,q) order.
    Manual { p: usize, q: usize },
}

/// Configuration for the SARIMA-GARCH simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of weeks to forecast.
    pub forecast_weeks: usize,
    /// Confidence level for intervals (e.g. 0.95 for 95%).
    pub confidence_level: f64,
    /// SARIMA order specification.
    pub sarima_order: SarimaOrder,
    /// GARCH order specification.
    pub garch_order: GarchOrder,
    /// Number of bootstrap paths for simulation.
    pub n_bootstrap: usize,
    /// Random seed for reproducibility (None = random).
    pub seed: Option<u64>,
}

// ─── Results ──────────────────────────────────────────────────────────────────

/// A forecast with point estimates and confidence bounds.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point forecast values.
    pub point: Vec<f64>,
    /// Lower confidence bound.
    pub lower: Vec<f64>,
    /// Upper confidence bound.
    pub upper: Vec<f64>,
    /// Confidence level used (e.g. 0.95).
    pub level: f64,
}

/// Summary statistics for the simulation.
#[derive(Debug, Clone)]
pub struct SummaryStats {
    /// Mean annualized return.
    pub mean_annual_return: f64,
    /// Annualized volatility.
    pub annual_volatility: f64,
    /// Sharpe ratio (assuming 0 risk-free rate).
    pub sharpe_ratio: f64,
    /// Worst-case max drawdown across all bootstrap paths (negative convention).
    pub max_drawdown: f64,
    /// Median max drawdown across bootstrap paths (negative convention).
    pub median_drawdown: f64,
    /// Return distribution skewness.
    pub skewness: f64,
    /// Return distribution excess kurtosis.
    pub kurtosis: f64,
    /// Estimated t-distribution degrees of freedom.
    pub t_df_estimate: f64,
    /// Annual return percentiles: [2.5%, 25%, 50%, 75%, 97.5%]
    pub return_percentiles: [f64; 5],
    /// Annual volatility percentiles: [2.5%, 25%, 50%, 75%, 97.5%]
    pub volatility_percentiles: [f64; 5],
    /// Sharpe ratio percentiles: [2.5%, 25%, 50%, 75%, 97.5%]
    pub sharpe_percentiles: [f64; 5],
}

/// Full result of a SARIMA-GARCH simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// SARIMA returns forecast with confidence intervals.
    pub returns_forecast: ForecastResult,
    /// GARCH volatility forecast with confidence intervals.
    pub volatility_forecast: ForecastResult,
    /// Complete bootstrap price paths (one Vec per path).
    pub bootstrap_paths: Vec<Vec<f64>>,
    /// Summary statistics.
    pub summary: SummaryStats,
    /// Actual GARCH order selected (p, q).
    pub garch_order_selected: (usize, usize),
    /// Actual SARIMA order as a display string.
    pub sarima_order_selected: String,
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
/// Tests all combinations of p in `1..=max_p` and q in `1..=max_q`,
/// fits each model, and returns the one with the lowest BIC.
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

// ─── SARIMA-GARCH Pipeline ────────────────────────────────────────────────────

/// Run the full SARIMA-GARCH simulation pipeline.
///
/// 1. Fits SARIMA (auto or manual) to weekly returns → return forecasts + CI
/// 2. Extracts SARIMA residuals
/// 3. Fits GARCH(1,1) on residuals → volatility forecasts + CI
/// 4. Runs parallel bootstrap for full path distributions
/// 5. Computes summary statistics
///
/// `weekly_returns` should be log-returns: ln(P_t / P_{t-1}).
pub fn run_sarima_garch(
    weekly_returns: &[f64],
    config: &SimulationConfig,
) -> Result<SimulationResult, FluxError> {
    if weekly_returns.len() < 20 {
        return Err(FluxError::SimulationError(
            "Need at least 20 weekly observations".into(),
        ));
    }

    use anofox_forecast::core::TimeSeries;
    use anofox_forecast::models::Forecaster;
    use chrono::{TimeZone, Utc};

    let timestamps: Vec<_> = (0..weekly_returns.len())
        .map(|i| {
            Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::weeks(i as i64)
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps.clone(), weekly_returns.to_vec())
        .map_err(|e| FluxError::SARIMAError(format!("Failed to create TimeSeries: {e}")))?;

    // ── Step 1: Fit SARIMA ────────────────────────────────────────────────
    match &config.sarima_order {
        SarimaOrder::Auto { seasonal_period } => {
            let mut m = anofox_forecast::models::arima::AutoARIMA::seasonal(*seasonal_period);
            m.fit(&ts)
                .map_err(|e| FluxError::SARIMAError(format!("AutoARIMA fit failed: {e}")))?;

            let sarima_order_str = m
                .selected_full_order()
                .map(|o| {
                    format!(
                        "({},{},{})({},{},{})[{}]",
                        o.p, o.d, o.q, o.cap_p, o.cap_d, o.cap_q, o.s
                    )
                })
                .unwrap_or_else(|| "AutoARIMA".into());

            let sarima_forecast = m
                .predict_with_intervals(config.forecast_weeks, config.confidence_level)
                .map_err(|e| FluxError::SARIMAError(format!("SARIMA predict failed: {e}")))?;

            let sarima_returns = ForecastResult {
                point: sarima_forecast.primary().to_vec(),
                lower: sarima_forecast
                    .lower_series(0)
                    .map(|v| v.to_vec())
                    .unwrap_or_default(),
                upper: sarima_forecast
                    .upper_series(0)
                    .map(|v| v.to_vec())
                    .unwrap_or_default(),
                level: config.confidence_level,
            };

            let residuals = m.residuals().ok_or_else(|| {
                FluxError::SARIMAError("No residuals available after SARIMA fit".into())
            })?;

            // ── Step 3: Fit GARCH on residuals ────────────────────────────
            let (garch_model, selected_p, selected_q) = match &config.garch_order {
                GarchOrder::Auto { max_p, max_q } => optimize_garch(residuals, *max_p, *max_q)?,
                GarchOrder::Manual { p, q } => {
                    let residual_ts = TimeSeries::univariate(timestamps, residuals.to_vec())
                        .map_err(|e| {
                            FluxError::GARCHError(format!(
                                "Failed to create residual TimeSeries: {e}"
                            ))
                        })?;
                    let mut gm = anofox_forecast::models::GARCH::new(*p, *q);
                    gm.fit(&residual_ts).map_err(|e| {
                        FluxError::GARCHError(format!("GARCH({p},{q}) fit failed: {e}"))
                    })?;
                    (gm, *p, *q)
                }
            };

            // GARCH variance forecast → volatility CI
            let variance_forecast = garch_model
                .forecast_variance(config.forecast_weeks)
                .map_err(|e| {
                    FluxError::GARCHError(format!("GARCH variance forecast failed: {e}"))
                })?;

            let vol_point: Vec<f64> = variance_forecast.iter().map(|v| v.sqrt()).collect();

            let vol_level = config.confidence_level;
            let z = normal_inv_cdf((1.0 + vol_level) / 2.0);
            let vol_lower: Vec<f64> = vol_point
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let se = v * (2.0 / (i as f64 + 1.0)).sqrt();
                    (v - z * se).max(0.0)
                })
                .collect();
            let vol_upper: Vec<f64> = vol_point
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let se = v * (2.0 / (i as f64 + 1.0)).sqrt();
                    v + z * se
                })
                .collect();

            let volatility_forecast = ForecastResult {
                point: vol_point,
                lower: vol_lower,
                upper: vol_upper,
                level: config.confidence_level,
            };

            // ── Step 4: Parallel bootstrap ────────────────────────────────
            let mean_return = weekly_returns.iter().sum::<f64>() / weekly_returns.len() as f64;
            let res_std =
                (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
            let standardized_residuals: Vec<f64> = if res_std > 0.0 {
                residuals.iter().map(|&r| r / res_std).collect()
            } else {
                residuals.to_vec()
            };
            let bootstrap_paths = run_parallel_bootstrap(
                &standardized_residuals,
                mean_return,
                &volatility_forecast.point,
                config.forecast_weeks,
                config.n_bootstrap,
                config.seed,
            );

            // ── Step 5: Summary statistics ────────────────────────────────
            let summary = compute_summary(&bootstrap_paths, weekly_returns);

            return Ok(SimulationResult {
                returns_forecast: sarima_returns,
                volatility_forecast,
                bootstrap_paths,
                summary,
                garch_order_selected: (selected_p, selected_q),
                sarima_order_selected: sarima_order_str,
            });
        }
        SarimaOrder::Manual {
            p,
            d,
            q,
            P,
            D,
            Q,
            s,
        } => {
            let mut m = anofox_forecast::models::arima::SARIMA::new(*p, *d, *q, *P, *D, *Q, *s);
            m.fit(&ts).map_err(|e| {
                FluxError::SARIMAError(format!(
                    "SARIMA({p},{d},{q})({P},{D},{Q})[{s}] fit failed: {e}"
                ))
            })?;

            let sarima_order_str = format!("({p},{d},{q})({P},{D},{Q})[{s}]");

            let sarima_forecast = m
                .predict_with_intervals(config.forecast_weeks, config.confidence_level)
                .map_err(|e| FluxError::SARIMAError(format!("SARIMA predict failed: {e}")))?;

            let sarima_returns = ForecastResult {
                point: sarima_forecast.primary().to_vec(),
                lower: sarima_forecast
                    .lower_series(0)
                    .map(|v| v.to_vec())
                    .unwrap_or_default(),
                upper: sarima_forecast
                    .upper_series(0)
                    .map(|v| v.to_vec())
                    .unwrap_or_default(),
                level: config.confidence_level,
            };

            let residuals = m.residuals().ok_or_else(|| {
                FluxError::SARIMAError("No residuals available after SARIMA fit".into())
            })?;

            let (garch_model, selected_p, selected_q) = match &config.garch_order {
                GarchOrder::Auto { max_p, max_q } => optimize_garch(residuals, *max_p, *max_q)?,
                GarchOrder::Manual { p: gp, q: gq } => {
                    let residual_ts = TimeSeries::univariate(timestamps, residuals.to_vec())
                        .map_err(|e| {
                            FluxError::GARCHError(format!(
                                "Failed to create residual TimeSeries: {e}"
                            ))
                        })?;
                    let mut gm = anofox_forecast::models::GARCH::new(*gp, *gq);
                    gm.fit(&residual_ts).map_err(|e| {
                        FluxError::GARCHError(format!("GARCH({gp},{gq}) fit failed: {e}"))
                    })?;
                    (gm, *gp, *gq)
                }
            };

            let variance_forecast = garch_model
                .forecast_variance(config.forecast_weeks)
                .map_err(|e| {
                    FluxError::GARCHError(format!("GARCH variance forecast failed: {e}"))
                })?;

            let vol_point: Vec<f64> = variance_forecast.iter().map(|v| v.sqrt()).collect();
            let vol_level = config.confidence_level;
            let z = normal_inv_cdf((1.0 + vol_level) / 2.0);
            let vol_lower: Vec<f64> = vol_point
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let se = v * (2.0 / (i as f64 + 1.0)).sqrt();
                    (v - z * se).max(0.0)
                })
                .collect();
            let vol_upper: Vec<f64> = vol_point
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let se = v * (2.0 / (i as f64 + 1.0)).sqrt();
                    v + z * se
                })
                .collect();

            let volatility_forecast = ForecastResult {
                point: vol_point,
                lower: vol_lower,
                upper: vol_upper,
                level: config.confidence_level,
            };

            let mean_return = weekly_returns.iter().sum::<f64>() / weekly_returns.len() as f64;
            let res_std =
                (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
            let standardized_residuals: Vec<f64> = if res_std > 0.0 {
                residuals.iter().map(|&r| r / res_std).collect()
            } else {
                residuals.to_vec()
            };
            let bootstrap_paths = run_parallel_bootstrap(
                &standardized_residuals,
                mean_return,
                &volatility_forecast.point,
                config.forecast_weeks,
                config.n_bootstrap,
                config.seed,
            );

            let summary = compute_summary(&bootstrap_paths, weekly_returns);

            return Ok(SimulationResult {
                returns_forecast: sarima_returns,
                volatility_forecast,
                bootstrap_paths,
                summary,
                garch_order_selected: (selected_p, selected_q),
                sarima_order_selected: sarima_order_str,
            });
        }
    }
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

/// Run parallel bootstrap simulation using rayon.
///
/// Resamples standardized residuals with replacement, reconstructs
/// synthetic return paths using GARCH conditional volatility, and
/// accumulates to price paths.
fn run_parallel_bootstrap(
    residuals: &[f64],
    mean_return: f64,
    vol_forecast: &[f64],
    forecast_weeks: usize,
    n_paths: usize,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    let _n_obs = residuals.len();

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
                let resampled = *residuals.choose(&mut rng).unwrap_or(&0.0);
                let vol = vol_forecast
                    .get(t)
                    .copied()
                    .unwrap_or(vol_forecast.last().copied().unwrap_or(0.02));
                let weekly_return = mean_return + vol * resampled;
                let prev = *path.last().unwrap_or(&1.0);
                path.push(prev * weekly_return.exp());
            }

            path
        })
        .collect()
}

// ─── Summary Statistics ───────────────────────────────────────────────────────

/// Compute quantile values from a dataset.
/// `quantiles` should be in [0.0, 1.0] range.
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
fn compute_summary(paths: &[Vec<f64>], _historical_returns: &[f64]) -> SummaryStats {
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
    // `mean` and `std` are computed from terminal log-returns over the full forecast period,
    // so we annualize by dividing by forecast years (= forecast_weeks/52).
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

    // Skewness and kurtosis of terminal returns
    let skewness = if std > 0.0 {
        let m3 = terminal_returns
            .iter()
            .map(|r| ((r - mean) / std).powi(3))
            .sum::<f64>()
            / n;
        m3
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
        t_df_estimate,
        return_percentiles,
        volatility_percentiles,
        sharpe_percentiles,
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
/// - Returns forecast chart with CI band
/// - Volatility forecast chart with CI band
/// - Bootstrap fan chart
/// - Terminal price distribution histogram
/// - Summary statistics table
pub fn generate_dashboard(
    ticker: &str,
    result: &SimulationResult,
    historical_prices: &[f64],
) -> Result<String, FluxError> {
    let n = result.returns_forecast.point.len();
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

    let level_pct = (result.returns_forecast.level * 100.0).round() as usize;

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fluxquant — {ticker} SARIMA-GARCH Forecast</title>
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
  <h1>{ticker} — SARIMA-GARCH Forecast</h1>
  <div class="subtitle">Generated by fluxquant &middot; {n} week forecast &middot; {level_pct}% confidence</div>
  <div class="model-badge">{sarima_order} + GARCH({gp},{gq})</div>
</header>

<div class="dashboard">
  <div class="card">
    <h3>Returns Forecast ({level}% CI)</h3>
    <canvas id="returnsChart"></canvas>
  </div>
  <div class="card">
    <h3>Volatility Forecast ({level}% CI)</h3>
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
      <tr><td>Sharpe Ratio</td><td class="value {sharpe_cls}">{sharpe}</td></tr>
      <tr><td>Max Drawdown (worst)</td><td class="value negative">{dd_pct}%</td></tr>
      <tr><td>Median Drawdown</td><td class="value negative">{md_pct}%</td></tr>
      <tr><td>Skewness</td><td class="value {skew_cls}">{skew}</td></tr>
      <tr><td>Excess Kurtosis</td><td class="value {kurt_cls}">{kurt}</td></tr>
      <tr><td>t-df Estimate</td><td class="value">{t_df}</td></tr>
      <tr><td>SARIMA Order</td><td class="value">{sarima_order}</td></tr>
      <tr><td>GARCH Order</td><td class="value">({gp},{gq})</td></tr>
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
    </table>
  </div>
</div>

<footer>
  fluxquant by Utkarsh Gaikwad &middot; SARIMA-GARCH Filtered Bootstrap Simulation
</footer>

<script>
const DATA = {{
  forecastLabels: {forecast_labels_json},
  histLabels: {hist_labels_json},
  allLabels: {all_labels_json},
  returnsPoint: {returns_point_json},
  returnsLower: {returns_lower_json},
  returnsUpper: {returns_upper_json},
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

function baseOpts(title) {{
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

// ── Returns Chart ──
new Chart(document.getElementById('returnsChart'), {{
  type: 'line',
  data: {{
    labels: DATA.forecastLabels,
    datasets: [
      {{ label: 'Upper CI', data: DATA.returnsUpper, borderColor: 'transparent', backgroundColor: 'rgba(0,212,170,0.15)', fill: '+1', pointRadius: 0 }},
      {{ label: 'Forecast', data: DATA.returnsPoint, borderColor: '#00d4aa', borderWidth: 2, backgroundColor: 'transparent', pointRadius: 0, tension: 0.3 }},
      {{ label: 'Lower CI', data: DATA.returnsLower, borderColor: 'transparent', backgroundColor: 'rgba(0,212,170,0.15)', fill: false, pointRadius: 0 }}
    ]
  }},
  options: baseOpts('Returns')
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
  options: baseOpts('Volatility')
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
  options: {{ ...baseOpts('Paths'), plugins: {{ ...baseOpts('Paths').plugins, legend: {{ display: false }} }} }}
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
    ...baseOpts('Distribution'),
    plugins: {{ ...baseOpts('Distribution').plugins, legend: {{ display: false }} }},
    scales: {{
      ...baseOpts('Distribution').scales,
      x: {{ ...baseOpts('Distribution').scales.x, title: {{ display: true, text: 'Terminal Price', color: '#888' }} }},
      y: {{ ...baseOpts('Distribution').scales.y, title: {{ display: true, text: 'Frequency', color: '#888' }} }}
    }}
  }}
}});
</script>
</body>
</html>"#,
        ticker = ticker,
        n = n,
        level = (result.returns_forecast.level * 100.0).round() as usize,
        sarima_order = result.sarima_order_selected,
        gp = result.garch_order_selected.0,
        gq = result.garch_order_selected.1,
        n_paths = result.bootstrap_paths.len(),
        ret_pct = format!("{:.2}", result.summary.mean_annual_return * 100.0),
        vol_pct = format!("{:.2}", result.summary.annual_volatility * 100.0),
        sharpe = format!("{:.3}", result.summary.sharpe_ratio),
        dd_pct = format!("{:.2}", result.summary.max_drawdown * 100.0),
        md_pct = format!("{:.2}", result.summary.median_drawdown * 100.0),
        skew = format!("{:.4}", result.summary.skewness),
        kurt = format!("{:.4}", result.summary.kurtosis),
        t_df = format!("{:.2}", result.summary.t_df_estimate),
        rp0 = format!("{:.2}", result.summary.return_percentiles[0] * 100.0),
        rp1 = format!("{:.2}", result.summary.return_percentiles[1] * 100.0),
        rp2 = format!("{:.2}", result.summary.return_percentiles[2] * 100.0),
        rp3 = format!("{:.2}", result.summary.return_percentiles[3] * 100.0),
        rp4 = format!("{:.2}", result.summary.return_percentiles[4] * 100.0),
        vp0 = format!("{:.2}", result.summary.volatility_percentiles[0] * 100.0),
        vp1 = format!("{:.2}", result.summary.volatility_percentiles[1] * 100.0),
        vp2 = format!("{:.2}", result.summary.volatility_percentiles[2] * 100.0),
        vp3 = format!("{:.2}", result.summary.volatility_percentiles[3] * 100.0),
        vp4 = format!("{:.2}", result.summary.volatility_percentiles[4] * 100.0),
        sp0 = format!("{:.3}", result.summary.sharpe_percentiles[0]),
        sp1 = format!("{:.3}", result.summary.sharpe_percentiles[1]),
        sp2 = format!("{:.3}", result.summary.sharpe_percentiles[2]),
        sp3 = format!("{:.3}", result.summary.sharpe_percentiles[3]),
        sp4 = format!("{:.3}", result.summary.sharpe_percentiles[4]),
        ret_cls = if result.summary.mean_annual_return >= 0.0 {
            "positive"
        } else {
            "negative"
        },
        sharpe_cls = if result.summary.sharpe_ratio >= 0.5 {
            "positive"
        } else if result.summary.sharpe_ratio >= 0.0 {
            "neutral"
        } else {
            "negative"
        },
        skew_cls = if result.summary.skewness.abs() < 0.5 {
            "neutral"
        } else {
            "negative"
        },
        kurt_cls = if result.summary.kurtosis > 1.0 {
            "negative"
        } else {
            "neutral"
        },
        forecast_labels_json = serde_json_vec_str(&forecast_labels),
        hist_labels_json = serde_json_vec_str(&hist_labels),
        all_labels_json = serde_json_vec_str(&all_labels),
        returns_point_json = serde_json_vec_f64(&result.returns_forecast.point),
        returns_lower_json = serde_json_vec_f64(&result.returns_forecast.lower),
        returns_upper_json = serde_json_vec_f64(&result.returns_forecast.upper),
        vol_point_json = serde_json_vec_f64(&result.volatility_forecast.point),
        vol_lower_json = serde_json_vec_f64(&result.volatility_forecast.lower),
        vol_upper_json = serde_json_vec_f64(&result.volatility_forecast.upper),
        sampled_paths_json = serde_json_sampled_paths(&sampled_paths),
        hist_prices_json = serde_json_vec_f64(&terminal_prices),
        hist_bins_json = serde_json_histogram(&hist_bins),
        historical_prices_json = serde_json_vec_f64(historical_prices),
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
pub struct SimulationEngine {
    pub paths: usize,
}

impl SimulationEngine {
    pub fn new() -> Self {
        Self { paths: 1000 }
    }

    pub fn builder() -> SimulationEngineBuilder {
        SimulationEngineBuilder { paths: 1000 }
    }

    pub fn run_monte_carlo(&self) -> Result<(), FluxError> {
        if self.paths == 0 {
            return Err(FluxError::SimulationError(
                "Path count must be greater than zero".into(),
            ));
        }
        Ok(())
    }

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

pub struct SimulationEngineBuilder {
    paths: usize,
}

impl SimulationEngineBuilder {
    pub fn paths(mut self, paths: usize) -> Self {
        self.paths = paths;
        self
    }

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
    fn normal_inv_cdf_sanity() {
        let p50 = normal_inv_cdf(0.5);
        assert!((p50).abs() < 0.001, "z(0.5) should be ~0, got {p50}");

        let p975 = normal_inv_cdf(0.975);
        assert!(
            (p975 - 1.96).abs() < 0.01,
            "z(0.975) should be ~1.96, got {p975}"
        );

        let p025 = normal_inv_cdf(0.025);
        assert!(
            (p025 + 1.96).abs() < 0.01,
            "z(0.025) should be ~-1.96, got {p025}"
        );
    }

    #[test]
    fn histogram_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let h = compute_histogram(&data, 5);
        assert_eq!(h.labels.len(), 5);
        assert_eq!(h.counts.iter().sum::<usize>(), 5);
    }

    #[test]
    fn max_drawdown_calculation() {
        let path = vec![1.0, 1.2, 1.1, 1.3, 0.9, 1.0];
        let dd = compute_max_drawdown(&path);
        assert!(dd < 0.0, "Max drawdown should be negative");
        assert!((dd - (-0.3077)).abs() < 0.01, "Expected ~-30.8%, got {dd}");
    }

    #[test]
    fn bootstrap_produces_paths() {
        let residuals = vec![0.1, -0.1, 0.05, -0.05, 0.15, -0.15, 0.08, -0.08];
        let vol_forecast = vec![0.02; 10];
        let paths = run_parallel_bootstrap(&residuals, 0.001, &vol_forecast, 10, 100, Some(42));
        assert_eq!(paths.len(), 100);
        assert_eq!(paths[0].len(), 11); // 10 weeks + initial price
        assert!(paths[0][0] == 1.0); // starts at 1.0
    }

    #[test]
    fn dashboard_generation() {
        let result = SimulationResult {
            returns_forecast: ForecastResult {
                point: vec![0.01, 0.02, 0.03],
                lower: vec![-0.01, 0.0, 0.01],
                upper: vec![0.03, 0.04, 0.05],
                level: 0.95,
            },
            volatility_forecast: ForecastResult {
                point: vec![0.02, 0.02, 0.02],
                lower: vec![0.015, 0.015, 0.015],
                upper: vec![0.025, 0.025, 0.025],
                level: 0.95,
            },
            bootstrap_paths: vec![vec![1.0, 1.01, 1.02], vec![1.0, 0.99, 1.0]],
            summary: SummaryStats {
                mean_annual_return: 0.08,
                annual_volatility: 0.18,
                sharpe_ratio: 0.44,
                max_drawdown: -0.12,
                median_drawdown: -0.06,
                skewness: -0.3,
                kurtosis: 0.5,
                t_df_estimate: 16.0,
                return_percentiles: [-0.15, -0.02, 0.06, 0.12, 0.29],
                volatility_percentiles: [0.10, 0.14, 0.17, 0.21, 0.28],
                sharpe_percentiles: [-0.3, 0.1, 0.4, 0.7, 1.2],
            },
            garch_order_selected: (1, 1),
            sarima_order_selected: "(1,1,1)(1,1,1)[52]".into(),
        };
        let html = generate_dashboard("AAPL", &result, &[100.0, 101.0, 102.0]).unwrap();
        assert!(html.contains("AAPL"));
        assert!(html.contains("chart.js"));
        assert!(html.contains("returnsChart"));
    }
}
