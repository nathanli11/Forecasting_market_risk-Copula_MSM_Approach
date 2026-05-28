"""Public API for the copula-MSM VaR replication package.
"""

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    REFERENCES_DIR,
    ensure_project_dirs,
)

# -----------------------------------------------------------------------------
# Data loading and return construction
# -----------------------------------------------------------------------------

from .data import (
    YahooIndexSpec,
    YAHOO_INDEXES,
    load_price_csv,
    simple_returns,
    log_returns,
    align_return_frame,
    resolve_yahoo_index,
    download_yahoo_index,
    download_yahoo_index_prices,
    save_yahoo_index_csv,
)

# -----------------------------------------------------------------------------
# Descriptive statistics, VaR backtests and SPA/GMM tests
# -----------------------------------------------------------------------------

from .risk import (
    summary_statistics,
    arch_lm_test,
    hill_tail_index,
    hurst_exponent,
    format_table_1,
    realized_volatility,
    standardize_returns,
    historical_var,
    gaussian_var,
    portfolio_returns as risk_portfolio_returns,
    var_exceedances as risk_var_exceedances,
    violation_rate,
    kupiec_pof_test,
    christoffersen_lr_test,
    violation_durations,
    meixner_polynomials,
    gmm_duration_test,
    stationary_bootstrap_indices,
    newey_west_variance,
    spa_pvalue_for_basis,
)

# -----------------------------------------------------------------------------
# MSM marginal model
# -----------------------------------------------------------------------------

from .msm import (
    MSMParams,
    MSMFitResult,
    make_msm_states,
    renewal_probabilities_from_gamma_k,
    transition_matrix_from_gammas,
    msm_loglikelihood,
    msm_filter,
    msm_filtered_cdf_series,
    msm_probability_integral_transform,
    msm_state_volatility_factors,
    msm_mixture_cdf,
    msm_mixture_quantile,
    msm_filter_from_result,
    build_msm_pit_frame,
    fit_msm,
    fit_msm_grid,
    msm_fit_result_to_dict,
)

# -----------------------------------------------------------------------------
# GARCH marginal model
# -----------------------------------------------------------------------------

from .garch import (
    GARCHFitResult,
    fit_garch_11,
    fit_garch_marginals,
    garch_standardized_residuals,
    garch_conditional_volatility,
    garch_probability_integral_transform,
    build_garch_pit_frame,
    build_garch_volatility_frame,
    garch_diagnostics,
    garch_fit_result_to_dict,
    garch_results_table,
    format_garch_table_3,
)

# -----------------------------------------------------------------------------
# Copula estimation
# -----------------------------------------------------------------------------

from .copulas import (
    CopulaFitResult,
    gaussian_copula_logpdf,
    student_copula_logpdf,
    clayton_copula_logpdf,
    rotated_clayton_copula_logpdf,
    gumbel_copula_logpdf,
    rotated_gumbel_copula_logpdf,
    frank_copula_logpdf,
    plackett_copula_logpdf,
    sjc_copula_logpdf,
    fit_copula,
    fit_all_copulas,
    fit_copula_grid,
    copula_fit_result_to_dict,
    copula_results_table,
    format_copula_table_4,
    pseudo_observations,
    gaussian_copula_correlation,
    simulate_gaussian_copula,
)

# -----------------------------------------------------------------------------
# Paper-like rolling VaR forecasts and VaR loss functions
# -----------------------------------------------------------------------------

from .var import (
    SUPPORTED_COPULAS,
    RollingSpec,
    prepare_bivariate_returns,
    validate_alpha_weights,
    rolling_windows,
    forecast_historical_var_rolling,
    forecast_variance_covariance_var_rolling,
    forecast_riskmetrics_var_rolling,
    copula_cdf,
    copula_conditional_cdf_u1_given_u2,
    portfolio_cdf_from_margins_and_copula,
    solve_portfolio_var,
    forecast_msm_copula_var_rolling,
    forecast_ccc_garch_var_rolling,
    forecast_garch_copula_var_rolling,
    portfolio_returns as var_portfolio_returns,
    var_exceedances as var_var_exceedances,
    violation_frequency,
    forecast_all_var_models,
    var_loss_series,
    smooth_var_loss_series,
    build_loss_panel,
)

# Backward-compatible short names used in the final notebook.
# These refer to the paper-like VaR convention from src.var.
portfolio_returns = var_portfolio_returns
var_exceedances = var_var_exceedances

# -----------------------------------------------------------------------------
# Reporting helpers for Tables 5-9
# -----------------------------------------------------------------------------

from .reporting import (
    BENCH_ORDER,
    COPULA_ORDER,
    STAT_ORDER,
    parse_model_name,
    make_lr_table,
    format_lr_table,
    make_gmm_table,
    rename_model_for_table9,
    order_table9_rows,
    make_spa_table9,
)

# -----------------------------------------------------------------------------
# Small I/O utilities
# -----------------------------------------------------------------------------

from .utils import (
    save_var,
    load_var,
    concat_var_series,
)

# -----------------------------------------------------------------------------
# Plotting and LaTeX export helpers
# -----------------------------------------------------------------------------

from .plotting import (
    plot_returns,
    save_plotly_figure,
    plot_price_evolution,
    plot_returns_and_squared_returns,
    plot_var_forecasts,
    csv_to_latex_table,
)


__all__ = [
    # config
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "REFERENCES_DIR",
    "ensure_project_dirs",

    # data
    "YahooIndexSpec",
    "YAHOO_INDEXES",
    "load_price_csv",
    "simple_returns",
    "log_returns",
    "align_return_frame",
    "resolve_yahoo_index",
    "download_yahoo_index",
    "download_yahoo_index_prices",
    "save_yahoo_index_csv",

    # risk and statistics
    "summary_statistics",
    "arch_lm_test",
    "hill_tail_index",
    "hurst_exponent",
    "format_table_1",
    "realized_volatility",
    "standardize_returns",
    "historical_var",
    "gaussian_var",
    "risk_portfolio_returns",
    "risk_var_exceedances",
    "violation_rate",
    "kupiec_pof_test",
    "christoffersen_lr_test",
    "violation_durations",
    "meixner_polynomials",
    "gmm_duration_test",
    "stationary_bootstrap_indices",
    "newey_west_variance",
    "spa_pvalue_for_basis",

    # MSM
    "MSMParams",
    "MSMFitResult",
    "make_msm_states",
    "renewal_probabilities_from_gamma_k",
    "transition_matrix_from_gammas",
    "msm_loglikelihood",
    "msm_filter",
    "msm_filtered_cdf_series",
    "msm_probability_integral_transform",
    "msm_state_volatility_factors",
    "msm_mixture_cdf",
    "msm_mixture_quantile",
    "msm_filter_from_result",
    "build_msm_pit_frame",
    "fit_msm",
    "fit_msm_grid",
    "msm_fit_result_to_dict",

    # GARCH
    "GARCHFitResult",
    "fit_garch_11",
    "fit_garch_marginals",
    "garch_standardized_residuals",
    "garch_conditional_volatility",
    "garch_probability_integral_transform",
    "build_garch_pit_frame",
    "build_garch_volatility_frame",
    "garch_diagnostics",
    "garch_fit_result_to_dict",
    "garch_results_table",
    "format_garch_table_3",

    # copulas
    "CopulaFitResult",
    "gaussian_copula_logpdf",
    "student_copula_logpdf",
    "clayton_copula_logpdf",
    "rotated_clayton_copula_logpdf",
    "gumbel_copula_logpdf",
    "rotated_gumbel_copula_logpdf",
    "frank_copula_logpdf",
    "plackett_copula_logpdf",
    "sjc_copula_logpdf",
    "fit_copula",
    "fit_all_copulas",
    "fit_copula_grid",
    "copula_fit_result_to_dict",
    "copula_results_table",
    "format_copula_table_4",
    "pseudo_observations",
    "gaussian_copula_correlation",
    "simulate_gaussian_copula",

    # rolling VaR
    "SUPPORTED_COPULAS",
    "RollingSpec",
    "prepare_bivariate_returns",
    "validate_alpha_weights",
    "rolling_windows",
    "forecast_historical_var_rolling",
    "forecast_variance_covariance_var_rolling",
    "forecast_riskmetrics_var_rolling",
    "copula_cdf",
    "copula_conditional_cdf_u1_given_u2",
    "portfolio_cdf_from_margins_and_copula",
    "solve_portfolio_var",
    "forecast_msm_copula_var_rolling",
    "forecast_ccc_garch_var_rolling",
    "forecast_garch_copula_var_rolling",
    "var_portfolio_returns",
    "var_var_exceedances",
    "portfolio_returns",
    "var_exceedances",
    "violation_frequency",
    "forecast_all_var_models",
    "var_loss_series",
    "smooth_var_loss_series",
    "build_loss_panel",

    # reporting
    "BENCH_ORDER",
    "COPULA_ORDER",
    "STAT_ORDER",
    "parse_model_name",
    "make_lr_table",
    "format_lr_table",
    "make_gmm_table",
    "rename_model_for_table9",
    "order_table9_rows",
    "make_spa_table9",

    # utils
    "save_var",
    "load_var",
    "concat_var_series",

    # plotting
    "plot_returns",
    "save_plotly_figure",
    "plot_price_evolution",
    "plot_returns_and_squared_returns",
    "plot_var_forecasts",
    "csv_to_latex_table",
]
