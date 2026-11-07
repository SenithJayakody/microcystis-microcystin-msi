import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# SETTINGS
# ============================================================

output_dir = "./Plots/toxity_cellcount_uncertainty_results"
os.makedirs(output_dir, exist_ok=True)

# Cell-count outputs and figure labels use x10^5 cells/mL.
CELL_UNIT_LABEL = "x10^5 cells/mL"
MC_UNIT_LABEL = "ug/L"

# Plot colors
LINE_COLOR = "red"
CI_COLOR = "orange"
PI_COLOR = "lightskyblue"

CI_ALPHA = 0.45
PI_ALPHA = 0.30


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)

    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    bias = np.mean(predicted - actual)
    residual_sd = np.std(predicted - actual, ddof=1)

    return r2, rmse, mae, bias, residual_sd


def add_constant(x):
    return sm.add_constant(np.asarray(x, dtype=float), has_constant="add")


# ============================================================
# MAIN TARGET ANALYSIS FUNCTION
# ============================================================

def run_target_analysis(
    target_name,
    train_csv,
    validation_csv,
    target_column,
    wasserstein_column,
    target_unit,
    output_prefix,
):
    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    tdf = pd.read_csv(train_csv)
    vtdf = pd.read_csv(validation_csv)

    x_train = tdf[target_column].to_numpy(dtype=float)
    y_train = tdf[wasserstein_column].to_numpy(dtype=float)

    x_val = vtdf[target_column].to_numpy(dtype=float)
    y_val = vtdf[wasserstein_column].to_numpy(dtype=float)

    # Fit concentration from Wasserstein using training data only.
    monitoring_model = sm.OLS(x_train, add_constant(y_train)).fit()

    monitoring_intercept = monitoring_model.params[0]
    monitoring_slope = monitoring_model.params[1]

    monitoring_training_label = (
        f"Fitted curve: y = {monitoring_intercept:.4f} "
        f"{monitoring_slope:+.4f}x"
    )

    monitoring_validation_label = (
        f"Calibrated curve: y = {monitoring_intercept:.4f} "
        f"{monitoring_slope:+.4f}x"
    )

    pred_train = monitoring_model.get_prediction(
        add_constant(y_train)
    ).summary_frame(alpha=0.05)

    pred_val = monitoring_model.get_prediction(
        add_constant(y_val)
    ).summary_frame(alpha=0.05)

    x_pred_train = pred_train["mean"].to_numpy()
    x_pred_val = pred_val["mean"].to_numpy()

    r2_train, rmse_train, mae_train, bias_train, residual_sd_train = regression_metrics(
        x_train, x_pred_train
    )

    r2_val, rmse_val, mae_val, bias_val, residual_sd_val = regression_metrics(
        x_val, x_pred_val
    )

    train_pi_coverage = np.mean(
        (x_train >= pred_train["obs_ci_lower"].to_numpy())
        & (x_train <= pred_train["obs_ci_upper"].to_numpy())
    )

    val_pi_coverage = np.mean(
        (x_val >= pred_val["obs_ci_lower"].to_numpy())
        & (x_val <= pred_val["obs_ci_upper"].to_numpy())
    )

    train_mean_ci_width = np.mean(
        pred_train["mean_ci_upper"].to_numpy()
        - pred_train["mean_ci_lower"].to_numpy()
    )

    train_mean_pi_width = np.mean(
        pred_train["obs_ci_upper"].to_numpy()
        - pred_train["obs_ci_lower"].to_numpy()
    )

    val_mean_ci_width = np.mean(
        pred_val["mean_ci_upper"].to_numpy()
        - pred_val["mean_ci_lower"].to_numpy()
    )

    val_mean_pi_width = np.mean(
        pred_val["obs_ci_upper"].to_numpy()
        - pred_val["obs_ci_lower"].to_numpy()
    )

    # ------------------------------------------------------------
    # Monitoring calibration plot: training only
    # Target concentration vs Wasserstein, with CI and PI.
    # ------------------------------------------------------------

    w_line_train = np.linspace(y_train.min(), y_train.max(), 200)

    pred_line_train = monitoring_model.get_prediction(
        add_constant(w_line_train)
    ).summary_frame(alpha=0.05)

    plt.figure(figsize=(8, 6))

    plt.fill_between(
        w_line_train,
        pred_line_train["obs_ci_lower"].to_numpy(),
        pred_line_train["obs_ci_upper"].to_numpy(),
        color=PI_COLOR,
        alpha=PI_ALPHA,
        label="95% prediction interval",
        zorder=1,
    )

    plt.fill_between(
        w_line_train,
        pred_line_train["mean_ci_lower"].to_numpy(),
        pred_line_train["mean_ci_upper"].to_numpy(),
        color=CI_COLOR,
        alpha=CI_ALPHA,
        label="95% confidence interval",
        zorder=2,
    )

    plt.scatter(
        y_train,
        x_train,
        alpha=0.75,
        label=f"Calibration data ($R^2$={r2_train:.4f}, RMSE={rmse_train:.4f} {target_unit})",
        zorder=3,
    )

    plt.plot(
        w_line_train,
        pred_line_train["mean"].to_numpy(),
        color=LINE_COLOR,
        linewidth=2,
        label=monitoring_training_label,
        zorder=4,
    )

    plt.xlabel("Wasserstein distance", fontsize=16)
    plt.ylabel(f"{target_name} ({target_unit})", fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{output_prefix}_wassestein_training_uncertainty.pdf"),
        dpi=300,
    )

    plt.close()

    # ------------------------------------------------------------
    # Monitoring calibration plot: validation only
    # Uses training-fitted model and training-derived uncertainty.
    # ------------------------------------------------------------

    w_line_val = np.linspace(y_val.min(), y_val.max(), 200)

    pred_line_val = monitoring_model.get_prediction(
        add_constant(w_line_val)
    ).summary_frame(alpha=0.05)

    plt.figure(figsize=(8, 6))

    plt.fill_between(
        w_line_val,
        pred_line_val["obs_ci_lower"].to_numpy(),
        pred_line_val["obs_ci_upper"].to_numpy(),
        color=PI_COLOR,
        alpha=PI_ALPHA,
        label="95% prediction interval",
        zorder=1,
    )

    plt.fill_between(
        w_line_val,
        pred_line_val["mean_ci_lower"].to_numpy(),
        pred_line_val["mean_ci_upper"].to_numpy(),
        color=CI_COLOR,
        alpha=CI_ALPHA,
        label="95% confidence interval",
        zorder=2,
    )

    plt.scatter(
        y_val,
        x_val,
        alpha=0.75,
        color="green",
        label=f"Validation data ($R^2$={r2_val:.4f}, RMSE={rmse_val:.4f} {target_unit})",
        zorder=3,
    )

    plt.plot(
        w_line_val,
        pred_line_val["mean"].to_numpy(),
        color=LINE_COLOR,
        linewidth=2,
        label=monitoring_validation_label,
        zorder=4,
    )

    plt.xlabel("Wasserstein distance", fontsize=16)
    plt.ylabel(f"{target_name} ({target_unit})", fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{output_prefix}_wassestein_validation_uncertainty.pdf"),
        dpi=300,
    )

    plt.close()

    # Summarize errors and analytical 95% intervals for each split.
    summary = pd.DataFrame([
        {
            "Target": target_name,
            "Method": "Proposed MSI-Wasserstein",
            "Unit": target_unit,
            "Split": "Training",
            "n_samples": len(x_train),
            "R2": r2_train,
            "RMSE": rmse_train,
            "MAE": mae_train,
            "Bias": bias_train,
            "Residual_SD": residual_sd_train,
            "Mean_95CI_width": train_mean_ci_width,
            "Mean_95PI_width": train_mean_pi_width,
            "95PI_coverage": train_pi_coverage,
            "Calibration_intercept": monitoring_model.params[0],
            "Calibration_slope": monitoring_model.params[1],
        },
        {
            "Target": target_name,
            "Method": "Proposed MSI-Wasserstein",
            "Unit": target_unit,
            "Split": "Validation",
            "n_samples": len(x_val),
            "R2": r2_val,
            "RMSE": rmse_val,
            "MAE": mae_val,
            "Bias": bias_val,
            "Residual_SD": residual_sd_val,
            "Mean_95CI_width": val_mean_ci_width,
            "Mean_95PI_width": val_mean_pi_width,
            "95PI_coverage": val_pi_coverage,
            "Calibration_intercept": monitoring_model.params[0],
            "Calibration_slope": monitoring_model.params[1],
        },
    ])

    summary_rounded = summary.copy()

    numeric_cols = summary_rounded.select_dtypes(include=[np.number]).columns
    summary_rounded[numeric_cols] = summary_rounded[numeric_cols].round(4)

    summary_rounded.to_csv(
        os.path.join(output_dir, f"{output_prefix}_uncertainty_summary.csv"),
        index=False,
    )


# ============================================================
# CELL COUNT ANALYSIS
# ============================================================

run_target_analysis(
    target_name="Cell concentration",
    train_csv="./Plots/cellcount/cellcount_wasserstein_training.csv",
    validation_csv="./Plots/cellcount/cellcount_wasserstein_validation.csv",
    target_column="cell_count_x10_5",
    wasserstein_column="wasserstein",
    target_unit=CELL_UNIT_LABEL,
    output_prefix="cellcount",
)


# ============================================================
# MC / TOXICITY ANALYSIS
# ============================================================

run_target_analysis(
    target_name="ELISA-equivalent MC concentration",
    train_csv="./Plots/toxity/toxity_wasserstein_training.csv",
    validation_csv="./Plots/toxity/toxity_wasserstein_validation.csv",
    target_column="toxity_ug_per_L",
    wasserstein_column="wasserstein",
    target_unit=MC_UNIT_LABEL,
    output_prefix="toxity",
)
