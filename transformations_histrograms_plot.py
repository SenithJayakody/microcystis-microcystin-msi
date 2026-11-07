import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression as PLSR
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, StandardScaler
from scipy.stats import wasserstein_distance, gaussian_kde
from scipy.signal import savgol_filter
from sklearn.preprocessing import PowerTransformer

from create_datafiles import wavelengths


def regression_metrics(actual, predicted):
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return r_squared, rmse

# ---------------------------------------------------------
# Helper function for safe file names
# ---------------------------------------------------------
def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^\w\-_\. ]", "_", text)
    text = text.replace(" ", "_")
    return text


def plot_transformation_histograms(
    before_vals,
    sqrt_vals,
    power_vals,
    feature_name,
    title_prefix,
    save_path
):
    # Remove NaN and infinite values before plotting
    before_vals = np.asarray(before_vals)
    before_vals = before_vals[np.isfinite(before_vals)]

    sqrt_vals = np.asarray(sqrt_vals)
    sqrt_vals = sqrt_vals[np.isfinite(sqrt_vals)]

    power_vals = np.asarray(power_vals)
    power_vals = power_vals[np.isfinite(power_vals)]

    datasets = [
        (
            before_vals,
            "MinMax normalized",
            "tab:blue"
        ),
        (
            sqrt_vals,
            "Square-root transformed",
            "tab:orange"
        ),
        (
            power_vals,
            "Yeo-Johnson transformed",
            "tab:green"
        )
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for values, histogram_name, color in datasets:
        if values.size == 0:
            continue

        skewness = pd.Series(values).skew()

        # Transparent histogram
        ax.hist(
            values,
            bins=25,
            density=True,
            alpha=0.30,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            label=f"{histogram_name}"
            # label=f"{histogram_name} (skewness = {skewness:.3f})"
        )

        # Smooth KDE curve
        # KDE requires at least two different values
        if values.size > 1 and np.unique(values).size > 1:
            kde = gaussian_kde(values)

            x_smooth = np.linspace(
                values.min(),
                values.max(),
                500
            )

            ax.plot(
                x_smooth,
                kde(x_smooth),
                color=color,
                linewidth=2.5
            )

    ax.set_xlabel("Feature value", fontsize=17)
    ax.set_ylabel("Density", fontsize=17)
    # ax.set_title(
    #     f"{title_prefix}\nFeature = {feature_name}"
    # )

    ax.legend(
        title="Transformation",title_fontsize=14,
        loc="best",
        fontsize=14
    )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)



csv_file = ".\Datafiles\dataAll_Water_Train.csv"
df = pd.read_csv(csv_file)

# Keep the configured transmittance bands and the cell-count label.
count_column = df.columns[-2]
feature_columns = [f'{wl}_T' for wl in wavelengths]
df = df[feature_columns + [count_column]]
print("After dropping:", df.head())
df = df.sort_values(by=df.columns[-1])
df = df.reset_index(drop=True)
print(df.head())

last_column = df.columns[-1]
value_counts = df[last_column].value_counts()
print(value_counts)

values = df[last_column].unique()
print("Unique values in the last column:", values)

for val, group in df.groupby(last_column).groups.items():
    idx_list = list(group)
    print(f"Label {val} → Row index range: {min(idx_list)}–{max(idx_list)}")

# Extract features and labels
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

feature_names = df.columns[:-1].tolist()

# If labels are numeric-like, convert safely
labels_numeric = pd.to_numeric(labels, errors='coerce')
unique_levels = np.unique(labels_numeric[~np.isnan(labels_numeric)])

# Scale features
scaler = MinMaxScaler()
features_minmax = scaler.fit_transform(features)

# Square-root transformation
features_sqrt = np.sqrt(np.clip(features_minmax, 0, None))

pt = PowerTransformer(method='yeo-johnson', standardize=False)
features_power = pt.fit_transform(features_sqrt)

# ---------------------------------------------------------
# Output folders
# ---------------------------------------------------------
regression_dir = r".\Plots"

root_plot_dir = os.path.join(regression_dir, "transformation_histograms")
overall_plot_dir = os.path.join(root_plot_dir, "overall_all_levels")
by_level_plot_dir = os.path.join(root_plot_dir, "by_contamination_level")

os.makedirs(overall_plot_dir, exist_ok=True)
os.makedirs(by_level_plot_dir, exist_ok=True)

# ---------------------------------------------------------
# 1) Plot for full columns using all samples together
#    (ignoring contamination levels)
# ---------------------------------------------------------
for j, feature_name in enumerate(feature_names):
    before_vals = features_minmax[:, j]
    sqrt_vals = features_sqrt[:, j]
    power_vals = features_power[:, j]

    save_path = os.path.join(
        overall_plot_dir,
        f"overall_feature_{j+1}_{safe_filename(feature_name)}.png"
    )

    plot_transformation_histograms(
        before_vals=before_vals,
        sqrt_vals=sqrt_vals,
        power_vals=power_vals,
        feature_name=feature_name,
        title_prefix="All contamination levels combined",
        save_path=save_path
    )

print(f"Saved overall feature histograms in: {overall_plot_dir}")

# ---------------------------------------------------------
# 2) Plot for each contamination level, for all columns
# ---------------------------------------------------------
for level in unique_levels:
    level_mask = np.isclose(labels_numeric, level)

    level_dir = os.path.join(by_level_plot_dir, f"level_{safe_filename(level)}")
    os.makedirs(level_dir, exist_ok=True)

    for j, feature_name in enumerate(feature_names):
        before_vals = features_minmax[level_mask, j]
        sqrt_vals = features_sqrt[level_mask, j]
        power_vals = features_power[level_mask, j]

        save_path = os.path.join(
            level_dir,
            f"level_{safe_filename(level)}_feature_{j+1}_{safe_filename(feature_name)}.pdf"
        )

        plot_transformation_histograms(
            before_vals=before_vals,
            sqrt_vals=sqrt_vals,
            power_vals=power_vals,
            feature_name=feature_name,
            title_prefix=f"Contamination level = {level}",
            save_path=save_path
        )

print(f"Saved level-wise feature histograms in: {by_level_plot_dir}")
