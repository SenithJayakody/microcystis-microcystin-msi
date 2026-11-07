from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

from create_datafiles import wavelengths


csv_file = r'Datafiles\dataAll_Water_Train.csv'
df = pd.read_csv(csv_file)

count_column = df.columns[-2]
toxicity_column = df.columns[-1]

# Use the transmission columns for PCA and keep count/toxicity for labels.
feature_columns = [f'{wl}_T' for wl in wavelengths]

df = df[feature_columns + [count_column, toxicity_column]]
df = df.sort_values(by=count_column).reset_index(drop=True)

print("Selected PCA features and labels:")
print(df.head())

value_counts = df[count_column].value_counts().sort_index()
print(value_counts)
print("Unique cell count values:", df[count_column].unique())

for val, group in df.groupby(count_column).groups.items():
    idx_list = list(group)
    tox_values = df.loc[idx_list, toxicity_column].unique()
    tox_text = ", ".join(f"{tox:g}" for tox in tox_values)
    print(f"CC={val:g}, Tox={tox_text} -> Row index range: {min(idx_list)}-{max(idx_list)}")

# Extract features and labels.
features = df[feature_columns].values
count_labels = df[count_column].values
toxicity_labels = df[toxicity_column].values

# Scale features.
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.sqrt(np.clip(features_scaled, 0, None))

pt = PowerTransformer(method='yeo-johnson', standardize=False)
features_scaled = pt.fit_transform(features_scaled)

# Apply PCA (retain 99% variance).
pca = PCA(n_components=0.99)
features_pca = pca.fit_transform(features_scaled)
print(f"PCA reduced features from {features.shape[1]} to {features_pca.shape[1]} dimensions")
features = features_pca

# Plot PCA 1 vs PCA 2.
fig, ax = plt.subplots(figsize=(8, 4))
unique_labels = np.sort(df[count_column].unique())
colors = plt.get_cmap('tab20', len(unique_labels))
legend_handles = []

for i, level in enumerate(unique_labels):
    mask = count_labels == level
    toxicity_value = toxicity_labels[mask][0]
    label = f'CC={level:g}, MCC={toxicity_value:g}'
    ax.scatter(
        features[mask, 0],
        features[mask, 1],
        c=[colors(i)],
        s=25,
    )
    legend_handles.append(
        Line2D([0], [0], color=colors(i), marker='o', linewidth=1.5, markersize=4, label=label)
    )

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
# ax.set_title('PCA 1 vs PCA 2 by Cell Count and Toxicity')
ax.grid(True)
ax.legend(
    handles=legend_handles,
    title='Cell Count (x 10^5), MC concentration (ug/L )',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    frameon=True,
    fontsize=8,
    title_fontsize=9
)

fig.tight_layout()
output_dir = Path("Plots") / "pca_plots"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(
    output_dir / "pca_plot.png",
    dpi=300,
    bbox_inches='tight'
)
fig.savefig(
    output_dir / "pca_plot.pdf",
    dpi=300,
    bbox_inches='tight'
)
plt.show()
