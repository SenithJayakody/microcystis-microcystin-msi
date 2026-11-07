import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression as PLSR
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, StandardScaler
from scipy.stats import wasserstein_distance
from scipy.signal import savgol_filter
from sklearn.preprocessing import PowerTransformer
from pathlib import Path


def regression_metrics(actual, predicted):
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return r_squared, rmse


output_dir = Path("Plots") / "toxity"
output_dir.mkdir(parents=True, exist_ok=True)

csv_file = r'Datafiles\dataAll_Water_Train.csv'
df = pd.read_csv(csv_file)

df = df.drop(df.columns[-2], axis=1)
print("After dropping:", df.head())
df = df.sort_values(by=df.columns[-1], kind='stable')  # Use stable sort to maintain order of equal elements
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

# # Create a mapping from labels to digits
# label_to_digit = {label: digit for digit, label in enumerate(np.unique(labels))}
# print("Label to digit mapping:", label_to_digit)

# # Convert labels to digits
# labels = np.array([label_to_digit[label] for label in labels])
# print("Discretized labels (first 10):", labels[:10])
# print("Unique discretized labels:", np.unique(labels))

# Scale features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

features_scaled = np.sqrt(np.clip(features_scaled, 0, None))#1e-6
# features_scaled = -np.log10(np.clip(features_scaled, 1e-6, None))#1e-6
# features_scaled = savgol_filter(features_scaled, window_length=5, polyorder=2)

pt = PowerTransformer(method='yeo-johnson',standardize=False)
features_scaled = pt.fit_transform(features_scaled)
# features_scaled = savgol_filter(features_scaled, window_length=5, polyorder=2)

# Apply PCA (retain 99% variance)
pca = PCA(n_components=0.99)
features_pca = pca.fit_transform(features_scaled)
print(f"PCA reduced features from {features.shape[1]} to {features_pca.shape[1]} dimensions")
features = features_pca

# Plot PCA 1 vs PCA 2
plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap('tab20', len(unique_labels))  # 'tab20' provides 20 distinct colors

for i, level in enumerate(unique_labels):
    mask = (labels == level)
    plt.scatter(features[mask, 0], features[mask, 1], 
                c=[colors(i)], label=f'Level {level}')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA 1 vs PCA 2 by toxity Levels')
plt.legend()
plt.grid(True)
# plt.savefig("C:\\0A___________________________________\\MSI\\MSI_Microcystis\\July_2025_Test_M8\\Regression\\pca_1_vs_2_by_toxity.png", dpi=300)
plt.show()




# Setup
numOfLevel = 11
numOfSamples = 8
wasserstein_values = np.zeros([numOfLevel * numOfSamples])
print(wasserstein_values.shape)

# Iterate through PCA bands
for band in range(features.shape[1]):
    wasserstein_sum = np.zeros([numOfLevel * numOfSamples])

    for i in range(numOfLevel * numOfSamples):
        reference = features[0:100, band]
        level = features[i*100:i*100+100, band]

        # Wasserstein distance
        wd = wasserstein_distance(reference, level)
        wasserstein_sum[i] = wd

    print(wasserstein_sum)
    wasserstein_values += wasserstein_sum
    print(wasserstein_values)

print(wasserstein_values)


values = [0, 1.26, 2.19, 2.82, 3.54, 4.21, 5.46, 5.84, 6.53, 6.76, 7.56] #toxity(ug/L)

x = []
for i in range(numOfLevel):
    x.extend([values[i % len(values)]] * numOfSamples)

# Save training cell count + Wasserstein to CSV
train_out = pd.DataFrame({"toxity_ug_per_L": x,"wasserstein": wasserstein_values})
train_out.to_csv(output_dir / "toxity_wasserstein_training.csv",
        index=False)


# Fit C_MC = beta0 + beta1 * W (paper Eq. 20): MC concentration regressed on Wasserstein
degree = 1
coefficients = np.polyfit(wasserstein_values, x, degree)   # returns [beta1, beta0]
polynomial = np.poly1d(coefficients)
beta1, beta0 = coefficients
fit_label = f"y = {beta0:.4f} {beta1:+.4f}x"

w_fit = np.linspace(wasserstein_values.min(), wasserstein_values.max(), 100)
c_fit = polynomial(w_fit)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(wasserstein_values, x, marker="o", label="Samples")
plt.plot(w_fit, c_fit, color="red", label=f"Fitted line: {fit_label}")
plt.xlabel("Wasserstein Distance")
plt.ylabel("Toxity (ug/L)")
plt.legend()
plt.grid(True)
plt.title("Wasserstein Distance vs Toxity")
plt.savefig(output_dir / "toxity_wasserstein_training.png", dpi=300)
plt.show()


# -------------------------------
# Toxity R^2/RMSE
# -------------------------------
toxity_pred = polynomial(wasserstein_values)
toxity_r2, toxity_rmse = regression_metrics(np.array(x), toxity_pred)

print("Fitted model (Eq. 20):", fit_label)
print("Toxity R^2 Value (Training):", round(toxity_r2, 4))
print("Toxity RMSE (Training):", toxity_rmse)


# Validating the prediction

# Load the dataset from the CSV file
dataset_val = pd.read_csv(
    r'Datafiles\dataAll_Water_Validation.csv'
)
dataset_val = dataset_val.drop(dataset_val.columns[-2], axis=1)
print("After dropping:", dataset_val.head())
dataset_val = dataset_val.sort_values(by=dataset_val.columns[-1], kind = 'stable')
print(dataset_val.columns[-1])
dataset_val = dataset_val.reset_index(drop=True)
print(dataset_val.head())

# last_column = dataset_val.columns[-1]
# value_counts = dataset_val[last_column].value_counts()
# print(value_counts)

values = dataset_val[last_column].unique()
print("Unique values in the last column:", values)

# for val, group in dataset_val.groupby(last_column).groups.items():
#     idx_list = list(group)
#     print(f"Label {val} → Row index range: {min(idx_list)}–{max(idx_list)}")


features_val = dataset_val.iloc[:, :-1].values
print("Number of features:", features_val.shape[1])
y_val = dataset_val.iloc[:, -1].values

features_val_scaled = scaler.transform(features_val)
features_val_scaled = np.sqrt(np.clip(features_val_scaled, 0, None))
features_val_scaled = pt.transform(features_val_scaled)

features_val_pca = pca.transform(features_val_scaled)
print("Validation features shape after PCA:", features_val_pca.shape)

features_val = features_val_pca


numOfLevel = 6
numOfSamples = 8

wasserstein_values_val = np.zeros([numOfLevel * numOfSamples])
print("Wasserstein distance vector shape:", wasserstein_values_val.shape)

# Iterate through all columns (bands x1 to x13)
for band in range(features_val.shape[1]):
    wasserstein_sum_val = np.zeros([numOfLevel * numOfSamples])

    for i in range(numOfLevel * numOfSamples):
        reference_val = features[0:100, band]   # from training features
        level_val = features_val[i*100:i*100+100, band]

        # Compute Wasserstein distance
        w_dist_val = wasserstein_distance(reference_val, level_val)

        wasserstein_sum_val[i] = w_dist_val

    # Accumulate across bands
    print(wasserstein_sum_val)
    wasserstein_values_val += wasserstein_sum_val
    print(f"Band {band+1} accumulated values:", wasserstein_values_val)

# wasserstein_values_val /= features.shape[1]
print("Final Wasserstein values:", wasserstein_values_val)

# -------------------------------
# Scatter Plot
# -------------------------------
values_val = [0.62, 1.9, 3.11, 4.94, 6.29, 6.98] #toxity(ug/L)
x_val = []
for i in range(numOfLevel):
    x_val.extend([values_val[i % len(values_val)]] * numOfSamples)

# Save validation cell count + Wasserstein to CSV
val_out = pd.DataFrame({"toxity_ug_per_L": x_val, "wasserstein": wasserstein_values_val})
val_out.to_csv(output_dir / "toxity_wasserstein_validation.csv",
    index=False)

plt.figure(figsize=(8, 6))
plt.scatter(wasserstein_values_val, x_val, color="green", marker='o', label='Validation set')
plt.plot(w_fit, c_fit, color="red", label=f"Calibrated line: {fit_label}")
plt.xlabel("Wasserstein Distance")
plt.ylabel("Toxity (ug/L)")
plt.legend()
plt.title("Wasserstein Distance vs Toxity")
plt.savefig(output_dir / "toxity_wasserstein_validation.png", dpi=300)
plt.show()

# -------------------------------
# Toxity R^2/RMSE (calibration model applied without refitting)
# -------------------------------
toxity_pred_val = polynomial(wasserstein_values_val)
toxity_r2_val, toxity_rmse_val = regression_metrics(
    np.array(x_val),
    toxity_pred_val,
)

print("Toxity R^2 Value (Validation):", round(toxity_r2_val, 4))
print("Toxity RMSE (Validation):", toxity_rmse_val)

# metrics_out = pd.DataFrame(
#     [
#         {
#             "dataset": "training",
#             "wasserstein_r2": wasserstein_r2,
#             "wasserstein_rmse": wasserstein_rmse,
#             "toxity_r2": toxity_r2,
#             "toxity_rmse": toxity_rmse,
#         },
#         {
#             "dataset": "validation",
#             "wasserstein_r2": wasserstein_r2_val,
#             "wasserstein_rmse": wasserstein_rmse_val,
#             "toxity_r2": toxity_r2_val,
#             "toxity_rmse": toxity_rmse_val,
#         },
#     ]
# )
# metrics_out.to_csv(
#     r"C:\0A___________________________________\MSI\MSI_Microcystis\July_2025_Test_M8\Regression\toxity_regression_metrics.csv",
#     index=False,
# )



