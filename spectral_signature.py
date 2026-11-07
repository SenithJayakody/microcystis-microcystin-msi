from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_datafiles import wavelengths

# create_datafiles.py writes 100 consecutive rows (10x10 blocks) per sample.
ROWS_PER_SAMPLE = 100

COLOR_LIST = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd', '#31a354', '#756bb1', '#636363', '#e6550d', '#9c9ede', '#8ca252', '#bd9e39', '#ad494a', '#a55194', '#6baed6', '#74c476', '#9e9ac8', '#bcbddc', '#969696']


def plot_spectral_signature(df, out_png):
    """Save one mean spectrum per sample, with one legend entry per count/toxicity pair."""
    # Use the same bands as CSV generation, including when reading older CSVs.
    wl_cols = [f"{wl}_T" for wl in wavelengths]
    missing = [c for c in wl_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing transmittance columns: {missing}")
    count_columns = [c for c in df.columns if c.startswith("Count(")]
    if not wl_cols or len(count_columns) != 1 or "Toxicity(ug/L)" not in df.columns:
        raise ValueError("Expected wavelength_T, Count(...), and Toxicity(ug/L) columns")
    if df.empty:
        raise ValueError("Cannot plot an empty CSV")
    X = df[wl_cols].to_numpy(dtype=float)
    labels = df[[count_columns[0], "Toxicity(ug/L)"]].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(labels).all():
        raise ValueError("Spectra and labels must contain finite numeric values")
    if len(X) % ROWS_PER_SAMPLE != 0:
        raise ValueError(f"Row count {len(X)} is not a multiple of {ROWS_PER_SAMPLE} blocks per sample")
    sample_id = np.arange(len(X)) // ROWS_PER_SAMPLE
    uniq_vals, group_id = np.unique(labels, axis=0, return_inverse=True)
    group_id = group_id.ravel()
    gmin = X.min(axis=0)
    gmax = X.max(axis=0)
    grng = np.where((gmax - gmin) == 0, 1.0, (gmax - gmin))
    Xn = (X - gmin) / grng
    x = np.array([float(c.split("_")[0]) for c in wl_cols])
    x_labels = [str(int(v)) for v in x]
    fig, ax = plt.subplots(figsize=(15, 8))
    order = np.lexsort((uniq_vals[:, 1], uniq_vals[:, 0]))
    for idx_plot, gid in enumerate(order):
        in_group = group_id == gid
        # Mean normalized intensity of the 100 blocks of each sample in this level.
        sample_means = np.array([
            Xn[in_group & (sample_id == s), :].mean(axis=0)
            for s in np.unique(sample_id[in_group])
        ])
        cc_val, tox_val = uniq_vals[gid]
        color = COLOR_LIST[idx_plot % len(COLOR_LIST)]
        # One curve per sample. Label the first curve of each level.
        lines = ax.plot(x, sample_means.T, color=color, linewidth=1.2, alpha=0.8)
        lines[0].set_label(f"CC={cc_val:g}, MCC={tox_val:g}")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xticks(x, x_labels, rotation=45, fontsize=15)
    ax.set_xlim(x.min() - 20, x.max() + 20)
    ax.set_xlabel("Wavelength/nm", fontsize=18)
    ax.set_ylabel("Normalized Transmitted Intensity", fontsize=18)
    ax.grid(alpha=0.4)
    ax.legend(
        title="Cell Count (x 10^5), MC concentration (ug/L )",
        title_fontsize=13,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return len(np.unique(sample_id)), len(uniq_vals)


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "Plots" / "Spectral Signature"
    input_path = root / "Datafiles" / "dataAll_Water_Train.csv"
    output_path = output_dir / "Spectral_Signature_CC_Tox_Transmittance_TRAIN.png"
    df = pd.read_csv(input_path)
    rows, groups = plot_spectral_signature(df, output_path)
    print(f"Saved {output_path}: {rows} curves, {groups} legend entries")


if __name__ == "__main__":
    main()
