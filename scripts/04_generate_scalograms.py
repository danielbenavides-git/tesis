"""
Generate CWT scalograms from the hourly real spot price series.

Window, wavelet and scales follow Benavides & Luna (2024):
- Window: 24 hours (one scalogram per day)
- Wavelet: Complex Morlet (cmor1.5-1.0)
- Scales: 1 to 128
- Output: magnitude of CWT coefficients, scaled to [0, 1] with the p99.5 value (values above it clipped)

Input is the log real price (run 03_process_external_data.py first). Each day's
mean is subtracted before the CWT (Torrence & Compo, 1998), so the scalogram
shows only the intraday shape. Without this step the zero-padding edge effect
grows with the day's level and dominates the large scales. The daily level
(mean and std of the log real price) is saved in the metadata instead.

Known limitation: with a 24-hour window, CWT coefficients at scales above ~8
fall outside the cone of influence (Torrence & Compo, 1998) and do not carry
reliable frequency information.

Usage:
    python scripts/04_generate_scalograms.py
    python scripts/04_generate_scalograms.py --save-png 50
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed" / "XM"
SCALO_DIR = ROOT / "data" / "processed" / "scalograms"
FIG_DIR = ROOT / "figures" / "scalograms"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

WINDOW = 24           # hours per scalogram
STRIDE = 24           # non-overlapping daily windows
WAVELET = "cmor1.5-1.0"
SCALES = np.arange(1, 129)  # 128 scales, matching Benavides & Luna Figure 5
SAMPLING_PERIOD = 1.0       # 1 hour


# ---------------------------------------------------------------------------
# Step 1: Load price series
# ---------------------------------------------------------------------------

def load_price_series() -> pd.DataFrame:
    path = PROC_DIR / "precio_bolsa_real.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    log.info(f"Loaded {len(df):,} rows from {path.name}")
    log.info(f"  Range: {df.datetime.min()} to {df.datetime.max()}")
    log.info(f"  Nulls: {df.precio_real_cop_kwh.isna().sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Pre-CWT transform
# ---------------------------------------------------------------------------

def log_price(series: np.ndarray) -> tuple[np.ndarray, dict]:
    """Natural log of the real price. Level is removed per day in create_windows."""

    if (series <= 0).any():
        raise ValueError("Real price has non-positive values, log is undefined")

    params = {
        "price_series": "precio_real_cop_kwh",
        "log_transform": True,
        "log_function": "log",
        "daily_demean": True,
    }

    log.info(f"  Log real price: min={np.log(series.min()):.4f}, max={np.log(series.max()):.4f}")
    return np.log(series), params


# ---------------------------------------------------------------------------
# Step 3: Window the series and remove the daily level
# ---------------------------------------------------------------------------

def create_windows(series: np.ndarray, datetimes: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Segment the series into non-overlapping 24-hour windows and subtract each
    window's mean. Returns the demeaned windows and a metadata DataFrame with
    the daily level (mean and std of the log real price)."""

    n_windows = len(series) // STRIDE
    usable = n_windows * STRIDE
    series = series[:usable]
    datetimes = datetimes[:usable]

    windows = series.reshape(n_windows, WINDOW)
    dt_windows = datetimes.reshape(n_windows, WINDOW)

    day_mean = windows.mean(axis=1)
    day_std = windows.std(axis=1)
    windows = windows - day_mean[:, np.newaxis]

    metadata = pd.DataFrame({
        "window_id": range(n_windows),
        "start_datetime": dt_windows[:, 0],
        "end_datetime": dt_windows[:, -1],
        "log_price_mean": day_mean,
        "log_price_std": day_std,
    })

    log.info(f"  {n_windows:,} windows of {WINDOW}h (stride={STRIDE}h), daily mean removed")
    return windows, metadata


# ---------------------------------------------------------------------------
# Step 4: Apply CWT to each window
# ---------------------------------------------------------------------------

def compute_scalograms(windows: np.ndarray) -> np.ndarray:
    """Apply CWT to each window. Returns array of shape (N, n_scales, WINDOW)."""

    n_windows = windows.shape[0]
    n_scales = len(SCALES)
    scalograms = np.empty((n_windows, n_scales, WINDOW), dtype=np.float32)

    for i in range(n_windows):
        coeffs, _ = pywt.cwt(windows[i], SCALES, WAVELET, sampling_period=SAMPLING_PERIOD)
        scalograms[i] = np.abs(coeffs).astype(np.float32)

        if (i + 1) % 2000 == 0 or i == n_windows - 1:
            log.info(f"  CWT: {i + 1:,}/{n_windows:,}")

    return scalograms


# ---------------------------------------------------------------------------
# Step 5: Post-CWT normalization
# ---------------------------------------------------------------------------

UPPER_PERCENTILE = 99.5

def normalize_scalograms(scalograms: np.ndarray) -> tuple[np.ndarray, dict]:
    """Scale to [0, 1] with the global minimum and a high percentile, clipping
    values above it. Using the maximum lets a few extreme days compress all
    other images into a narrow dark range."""

    global_min = float(scalograms.min())
    upper = float(np.percentile(scalograms, UPPER_PERCENTILE))
    n_clipped = int((scalograms > upper).sum())
    scalograms = np.clip((scalograms - global_min) / (upper - global_min), 0.0, 1.0)

    params = {
        "method": "min_percentile_clip",
        "global_min": global_min,
        "upper_percentile": UPPER_PERCENTILE,
        "upper_value": upper,
        "clipped_fraction": n_clipped / scalograms.size,
    }

    log.info(f"  Normalization: [{global_min:.6f}, p{UPPER_PERCENTILE}={upper:.6f}] -> [0, 1]")
    log.info(f"  Clipped {n_clipped:,} values ({n_clipped / scalograms.size:.3%})")
    return scalograms.astype(np.float32), params


# ---------------------------------------------------------------------------
# Step 6: Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    scalograms: np.ndarray,
    metadata: pd.DataFrame,
    pre_cwt_params: dict,
    post_cwt_params: dict,
):
    SCALO_DIR.mkdir(parents=True, exist_ok=True)

    npy_path = SCALO_DIR / "scalograms.npy"
    np.save(npy_path, scalograms)
    size_mb = npy_path.stat().st_size / (1024 * 1024)
    log.info(f"  Saved {npy_path.name}: shape={scalograms.shape}, {size_mb:.1f} MB")

    meta_path = SCALO_DIR / "scalogram_metadata.csv"
    metadata.to_csv(meta_path, index=False)
    log.info(f"  Saved {meta_path.name}: {len(metadata):,} rows")

    norm_path = SCALO_DIR / "normalization_params.json"
    params = {
        "pre_cwt": pre_cwt_params,
        "post_cwt": post_cwt_params,
        "wavelet": WAVELET,
        "scales": SCALES.tolist(),
        "window_hours": WINDOW,
        "stride_hours": STRIDE,
        "sampling_period_hours": SAMPLING_PERIOD,
    }
    with open(norm_path, "w") as f:
        json.dump(params, f, indent=2)
    log.info(f"  Saved {norm_path.name}")


def save_sample_pngs(scalograms: np.ndarray, metadata: pd.DataFrame, n_samples: int):
    """Save a sample of scalograms as PNG for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Arial'

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(scalograms)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    for idx in indices:
        row = metadata.iloc[idx]
        start = pd.Timestamp(row.start_datetime).strftime("%Y-%m-%d")

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            scalograms[idx],
            aspect="auto",
            origin="upper",
            cmap="jet",
            extent=[0, WINDOW, SCALES[-1], SCALES[0]],
        )
        ax.set_xlabel("Time (Hour Index)")
        ax.set_ylabel("Scale")
        ax.set_title(f"Wavelet Scalogram - {start}")
        fig.colorbar(im, ax=ax, label="Magnitude")
        fig.tight_layout()

        fname = f"scalogram_{start}.png"
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)

    log.info(f"  Saved {n_samples} sample PNGs to {FIG_DIR}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(scalograms: np.ndarray):
    assert not np.isnan(scalograms).any(), "NaN values in scalograms"
    assert not np.isinf(scalograms).any(), "Inf values in scalograms"
    assert scalograms.min() >= 0.0, f"Min below 0: {scalograms.min()}"
    assert scalograms.max() <= 1.0, f"Max above 1: {scalograms.max()}"
    log.info(f"  Validation passed: shape={scalograms.shape}, range=[0, 1], no NaN/Inf")

    freqs = pywt.scale2frequency(WAVELET, SCALES, precision=8) / SAMPLING_PERIOD
    periods = 1.0 / freqs
    log.info(f"  Frequency range: {freqs.min():.4f} to {freqs.max():.4f} cycles/hour")
    log.info(f"  Period range: {periods.min():.1f} to {periods.max():.1f} hours")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate CWT scalograms from the real spot price")
    parser.add_argument(
        "--save-png", type=int, default=0, metavar="N",
        help="Save N sample scalograms as PNG (default: 0, no PNGs)"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("CWT Scalogram Generation")
    log.info(f"  Wavelet: {WAVELET}")
    log.info(f"  Scales: {SCALES[0]} to {SCALES[-1]} ({len(SCALES)} scales)")
    log.info(f"  Window: {WINDOW}h, Stride: {STRIDE}h")
    log.info("=" * 60)

    # Step 1
    log.info("\n[1] Loading real price series")
    df = load_price_series()

    # Step 2
    log.info("\n[2] Pre-CWT transform")
    log_series, pre_cwt_params = log_price(df.precio_real_cop_kwh.values)

    # Step 3
    log.info("\n[3] Windowing and daily demeaning")
    windows, metadata = create_windows(log_series, df.datetime.values)

    # Step 4
    log.info("\n[4] Computing CWT")
    scalograms = compute_scalograms(windows)

    # Step 5
    log.info("\n[5] Post-CWT normalization")
    scalograms, post_cwt_params = normalize_scalograms(scalograms)

    # Validation
    log.info("\n[6] Validation")
    validate(scalograms)

    # Step 7
    log.info("\n[7] Saving outputs")
    save_outputs(scalograms, metadata, pre_cwt_params, post_cwt_params)

    if args.save_png > 0:
        log.info(f"\n[8] Saving {args.save_png} sample PNGs")
        save_sample_pngs(scalograms, metadata, args.save_png)

    log.info("\nDone.")


if __name__ == "__main__":
    main()

# Run the following command to generate scalograms:
# python 04_generate_scalograms.py --save-png 10