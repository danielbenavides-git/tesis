"""
Generate CWT scalograms from the hourly real spot price series.

Wavelet and scales follow Benavides & Luna (2024):
- Window: 24 hours (--window day), 168 hours (--window week) or one calendar week (--window calweek).
  With day and week there is one scalogram per day: the weekly window of day t covers day t and the 6 days
  before it, so it never uses later days. The window moves 24 hours from one scalogram to the next, and the
  first 6 days have no weekly scalogram. With calweek there is one scalogram per calendar week, Monday 00:00
  to Sunday 23:00 (168 hours), and consecutive scalograms do not overlap. The days before the first Monday
  and after the last Sunday are left out (1386 weeks, from 2000-01-03 to 2026-07-26).
- Wavelet: Complex Morlet (cmor1.5-1.0)
- Scales: 1 to 128
- Output: magnitude of CWT coefficients, scaled to [0, 1] with the p99.5 value (values above it clipped)

Input is the real price (run 03_process_external_data.py first), either in logs
(--price log) or in COP/kWh (--price linear). The log is the only difference
between the two sets. The window's mean is subtracted before the CWT (Torrence &
Compo, 1998), so the scalogram shows only the shape of the price inside the
window. Without this step the zero-padding edge effect grows with the window's
level and dominates the large scales. With --window day this is each day's mean.
With --window week and calweek it is one mean for the 7 days, so the changes from
one day to the next inside the week stay in the image. The level is saved in the
metadata instead: log_price_mean/log_price_std for --price log and
price_mean/price_std for --price linear. With day and week it is the level of day t
(mean and std of the transformed series on that day); with calweek it is the level
of the whole week (mean and std over its 168 hours).

The metadata has one row per scalogram. With day and week, start_datetime and
end_datetime are the first and last hour of the day the scalogram belongs to (day t),
so the dates and the level columns mean the same in both windows; the weekly metadata
also has window_start_datetime, the first hour of the window (day t - 6). With
calweek, start_datetime is Monday 00:00 and end_datetime is Sunday 23:00 of the week,
and the column unit = week tells src.data.load_context to build one row per week.

Outputs:
    --price log                    data/processed/scalograms/real_price_log/
    --price log --window week      data/processed/scalograms/real_price_log_weekly/
    --price linear                 data/processed/scalograms/real_price/
    --price linear --window week   data/processed/scalograms/real_price_weekly/
    --price log --window calweek   data/processed/scalograms/real_price_log_calweek/

Known limitation: CWT coefficients at scales above about W / (2 * sqrt(2)) fall
outside the cone of influence (Torrence & Compo, 1998) and do not carry reliable
frequency information, where W is the window in hours: about 8 with the daily
window and about 59 with the weekly window.

Usage:
    python scripts/04_generate_scalograms.py --price linear
    python scripts/04_generate_scalograms.py --price log --save-png 50
    python scripts/04_generate_scalograms.py --price log --window week --save-png 50
    python scripts/04_generate_scalograms.py --price log --window calweek --save-png 50
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
SCALO_ROOT = ROOT / "data" / "processed" / "scalograms"
FIG_ROOT = ROOT / "figures" / "scalograms"
PRICE_SETS = {"log": "real_price_log", "linear": "real_price"}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

window_hours = {"day": 24, "week": 168, "calweek": 168}  # hours per scalogram for each --window
set_suffix = {"day": "", "week": "_weekly", "calweek": "_calweek"}  # suffix of the set folder for each --window
STRIDE = 24           # hours between consecutive scalograms (one per day)
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

def transform_price(series: np.ndarray, price: str) -> tuple[np.ndarray, dict, str]:
    """Natural log of the real price (price="log") or the real price as is (price="linear").
    Level is removed per day in create_windows. Returns the series, its params and the
    prefix of the level columns in the metadata."""

    use_log = price == "log"
    if use_log and (series <= 0).any():
        raise ValueError("Real price has non-positive values, log is undefined")

    params = {
        "price_series": "precio_real_cop_kwh",
        "price_set": PRICE_SETS[price],
        "log_transform": use_log,
        "log_function": "log" if use_log else None,
        "daily_demean": True,
    }

    out = np.log(series) if use_log else series.astype(np.float64)
    log.info(f"  {'Log real' if use_log else 'Real'} price: min={out.min():.4f}, max={out.max():.4f}")
    return out, params, "log_price" if use_log else "price"


# ---------------------------------------------------------------------------
# Step 3: Window the series and remove the daily level
# ---------------------------------------------------------------------------

def create_windows(series: np.ndarray, datetimes: np.ndarray, level_prefix: str,
                   window: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Cut the series into windows of `window` hours, one per day, and subtract
    each window's mean. The window of day t ends at the last hour of day t, so
    with window = 24 the windows do not overlap and with window = 168 each one is
    day t and the 6 days before it. Returns the demeaned windows and a metadata
    DataFrame with the level of day t (mean and std of the transformed series)."""

    n_days = len(series) // STRIDE
    usable = n_days * STRIDE
    series = series[:usable]
    datetimes = datetimes[:usable]

    days = series.reshape(n_days, STRIDE)
    dt_days = datetimes.reshape(n_days, STRIDE)
    day_mean = days.mean(axis=1)
    day_std = days.std(axis=1)

    window_days = window // STRIDE
    last_day = np.arange(window_days - 1, n_days)  # day t of each window
    windows = np.lib.stride_tricks.sliding_window_view(series, window)[::STRIDE]
    windows = windows - windows.mean(axis=1, keepdims=True)

    columns = {"window_id": range(len(last_day))}
    if window_days > 1:
        columns["window_start_datetime"] = dt_days[last_day - window_days + 1, 0]
    columns.update({
        "start_datetime": dt_days[last_day, 0],
        "end_datetime": dt_days[last_day, -1],
        f"{level_prefix}_mean": day_mean[last_day],
        f"{level_prefix}_std": day_std[last_day],
    })
    metadata = pd.DataFrame(columns)

    log.info(f"  {len(windows):,} windows of {window}h (stride={STRIDE}h), window mean removed")
    if window_days > 1:
        log.info(f"  The first {window_days - 1} days have no window")
    return windows, metadata


def create_calendar_weeks(series: np.ndarray, datetimes: np.ndarray,
                          level_prefix: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Cut the series into calendar weeks, Monday 00:00 to Sunday 23:00 (168 hours),
    one window per week with no overlap, and subtract each week's mean. The days
    before the first Monday and after the last Sunday are left out. Returns the
    demeaned windows and a metadata DataFrame with one row per week: start_datetime
    (Monday 00:00), end_datetime (Sunday 23:00), the level of the whole week (mean
    and std of the transformed series over its 168 hours) and unit = week."""

    n_days = len(series) // STRIDE
    hours = pd.DatetimeIndex(datetimes[:n_days * STRIDE])
    first = int(np.flatnonzero(hours[::STRIDE].dayofweek == 0)[0])  # first Monday
    n_weeks = (n_days - first) // 7
    start, end = first * STRIDE, (first + 7 * n_weeks) * STRIDE

    weeks = series[start:end].reshape(n_weeks, 7 * STRIDE)
    dt_weeks = hours[start:end].to_numpy().reshape(n_weeks, 7 * STRIDE)
    windows = weeks - weeks.mean(axis=1, keepdims=True)

    metadata = pd.DataFrame({
        "window_id": range(n_weeks),
        "start_datetime": dt_weeks[:, 0],
        "end_datetime": dt_weeks[:, -1],
        f"{level_prefix}_mean": weeks.mean(axis=1),
        f"{level_prefix}_std": weeks.std(axis=1),
        "unit": "week",
    })

    log.info(f"  {n_weeks:,} calendar weeks of 168h (Monday to Sunday, no overlap), week mean removed")
    log.info(f"  From {metadata.start_datetime.iloc[0]} to {metadata.end_datetime.iloc[-1]}; "
             f"{n_days - 7 * n_weeks} days outside full weeks left out")
    return windows, metadata


# ---------------------------------------------------------------------------
# Step 4: Apply CWT to each window
# ---------------------------------------------------------------------------

def compute_scalograms(windows: np.ndarray) -> np.ndarray:
    """Apply CWT to each window. Returns array of shape (N, n_scales, window hours)."""

    n_windows = windows.shape[0]
    n_scales = len(SCALES)
    scalograms = np.empty((n_windows, n_scales, windows.shape[1]), dtype=np.float32)

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
    out_dir: Path,
    scalograms: np.ndarray,
    metadata: pd.DataFrame,
    pre_cwt_params: dict,
    post_cwt_params: dict,
    stride: int = STRIDE,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "scalograms.npy"
    np.save(npy_path, scalograms)
    size_mb = npy_path.stat().st_size / (1024 * 1024)
    log.info(f"  Saved {npy_path.name}: shape={scalograms.shape}, {size_mb:.1f} MB")

    meta_path = out_dir / "scalogram_metadata.csv"
    metadata.to_csv(meta_path, index=False)
    log.info(f"  Saved {meta_path.name}: {len(metadata):,} rows")

    norm_path = out_dir / "normalization_params.json"
    params = {
        "pre_cwt": pre_cwt_params,
        "post_cwt": post_cwt_params,
        "wavelet": WAVELET,
        "scales": SCALES.tolist(),
        "window_hours": scalograms.shape[2],
        "stride_hours": stride,
        "sampling_period_hours": SAMPLING_PERIOD,
    }
    with open(norm_path, "w") as f:
        json.dump(params, f, indent=2)
    log.info(f"  Saved {norm_path.name}")


def save_sample_pngs(fig_dir: Path, scalograms: np.ndarray, metadata: pd.DataFrame, n_samples: int):
    """Save a sample of scalograms as PNG for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Arial'

    fig_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(scalograms)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
    window = scalograms.shape[2]

    for idx in indices:
        row = metadata.iloc[idx]
        start = pd.Timestamp(row.start_datetime).strftime("%Y-%m-%d")
        title = f"Wavelet Scalogram - {start}"
        if "unit" in metadata.columns:
            title = f"Wavelet Scalogram - week {start} to {pd.Timestamp(row.end_datetime):%Y-%m-%d}"
        elif "window_start_datetime" in metadata.columns:
            title = f"Wavelet Scalogram - {pd.Timestamp(row.window_start_datetime):%Y-%m-%d} to {start}"

        fig, ax = plt.subplots(figsize=(6 if window == 24 else 10, 4))
        im = ax.imshow(
            scalograms[idx],
            aspect="auto",
            origin="upper",
            cmap="jet",
            extent=[0, window, SCALES[-1], SCALES[0]],
        )
        ax.set_xlabel("Time (Hour Index)")
        ax.set_ylabel("Scale")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Magnitude")
        fig.tight_layout()

        fname = f"scalogram_{start}.png"
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    log.info(f"  Saved {n_samples} sample PNGs to {fig_dir}")


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
        "--price", choices=sorted(PRICE_SETS), required=True,
        help="log: log real price -> real_price_log/; linear: real price -> real_price/"
    )
    parser.add_argument(
        "--window", choices=sorted(window_hours), default="day",
        help="day: 24 hours (default); week: 168 hours, day t and the 6 days before it -> <set>_weekly/; "
             "calweek: one calendar week (Monday to Sunday), no overlap -> <set>_calweek/"
    )
    parser.add_argument(
        "--save-png", type=int, default=0, metavar="N",
        help="Save N sample scalograms as PNG (default: 0, no PNGs)"
    )
    args = parser.parse_args()
    window = window_hours[args.window]
    set_name = PRICE_SETS[args.price] + set_suffix[args.window]
    out_dir = SCALO_ROOT / set_name

    log.info("=" * 60)
    log.info(f"CWT Scalogram Generation ({set_name})")
    log.info(f"  Wavelet: {WAVELET}")
    log.info(f"  Scales: {SCALES[0]} to {SCALES[-1]} ({len(SCALES)} scales)")
    log.info(f"  Window: {window}h, Stride: {7 * STRIDE if args.window == 'calweek' else STRIDE}h")
    log.info("=" * 60)

    # Step 1
    log.info("\n[1] Loading real price series")
    df = load_price_series()

    # Step 2
    log.info("\n[2] Pre-CWT transform")
    series, pre_cwt_params, level_prefix = transform_price(df.precio_real_cop_kwh.values, args.price)
    if args.window in ("week", "calweek"):
        pre_cwt_params.update({"daily_demean": False, "window_demean": True})
    if args.window == "calweek":
        pre_cwt_params["calendar_weeks"] = True

    # Step 3
    log.info("\n[3] Windowing and demeaning")
    if args.window == "calweek":
        windows, metadata = create_calendar_weeks(series, df.datetime.values, level_prefix)
    else:
        windows, metadata = create_windows(series, df.datetime.values, level_prefix, window)

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
    save_outputs(out_dir, scalograms, metadata, pre_cwt_params, post_cwt_params,
                 stride=7 * STRIDE if args.window == "calweek" else STRIDE)

    if args.save_png > 0:
        log.info(f"\n[8] Saving {args.save_png} sample PNGs")
        save_sample_pngs(FIG_ROOT / set_name, scalograms, metadata, args.save_png)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
