"""
Process raw XM CSVs into clean long-format files in data/processed/.

Hourly files: melt Values_Hour01..24 into rows with a datetime column.
Daily files: rename columns, drop redundant fields.

Usage:
    python scripts/process_raw_data.py
"""

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger(__name__)

# Files to process and their output names
hourly_files = [
    ("precio_bolsa_horario.csv",        "precio_bolsa",     "precio_cop_kwh"),
    ("generacion_real_horaria.csv",      "generacion_real",  "generacion_kwh"),
    ("demanda_real_horaria.csv",         "demanda_real",     "demanda_kwh"),
]

daily_files = [
    ("aportes_energia_diario.csv",              "aportes_energia",      "aportes_kwh"),
    ("porcentaje_volumen_util_diario.csv",       "volumen_util_pct",     "volumen_util_pct"),
    ("precio_bolsa_diario_ponderado.csv",        "precio_bolsa_diario",  "precio_cop_kwh"),
    ("demanda_sin_diaria.csv",                   "demanda_sin_diaria",   "demanda_kwh"),
]


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def process_hourly(filename: str, var_name: str, value_col: str) -> pd.DataFrame:
    """Melt hourly wide format into long format with datetime index."""
    filepath = RAW_DIR / filename
    if not filepath.exists():
        log.warning(f"  File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Identify hour columns
    hour_cols = [c for c in df.columns if c.startswith("Values_Hour")]

    # Melt: one row per date becomes 24 rows (one per hour)
    melted = df.melt(
        id_vars=["Date"],
        value_vars=hour_cols,
        var_name="hour_col",
        value_name=value_col,
    )

    # Extract hour number (1-24) and build datetime
    melted["hour"] = melted["hour_col"].str.extract(r"(\d+)$").astype(int)
    melted["datetime"] = (
        pd.to_datetime(melted["Date"])
        + pd.to_timedelta(melted["hour"] - 1, unit="h")
    )

    # Clean up
    result = (
        melted[["datetime", value_col]]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return result


def process_daily(filename: str, var_name: str, value_col: str) -> pd.DataFrame:
    """Clean daily format: rename columns, parse dates."""
    filepath = RAW_DIR / filename
    if not filepath.exists():
        log.warning(f"  File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    result = pd.DataFrame({
        "date": pd.to_datetime(df["Date"]),
        value_col: df["Value"],
    }).sort_values("date").reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Processing raw XM data")
    log.info(f"Input:  {RAW_DIR}")
    log.info(f"Output: {PROC_DIR}")
    log.info("=" * 60)

    # --- Hourly files ---
    for filename, var_name, value_col in hourly_files:
        log.info(f"\n[hourly] {filename}")
        df = process_hourly(filename, var_name, value_col)
        if df is None:
            continue
        out_path = PROC_DIR / f"{var_name}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"  {len(df):>10,} rows  |  {df['datetime'].min()} -> {df['datetime'].max()}")
        log.info(f"  -> {out_path.name}")

    # --- Daily files ---
    for filename, var_name, value_col in daily_files:
        log.info(f"\n[daily]  {filename}")
        df = process_daily(filename, var_name, value_col)
        if df is None:
            continue
        out_path = PROC_DIR / f"{var_name}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"  {len(df):>10,} rows  |  {df['date'].min()} -> {df['date'].max()}")
        log.info(f"  -> {out_path.name}")

    # --- Summary ---
    log.info("\n" + "=" * 60)
    log.info("Output files:")
    for f in sorted(PROC_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        log.info(f"  {f.name:40s} {size_mb:6.1f} MB")


if __name__ == "__main__":
    main()
