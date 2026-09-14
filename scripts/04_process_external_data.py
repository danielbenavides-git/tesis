"""
Process external data sources and generate derived series.

1. NOAA ONI: parse oni.ascii.txt into a monthly CSV with ENSO phase labels.
2. DANE IPC: use the cumulative CPI index to deflate spot price series
   to constant pesos (base = most recent available month).

Inputs:
    data/raw/NOAA/oni.ascii.txt
    data/raw/DANE/ipc.csv
    data/processed/XM/precio_bolsa.csv
    data/processed/XM/precio_bolsa_diario.csv

Outputs:
    data/processed/NOAA/oni.csv
    data/processed/XM/precio_bolsa_real.csv
    data/processed/XM/precio_bolsa_diario_real.csv

Usage:
    python scripts/04_process_external_data.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# Map each 3-month season code to its center month
SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def process_oni():
    """Parse NOAA ONI text file into a clean monthly CSV."""
    src = RAW / "NOAA" / "oni.ascii.txt"
    out = PROC / "NOAA" / "oni.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading {src}")
    df = pd.read_fwf(src)
    df.columns = ["season", "year", "sst_total", "oni"]

    df["month"] = df["season"].map(SEASON_TO_MONTH)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )

    # Classify ENSO phase using standard ONI thresholds
    conditions = [
        df["oni"] >= 0.5,
        df["oni"] <= -0.5,
    ]
    choices = ["El Nino", "La Nina"]
    df["phase"] = np.select(conditions, choices, default="Neutral")

    result = df[["date", "year", "month", "season", "oni", "phase"]].copy()
    result = result.sort_values("date").reset_index(drop=True)
    result.to_csv(out, index=False)
    log.info(f"Saved {out}: {len(result)} rows, {result['date'].min()} to {result['date'].max()}")
    return result


def process_deflation():
    """Deflate spot price series using DANE CPI index."""
    ipc_path = RAW / "DANE" / "ipc.csv"
    hourly_path = PROC / "XM" / "precio_bolsa.csv"
    daily_path = PROC / "XM" / "precio_bolsa_diario.csv"

    log.info(f"Reading IPC from {ipc_path}")
    ipc = pd.read_csv(ipc_path)
    ipc["date"] = pd.to_datetime(ipc["date"], format="%Y-%m")
    ipc["year_month"] = ipc["date"].dt.to_period("M")

    # Reference period: most recent month in the IPC series
    ipc_ref = ipc["ipc_base_2000_01"].iloc[-1]
    ref_date = ipc["date"].iloc[-1].strftime("%Y-%m")
    log.info(f"Reference period: {ref_date}, IPC = {ipc_ref:.2f}")

    # Build a year-month lookup table
    ipc_lookup = ipc.set_index("year_month")["ipc_base_2000_01"]

    # --- Deflate daily prices ---
    log.info(f"Reading daily prices from {daily_path}")
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    daily["year_month"] = daily["date"].dt.to_period("M")
    daily["ipc"] = daily["year_month"].map(ipc_lookup)

    # Rows before IPC coverage start (Jan 2000) will have NaN
    n_missing = daily["ipc"].isna().sum()
    if n_missing > 0:
        log.warning(f"  {n_missing} daily rows outside IPC coverage, will be dropped")

    daily = daily.dropna(subset=["ipc"])
    daily["precio_real_cop_kwh"] = daily["precio_cop_kwh"] * (ipc_ref / daily["ipc"])

    daily_out = PROC / "XM" / "precio_bolsa_diario_real.csv"
    daily[["date", "precio_cop_kwh", "precio_real_cop_kwh"]].to_csv(daily_out, index=False)
    log.info(f"Saved {daily_out}: {len(daily)} rows")

    # --- Deflate hourly prices ---
    log.info(f"Reading hourly prices from {hourly_path}")
    hourly = pd.read_csv(hourly_path, parse_dates=["datetime"])
    hourly["year_month"] = hourly["datetime"].dt.to_period("M")
    hourly["ipc"] = hourly["year_month"].map(ipc_lookup)

    n_missing = hourly["ipc"].isna().sum()
    if n_missing > 0:
        log.warning(f"  {n_missing} hourly rows outside IPC coverage, will be dropped")

    hourly = hourly.dropna(subset=["ipc"])
    hourly["precio_real_cop_kwh"] = hourly["precio_cop_kwh"] * (ipc_ref / hourly["ipc"])

    hourly_out = PROC / "XM" / "precio_bolsa_real.csv"
    hourly[["datetime", "precio_cop_kwh", "precio_real_cop_kwh"]].to_csv(hourly_out, index=False)
    log.info(f"Saved {hourly_out}: {len(hourly)} rows")

    return ref_date, ipc_ref


def main():
    log.info("=" * 60)
    log.info("Processing external data sources")
    log.info("=" * 60)

    log.info("-" * 40)
    log.info("1. NOAA ONI")
    process_oni()

    log.info("-" * 40)
    log.info("2. DANE IPC deflation")
    ref_date, ipc_ref = process_deflation()

    log.info("=" * 60)
    log.info("Done")
    log.info(f"  Deflation reference: {ref_date} (IPC = {ipc_ref:.2f})")
    log.info(f"  All real prices expressed in {ref_date} pesos")


if __name__ == "__main__":
    main()
