"""
Download historical data from XM (Colombian electricity system operator).

Uses pydataxm (Sinergox API). The library handles monthly chunking
and async requests internally. No API key required.

Requirements:
    pip install pydataxm pandas

Usage:
    # Step 1: discover available MetricIds and save catalog
    python scripts/download_xm_data.py --discover

    # Step 2: download all configured variables
    python scripts/download_xm_data.py --download

    # Both
    python scripts/download_xm_data.py --all
"""

import argparse
import logging
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
from pydataxm.pydataxm import ReadDB

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger(__name__)

# MEM operates since 1995-07-20. Hourly price data typically from ~2000.
START = dt.date(2000, 1, 1)
END = dt.date(2026, 7, 31)


# ---------------------------------------------------------------------------
# Variables to download
# ---------------------------------------------------------------------------
# "PrecBolsNaci" + "Sistema" confirmed in pydataxm.py source (line 206).
# "Gene" + "Recurso" confirmed in README example.
#
# The remaining MetricIds follow XM's abbreviated naming convention.
# RUN --discover FIRST to validate all MetricIds against the live catalog
# and correct any that don't match.
#
# If a MetricId is wrong, the library prints:
#   "No existe la metrica <X>" and returns an empty DataFrame.

VARIABLES = [
    # --- CORE: price series for CWT scalograms ---
    {
        "metric_id": "PrecBolsNaci",
        "entity": "Sistema",
        "filename": "precio_bolsa_nacional_horario.csv",
        "desc": "Precio de Bolsa Nacional (hourly, $/kWh)",
    },
    # --- VALIDATION: hydrology ---
    {
        "metric_id": "VoluUtilDiarwordsEner",
        "entity": "Sistema",
        "filename": "volumen_util_diario.csv",
        "desc": "Volumen Util Diario del SIN (daily, GWh)",
    },
    {
        "metric_id": "AporEner",
        "entity": "Sistema",
        "filename": "aportes_diarios.csv",
        "desc": "Aportes Diarios del SIN (daily, GWh)",
    },
    # --- VALIDATION: generation ---
    {
        "metric_id": "Gene",
        "entity": "Sistema",
        "filename": "generacion_real_total.csv",
        "desc": "Generacion Real Total (hourly, kWh)",
    },
    # --- VALIDATION: demand ---
    {
        "metric_id": "DemaSIN",
        "entity": "Sistema",
        "filename": "demanda_sin.csv",
        "desc": "Demanda del SIN (hourly, kWh)",
    },
    # --- VALIDATION: scarcity price ---
    {
        "metric_id": "PrecEscwordsActi",
        "entity": "Sistema",
        "filename": "precio_escasez_activacion.csv",
        "desc": "Precio de Escasez de Activacion (daily, $/kWh)",
    },
    # --- EXTRA: max offer price ---
    {
        "metric_id": "MaxPrecOfe",
        "entity": "Sistema",
        "filename": "maximo_precio_oferta.csv",
        "desc": "Maximo Precio de Oferta (daily)",
    },
]


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------

def discover(api: ReadDB):
    """Fetch the full metrics catalog and save to CSV for reference."""
    log.info("Fetching metrics catalog from ListadoMetricas...")
    catalog = api.get_collections()
    out = RAW_DIR / "catalogo_metricas_xm.csv"
    catalog.to_csv(out, index=False)
    log.info(f"Saved {len(catalog)} metrics to {out}")
    log.info(f"Columns: {list(catalog.columns)}")

    # Validate configured MetricIds
    if "MetricId" in catalog.columns:
        log.info("")
        log.info("Validating configured MetricIds:")
        all_ids = set(catalog["MetricId"].values)
        for var in VARIABLES:
            mid = var["metric_id"]
            found = mid in all_ids
            status = "OK" if found else "NOT FOUND"
            log.info(f"  {mid:25s} -> {status}")
            if not found:
                # Search for similar
                similar = [x for x in all_ids if mid[:4].lower() in x.lower()]
                if similar:
                    log.info(f"    Similar: {similar[:5]}")

    return catalog


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_one(api: ReadDB, var: dict):
    """Download a single variable and save to CSV."""
    metric_id = var["metric_id"]
    entity = var["entity"]
    desc = var["desc"]
    filepath = RAW_DIR / var["filename"]

    log.info(f"  Requesting: {metric_id} / {entity}")
    log.info(f"  Range: {START} -> {END}")

    try:
        df = api.request_data(metric_id, entity, START, END)
    except Exception as e:
        log.error(f"  Error: {e}")
        return None

    if df is None or df.empty:
        log.warning(f"  Empty result for {desc}")
        return None

    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info(f"  Saved {len(df)} rows ({size_mb:.1f} MB) -> {filepath.name}")
    return df


def download_all(api: ReadDB):
    """Download all configured variables."""
    log.info("=" * 60)
    log.info("XM Data Download")
    log.info(f"Range: {START} -> {END}")
    log.info(f"Output: {RAW_DIR}")
    log.info("=" * 60)

    results = {}
    for var in VARIABLES:
        log.info("-" * 40)
        log.info(f"{var['desc']}")
        df = download_one(api, var)
        results[var["filename"]] = df

    # Summary
    log.info("=" * 60)
    log.info("Summary")
    for fname, df in results.items():
        status = f"{len(df)} rows" if df is not None else "FAILED"
        log.info(f"  {fname}: {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download XM electricity market data"
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Fetch metrics catalog and validate MetricIds"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download all configured variables"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Discover then download"
    )
    args = parser.parse_args()

    if not (args.discover or args.download or args.all):
        parser.print_help()
        sys.exit(1)

    log.info("Initializing pydataxm API client...")
    api = ReadDB()
    log.info("Connected.")

    if args.discover or args.all:
        discover(api)

    if args.download or args.all:
        download_all(api)


if __name__ == "__main__":
    main()
