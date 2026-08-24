"""
Download historical data from XM (Colombian electricity system operator).

Uses pydataxm (Sinergox API). The library handles monthly chunking
and async requests internally. No API key required.

Requirements:
    pip install pydataxm pandas

Usage:
    python scripts/download_xm_data.py --discover     # validate MetricIds
    python scripts/download_xm_data.py --download      # download all
    python scripts/download_xm_data.py --all            # both
"""

import argparse
import logging
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
from pydataxm.pydataxm import ReadDB # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger(__name__)

START = dt.date(2000, 1, 1)
END = dt.date(2026, 7, 31)

# ---------------------------------------------------------------------------
# Variables — MetricIds validated against catalogo_metricas_xm.csv
# ---------------------------------------------------------------------------

VARIABLES = [
    # === CORE ===
    {
        "metric_id": "PrecBolsNaci",
        "entity": "Sistema",
        "filename": "precio_bolsa_horario.csv",
        "desc": "Precio Bolsa Nacional (hourly, $/kWh)",
    },
    {
        "metric_id": "PPPrecBolsNaci",
        "entity": "Sistema",
        "filename": "precio_bolsa_diario_ponderado.csv",
        "desc": "Precio Bolsa Nacional Ponderado (daily, $/kWh)",
    },
    # === HYDROLOGY ===
    {
        "metric_id": "VoluUtilDiarEner",
        "entity": "Sistema",
        "filename": "volumen_util_diario.csv",
        "desc": "Volumen Util Diario Energia (daily, GWh)",
    },
    {
        "metric_id": "AporEner",
        "entity": "Sistema",
        "filename": "aportes_energia_diario.csv",
        "desc": "Aportes Energia del SIN (daily, GWh)",
    },
    {
        "metric_id": "PorcVoluUtilDiar",
        "entity": "Sistema",
        "filename": "porcentaje_volumen_util_diario.csv",
        "desc": "Volumen Util Diario % (daily)",
    },
    # === GENERATION ===
    {
        "metric_id": "Gene",
        "entity": "Sistema",
        "filename": "generacion_real_horaria.csv",
        "desc": "Generacion Real Total (hourly, kWh)",
    },
    # === DEMAND ===
    {
        "metric_id": "DemaReal",
        "entity": "Sistema",
        "filename": "demanda_real_horaria.csv",
        "desc": "Demanda Real del SIN (hourly, kWh)",
    },
    {
        "metric_id": "DemaSIN",
        "entity": "Sistema",
        "filename": "demanda_sin_diaria.csv",
        "desc": "Demanda Energia SIN (daily, kWh)",
    },
    # === PRICES ===
    {
        "metric_id": "PrecEscaSup",
        "entity": "Sistema",
        "filename": "precio_escasez_superior.csv",
        "desc": "Precio Escasez Superior (daily, $/kWh)",
    },
    {
        "metric_id": "MaxPrecOferNal",
        "entity": "Sistema",
        "filename": "max_precio_oferta_horario.csv",
        "desc": "Maximo Precio Oferta Nacional (hourly, $/kWh)",
    },
]


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------

def discover(api: ReadDB):
    """Fetch metrics catalog, validate configured MetricIds."""
    log.info("Fetching metrics catalog...")
    catalog = api.get_collections()
    out = RAW_DIR / "catalogo_metricas_xm.csv"
    catalog.to_csv(out, index=False)
    log.info(f"Saved {len(catalog)} metrics to {out}")

    all_ids = set(catalog["MetricId"].values)
    log.info("\nValidating configured MetricIds:")
    for var in VARIABLES:
        mid = var["metric_id"]
        ent = var["entity"]
        match = catalog[
            (catalog["MetricId"] == mid) & (catalog["Entity"] == ent)
        ]
        if not match.empty:
            log.info(f"  OK   {mid:25s} / {ent}")
        else:
            log.error(f"  FAIL {mid:25s} / {ent}")
            similar = [x for x in all_ids if mid[:4].lower() in x.lower()]
            if similar:
                log.info(f"       Similar: {similar[:5]}")

    return catalog


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_one(api: ReadDB, var: dict):
    """Download one variable across full date range."""
    metric_id = var["metric_id"]
    entity = var["entity"]
    filepath = RAW_DIR / var["filename"]

    log.info(f"  {metric_id} / {entity}  ({START} -> {END})")

    try:
        df = api.request_data(metric_id, entity, START, END)
    except Exception as e:
        log.error(f"  Error: {e}")
        return None

    if df is None or df.empty:
        log.warning(f"  Empty result")
        return None

    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info(f"  -> {len(df)} rows ({size_mb:.1f} MB) saved to {filepath.name}")
    return df


def download_all(api: ReadDB):
    """Download all configured variables."""
    log.info("=" * 60)
    log.info(f"XM Data Download  |  {START} -> {END}")
    log.info(f"Output: {RAW_DIR}")
    log.info("=" * 60)

    results = {}
    for var in VARIABLES:
        log.info(f"\n{var['desc']}")
        df = download_one(api, var)
        results[var["filename"]] = df

    log.info("\n" + "=" * 60)
    log.info("Summary")
    for fname, df in results.items():
        status = f"{len(df):>10,} rows" if df is not None else "     FAILED"
        log.info(f"  {fname:45s} {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Download XM electricity market data")
    p.add_argument("--discover", action="store_true", help="Validate MetricIds")
    p.add_argument("--download", action="store_true", help="Download all variables")
    p.add_argument("--all", action="store_true", help="Discover then download")
    args = p.parse_args()

    if not (args.discover or args.download or args.all):
        p.print_help()
        sys.exit(1)

    log.info("Connecting to XM API...")
    api = ReadDB()
    log.info("Connected.\n")

    if args.discover or args.all:
        discover(api)

    if args.download or args.all:
        download_all(api)


if __name__ == "__main__":
    main()