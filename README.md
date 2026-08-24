# Detección de Regímenes de Mercado — Mercado Eléctrico Colombiano

Thesis project: unsupervised detection of market regimes in the Colombian electricity spot market using CWT scalograms and VAE-based neural networks.

**Author:** Daniel Benavides Santacruz (202220428)  
**Advisor:** Juan Fernando Pérez Bernal  
**Program:** Industrial Engineering, Universidad de los Andes  
**Semester:** 2026-10

## Structure

```
documents/          RPG proposal and thesis drafts
data/
  raw/              Raw CSVs from XM (not tracked in git)
  processed/        Cleaned and transformed data
scripts/            Data acquisition and processing
```

## Data Acquisition

```bash
pip install -r requirements.txt

# 1. Discover available MetricIds and validate config
python scripts/download_xm_data.py --discover

# 2. Download all variables
python scripts/download_xm_data.py --download
```

Source: XM S.A. E.S.P. (https://www.xm.com.co), via Sinergox API.
