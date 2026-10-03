# Detección de Regímenes de Mercado - Mercado Eléctrico Colombiano

Thesis project: unsupervised detection of market regimes in the Colombian electricity spot market using CWT scalograms and VAE-based neural networks.

**Author:** Daniel Benavides Santacruz (202220428)  
**Advisor:** Juan Fernando Pérez Bernal  
**Program:** Industrial Engineering, Universidad de los Andes  
**Semester:** 2026-10

## Structure

```
documents/          Reference papers
data/
  raw/              Raw data from XM, DANE and NOAA
  processed/        Cleaned series and scalograms (scalograms/ holds scalograms.npy and metadata)
scripts/            Data pipeline, run in order (01 download, 02 process, 03 scalograms, 04 external data)
src/                Model code imported by the notebooks
  data.py           Scalogram loading, chronological split, daily and sequence datasets, context variables
  training.py       Training loop with KL warm-up, run saving
  clustering.py     GMM regime assignment, regime ordering, metrics and profiles
  plots.py          Loss curves, model selection, t-SNE and regime timeline charts
  models/
    conv_vae.py     Convolutional VAE (encoder, decoder, losses)
    temporal_vae.py ConvVAE with a causal LSTM over consecutive days
    baselines.py    Raw PCA and frozen MobileNetV2 + PCA
notebooks/
  01_eda.ipynb
  02_vae.ipynb           ConvVAE + GMM
  03_temporal_vae.ipynb  Temporal VAE + GMM
  04_baselines.ipynb     Raw PCA and MobileNetV2 baselines, comparison of all models
  archive/               Earlier test notebooks
models/             Checkpoints, latent vectors and regime labels per model (not tracked in git)
figures/            Figures per notebook (eda/, vae/, temporal_vae/, baselines/)
```

## Modeling pipeline

```bash
pip install -r requirements.txt
python scripts/03_generate_scalograms.py   # if scalograms.npy is missing
```

Then run notebooks 02, 03 and 04 in order. Days are split chronologically (60% train, 20% validation, 20% test) and the GMM is fitted on training days only.

## Data Acquisition

```bash
pip install -r requirements.txt

# 1. Discover available MetricIds and validate config
python scripts/01_download_xm_data.py --discover

# 2. Download all variables
python scripts/01_download_xm_data.py --download
```

Source: XM S.A. E.S.P. (https://www.xm.com.co), via Sinergox API.
