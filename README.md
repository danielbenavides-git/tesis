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
  processed/        Cleaned series (XM/, NOAA/) and scalograms
    scalograms/
      real_price_log/   Log real price scalograms
      real_price/       Real price scalograms (COP/kWh, no log)
scripts/            Data pipeline, run in order (01 download, 02 process, 03 external data and deflation, 04 scalograms)
src/                Model code imported by the notebooks
  data.py           Scalogram loading, chronological split, daily and sequence datasets, context variables
  training.py       Training loop with KL warm-up
  clustering.py     K-means and GMM regime assignment, k selection, regime ordering, metrics and profiles
  experiments.py    Run folders, saved config/metrics/tables, registry and PrettyTable builders
  comparison.py     Pairwise run comparison, winners by metric family, comparison CSVs
  plots.py          Loss curves, k selection, t-SNE and regime timeline charts
  models/
    conv_vae.py     Convolutional VAE (encoder, decoder, losses)
    temporal_vae.py ConvVAE with a causal LSTM over consecutive days
notebooks/
  01_eda.ipynb
  02_vae_gmm.ipynb              ConvVAE + GMM
  02_vae_kmeans.ipynb           ConvVAE + k-means
  03_temporal_vae_gmm.ipynb     Temporal VAE + GMM
  03_temporal_vae_kmeans.ipynb  Temporal VAE + k-means
  04_comparisons.ipynb          K-means vs GMM per model, then ConvVAE vs Temporal VAE
experiments/
  registry.csv               One row per run: hyperparameters that change and main metrics
  comparisons.csv            One row per compared pair and metric (notebook 04)
  comparisons_summary.csv    One row per compared pair: winner per metric family and overall
  conv_vae/                  One folder per run, e.g. 001_real_price_kmeans_k2/
  temporal_vae/
figures/            EDA figures (eda/) and sample scalograms (scalograms/<set>/)
```

## Modeling pipeline

```bash
pip install -r requirements.txt
python scripts/03_process_external_data.py            # ONI and real (deflated) prices
python scripts/04_generate_scalograms.py --price linear   # -> scalograms/real_price/
python scripts/04_generate_scalograms.py --price log      # -> scalograms/real_price_log/
```

The scalograms (`scalograms.npy`, about 114 MB per set) are not in git. Regenerate them with the commands above; the metadata and normalization parameters of each set are tracked.

Both sets follow the same steps on the hourly real price, and the log is the only difference. Each day's mean is removed before the CWT, so the image shows only the intraday shape. Magnitudes are scaled to [0, 1] with the global minimum and the set's own 99.5th percentile, clipping values above it. The daily level (mean and standard deviation of the same series) is saved in `scalogram_metadata.csv` and appended to the latent vector before clustering: `log_price_mean` and `log_price_std` for `real_price_log`, `price_mean` and `price_std` for `real_price`.

Days are split chronologically (60% train, 20% validation, 20% test). K-means and the GMM are fitted on training days only and then assign every day.

## Experiments

Each run of notebooks 02 and 03 is set in the first cell (`SCALOGRAM_SET`, `CLUSTERING`, `N_REGIMES` and the hyperparameters) and creates `experiments/<model>/NNN_<set>_<clustering>_k<k>/` with a consecutive number:

```
config.json, history.json, metrics.json, regimes.csv
tables/     splits, hyperparameters, k_selection, regime_profile, clustering_metrics (PrettyTable .txt)
figures/    loss_curves, reconstructions, k_selection, tsne_regimes, regime_timeline
artifacts/  model.pt, latents.npy (not in git)
```

The last cell adds the run to `experiments/registry.csv` with the scalogram set, clustering algorithm, k, filters, latent dimension, beta, lambda, KL warm-up, epochs, silhouette, Davies-Bouldin, Calinski-Harabasz, mean run length, switches per year and the git commit (`-dirty` if `src/` or `scripts/` had uncommitted changes). Commit code changes before a run so the commit identifies the code that produced it.

## Comparisons

Notebook 04 reads finished runs and compares them in pairs: k-means vs GMM for each model, then the two winners (ConvVAE vs Temporal VAE). The run ids go in its first cell. Metrics are grouped in two families, separation (silhouette, Davies-Bouldin, Calinski-Harabasz) and persistence (mean run length, switches per year). A run wins a family when it wins more of its metrics; the overall winner wins both families, or one with the other tied, and a split is resolved by total metric wins and marked as such. Each pair also reports what changed between the runs, the regime profiles side by side and the share of days assigned to the same regime. Results go to `experiments/comparisons.csv` and `experiments/comparisons_summary.csv`.

## Data Acquisition

```bash
pip install -r requirements.txt

# 1. Discover available MetricIds and validate config
python scripts/01_download_xm_data.py --discover

# 2. Download all variables
python scripts/01_download_xm_data.py --download
```

Source: XM S.A. E.S.P. (https://www.xm.com.co), via Sinergox API.
