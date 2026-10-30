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
    scalograms/real_price_log/   Log real price scalograms
scripts/            Data pipeline, run in order (01 download, 02 process, 03 external data and deflation, 04 scalograms,
                    05 El Nino price statistics), 06_regime_metrics.py (all metrics), inspect_regimes.py (one run),
                    archive_runs.py (one-off move to archive/)
src/                Code imported by the notebooks and scripts
  data.py           Scalogram loading, chronological split, sequence dataset, context variables
  training.py       Training loop with KL warm-up, active latent dims from the KL of each dim
  clustering.py     Price columns, k-means, regime ordering, run lengths, regime profile, separation metrics
  experiments.py    Run folders, saved config/history/tables, registry, loading a trained run, PrettyTable builders
  runs.py           Steps shared by notebooks 03 and 05: one run folder per k, clustering, regimes.csv, figures
  metrics.py        Every metric of a regime assignment (persistence, El Nino, expensive days) and the ranking
  plots.py          Loss curves, reconstructions, t-SNE and regime timeline charts
  models/
    conv_vae.py     Convolutional encoder, decoder and VAE losses
    temporal_vae.py Temporal VAE: CNN per day and a causal LSTM over consecutive days
notebooks/
  01_eda.ipynb
  03_temporal_vae.ipynb         Trains the Temporal VAE and saves its regimes (no analysis)
  05_level_baseline.ipynb       Baseline: clustering on the daily price only (no CWT, no VAE)
  06_latent_smoothing.ipynb     Smoothed (7-day) and scaled latents of the trained models, saves their regimes
experiments/
  registry.csv                   One row per run: what was run (study, k, z_dim, lam, beta, seed, epochs, trained_in, commit)
  level_only/kmeans/baseline/    Daily and 7-day price baselines (notebook 05)
  temporal_vae/kmeans/<study>/   Temporal VAE runs (notebook 03): hyperparameters, lam_beta_grid, z16
  temporal_vae/kmeans/latent_smoothing/   regimes.csv and inputs.csv of notebook 06
metrics/            Output of scripts/06_regime_metrics.py
archive/            Runs and files no longer used (not in git, except archive/README.md)
figures/            EDA figures (eda/) and sample scalograms (scalograms/<set>/)
```

## Modeling pipeline

```bash
pip install -r requirements.txt
python scripts/03_process_external_data.py          # ONI and real (deflated) prices
python scripts/04_generate_scalograms.py --price log  # -> scalograms/real_price_log/
```

The scalograms (`scalograms.npy`, about 114 MB) are not in git. Regenerate them with the commands above; the metadata and normalization parameters are tracked.

Each day's mean is removed from the hourly log real price before the CWT, so the image shows only the intraday shape. Magnitudes are scaled to [0, 1] with the global minimum and the 99.5th percentile, clipping values above it. The daily mean and standard deviation of the log real price are saved in `scalogram_metadata.csv` (`log_price_mean`, `log_price_std`) and appended to the latent vector before clustering, z-scored with training-day statistics.

Days are split chronologically (60% train, 20% validation, 20% test). K-means is fitted on training days only and then assigns every day. Regimes are numbered by ascending mean real price, so the highest number is the most expensive regime. Experiments use k = 3, 4 and 5.

## Experiments

Notebooks 03 and 05 only produce regimes; they do not analyze them. Notebook 03 is set in its configuration cell: `k_values`, `study`, `tag`, `load_from`, `active_threshold`, the `grid` of `lam`, `beta` and `seed` (one training per combination) and the fixed hyperparameters in `config`. Each trained model is clustered for each k, and each k creates `experiments/<model>/kmeans/<study>/NNN_<set>_kmeans_k<k>[_<tag>]/`:

```
config.json, history.json, regimes.csv, metrics.json
tables/     hyperparameters, regime_profile, clustering_metrics
figures/    loss_curves, reconstructions, tsne_regimes, regime_timeline
artifacts/  model.pt, latents.npy (not in git, only in runs with a model)
```

- `regimes.csv` has one row per day: `date`, `split` (train, val, test), `regime`, `real_price` (the price used to number the regimes), `oni`, `nino` (1 if the ONI of the month is >= 0.5), `episode` (2015-2016, 2023-2024 or empty), `top_regime` (1 if the regime is the most expensive one) and `captured` (1 if an El Nino day fell in the most expensive regime). `python scripts/inspect_regimes.py <run folder>` prints the days, price and El Nino days of each regime (`--missed` lists the episode days outside the most expensive regime).
- `metrics.json` has only the separation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz), which need the clustering features. Every other metric is computed by `scripts/06_regime_metrics.py`.
- `trained_in` in `config.json` names the run where the model was trained; runs with the same `trained_in` share the same model and latents. It is empty in the baseline.
- A latent dim is active when its KL on the training days at the last epoch is at least `active_threshold` (0.75 nats).
- `load_from` takes the `run_id` of an earlier run and skips training: the notebook clusters its saved latents again.
- Notebook 03 shows, for the models with seed 42, the t-SNE for k = 3, 4 and 5, the regime timeline of each k and the regime strips.

Each run writes its row in `experiments/registry.csv`, with the git commit (`-dirty` if `src/` or `scripts/` had uncommitted changes). Commit code changes before a run so the commit identifies the code that produced it.

## Baseline

Notebook 05 clusters the days with only the daily mean and standard deviation of the log real price. `level_window = 1` uses each day's own values (daily baseline) and `level_window = 7` replaces both by their backward 7-day mean, the day and the 6 days before it (7-day baseline, `tag = "7d"`). Runs go to `experiments/level_only/kmeans/baseline/`.

## Smoothed and scaled latents

Notebook 06 reads the latents of every trained model of the studies in its configuration and clusters three more inputs per model and k: latents scaled with training-day statistics, and latents and price both replaced by their backward 7-day mean, raw and scaled. It saves the regimes in `experiments/temporal_vae/kmeans/latent_smoothing/regimes.csv` (one column per input) and what each column is in `inputs.csv`.

## Metrics

`python scripts/06_regime_metrics.py` (after notebook 06) computes every metric from the saved regimes, with the definitions in `src/metrics.py`, and writes `metrics/`: `data_reference.csv`, `regime_metrics.csv` and `.txt`, `regime_profiles.csv`, `ranking.csv` and `.txt`. A day is expensive when its real price is above the 66.7, 75 or 80 percentile of the series for k = 3, 4 and 5; smoothed inputs are compared with the backward 7-day mean of the price. The ranking keeps the configurations with at most twice the switches per year of the 7-day baseline and orders them by the F1 between the share of the most expensive regime that is expensive and the share of expensive days in it. It stops if a trained model has not been read by notebook 06.

## Archive

`archive/` holds the runs, notebooks and files that are no longer used (ConvVAE, GMM, 16-dim Temporal VAE with blocks every 7 days, k = 2 and 6, notebook 04 and its comparisons, scalograms without log). They were moved with `python scripts/archive_runs.py`; `archive/README.md` lists what was moved, why, and the last commit before moving.

## Data Acquisition

```bash
pip install -r requirements.txt

# 1. Discover available MetricIds and validate config
python scripts/01_download_xm_data.py --discover

# 2. Download all variables
python scripts/01_download_xm_data.py --download
```

Source: XM S.A. E.S.P. (https://www.xm.com.co), via Sinergox API.
