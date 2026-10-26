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
  experiments.py    Run folders, saved config/metrics/tables, registry, loading a trained run and PrettyTable builders
  runs.py           Steps shared by notebooks 02 and 03: one run folder per k, clustering, figures, tables, k sweep
  comparison.py     Pairwise run comparison, winners by metric family, comparison CSVs
  plots.py          Loss curves, k selection, t-SNE and regime timeline charts
  models/
    conv_vae.py     Convolutional VAE (encoder, decoder, losses)
    temporal_vae.py ConvVAE with a causal LSTM over consecutive days
notebooks/
  01_eda.ipynb
  02_conv_vae.ipynb             ConvVAE + k-means or GMM
  03_temporal_vae.ipynb         Temporal VAE + k-means or GMM
  04_comparisons.ipynb          K-means vs GMM per model, then ConvVAE vs Temporal VAE
  05_level_baseline.ipynb       Baseline: clustering on the daily price level only (no CWT, no VAE)
experiments/
  registry.csv               One row per run: hyperparameters that change and main metrics
  comparisons.csv            One row per compared pair and metric (notebook 04)
  comparisons_summary.csv    One row per compared pair: winner per metric family and overall
  conv_vae/                  <clustering>/<study>/ with one folder per run, e.g. kmeans/k_sweep/001_real_price_log_kmeans_k2/
  temporal_vae/
  level_only/                Level-only baseline runs (notebook 05)
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

Notebooks 02 and 03 are set in their first configuration cell: `scalogram_set`, `clustering` (`"kmeans"` or `"gmm"`), `k_values`, `study`, `load_from` and the hyperparameters. The model is trained once and the clustering runs once per k in `k_values`. Notebook 03 also has a `grid` with lists of `lam`, `beta` and `seed`: it trains the model once for every combination, adds the values to the folder name (e.g. `lam100_beta0.5_seed42`) and saves `grid_results.csv` and `grid_summary.csv` (mean and standard deviation over seeds) in the study folder. Each k creates `experiments/<model>/<clustering>/<study>/NNN_<set>_<clustering>_k<k>/`, numbered consecutively inside each study folder. When `tag` is set (a short label such as `7d` or `z4`), it is added at the end of the folder name:

```
config.json, history.json, metrics.json, regimes.csv
tables/     hyperparameters, k_selection, k_sweep, regime_profile, clustering_metrics,
            price_spread, weekday_share, feature_checks
figures/    loss_curves, reconstructions, k_selection, tsne_regimes, regime_timeline
artifacts/  model.pt, latents.npy (not in git, only in runs with a model)
```

- `regimes.csv` has one row per day: `date`, `split` (train, val, test), `regime`, `real_price` (the price used to number the regimes), `oni`, `nino` (1 if the ONI of the month is >= 0.5), `episode` (2015-2016, 2023-2024 or empty), `top_regime` (1 if the regime is the most expensive one) and `captured` (1 if an El Nino day fell in the most expensive regime), plus `prob_0..` for the GMM. `python scripts/inspect_regimes.py <run folder>` prints the days, price and El Nino days of each regime, the share of each episode captured by the most expensive regime and the switches per year (`--missed` lists the episode days that fell elsewhere). Runs saved before these columns existed get them with `python scripts/inspect_regimes.py --update-all`.
- `study` names the question a run answers: `k_sweep`, `price_set`, `clustering`, `vae_model`, `hyperparameters` or `baseline`. Filter the registry with `registry[registry["study"] == "k_sweep"]`.
- `trained_in` in `config.json` names the run where the model was trained. Runs with the same `trained_in` share the same model and latents. It is empty in runs without a model (the baseline).
- `load_from` takes the `run_id` of an earlier run and skips training: the notebook loads its model, history and latents, and uses its set and hyperparameters. Use it to try other k or the other clustering algorithm on the same latents.

Each run adds a row to `experiments/registry.csv` with the study, scalogram set, clustering algorithm, k, filters, latent dimension, beta, lambda, KL warm-up, epochs, silhouette, Davies-Bouldin, Calinski-Harabasz, mean run length, switches per year and the git commit (`-dirty` if `src/` or `scripts/` had uncommitted changes; notebook settings are saved in each `config.json`). Commit code changes before a run so the commit identifies the code that produced it.

## Baseline

Notebook 05 clusters the days with only the daily price level (mean and standard deviation of the day's price, in logs for `real_price_log`), without scalograms or VAE. It is the reference every model has to beat: if the latents do not improve persistence or the capture of the El Nino episodes over this baseline, the intraday shape is not adding information about the regime. There are two formal baselines, set with `level_window`: 1 uses each day's own mean and standard deviation, and 7 replaces both by their backward 7-day mean (the day and the 6 days before it), which removes most of the day-to-day noise. Its runs go to `experiments/level_only/<clustering>/baseline/` (the 7-day runs with `tag = "7d"`) with `study = "baseline"` and are compared in notebook 04 like any other run.

## Comparisons

Notebook 04 reads finished runs and compares them in pairs: k-means vs GMM, log vs linear price, ConvVAE vs Temporal VAE and each model against the two baselines. The run ids go in its first cell. Metrics are grouped in three families: separation (silhouette, Davies-Bouldin, Calinski-Harabasz), persistence (mean run length, switches per year) and El Nino (share of the 2015-2016 and 2023-2024 El Nino days in the most expensive regime, and share of El Nino days among that regime's days). Separation is counted only when both runs share the model and the scalogram set; against a baseline or between different models only persistence and El Nino are counted. A run wins a family when it wins more of its metrics; the overall winner wins every counted family that is not tied, and a split is resolved by total wins over the counted metrics and marked as such. Each pair also reports what changed between the runs, the regime profiles side by side and the share of days assigned to the same regime. Results go to `experiments/comparisons.csv` and `experiments/comparisons_summary.csv`.

## Data Acquisition

```bash
pip install -r requirements.txt

# 1. Discover available MetricIds and validate config
python scripts/01_download_xm_data.py --discover

# 2. Download all variables
python scripts/01_download_xm_data.py --download
```

Source: XM S.A. E.S.P. (https://www.xm.com.co), via Sinergox API.
