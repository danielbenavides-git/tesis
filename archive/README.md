# Archive

Runs, notebooks and files that are no longer part of the thesis. Only this file is in git; the rest stays on the computer that archived it and in the git history (last commit before archiving, below).

## 2026-10-29, from commit 9d23a16
- `experiments/conv_vae`: ConvVAE runs (GMM, k sweep, log vs linear price): the ConvVAE is no longer used
- `experiments/temporal_vae/kmeans/vae_model`: Temporal VAE with 16 dims and blocks every 7 days, replaced by study z16
- `experiments/temporal_vae/kmeans/hyperparameters/005_real_price_log_kmeans_k6_z4_s14`: k = 6; experiments use k = 3, 4 and 5
- `experiments/level_only/kmeans/baseline/001_real_price_log_kmeans_k2`: k = 2; experiments use k = 3, 4 and 5
- `experiments/level_only/kmeans/baseline/005_real_price_log_kmeans_k6`: k = 6; experiments use k = 3, 4 and 5
- `experiments/level_only/kmeans/baseline/006_real_price_log_kmeans_k2_7d`: k = 2; experiments use k = 3, 4 and 5
- `experiments/level_only/kmeans/baseline/010_real_price_log_kmeans_k6_7d`: k = 6; experiments use k = 3, 4 and 5
- `experiments/comparisons.csv`: output of notebook 04, replaced by metrics/
- `experiments/comparisons_summary.csv`: output of notebook 04, replaced by metrics/
- `experiments/temporal_vae/kmeans/latent_smoothing/smoothing_results.csv`: old notebook 06 output; notebook 06 now saves regimes, metrics/ has the metrics
- `experiments/temporal_vae/kmeans/latent_smoothing/smoothing_summary.csv`: old notebook 06 output; notebook 06 now saves regimes, metrics/ has the metrics
- `experiments/temporal_vae/kmeans/latent_smoothing/regime_profiles.csv`: old notebook 06 output, replaced by metrics/regime_profiles.csv
- `metrics/smoothing_metrics.csv`: old notebook 06 output; scripts/06_regime_metrics.py now reads the regimes
- `notebooks/02_conv_vae.ipynb`: ConvVAE notebook: the ConvVAE is no longer used
- `notebooks/04_comparisons.ipynb`: pairwise comparisons, replaced by metrics/ and scripts/06_regime_metrics.py
- `check_extraction.py`: diagnostic of the continuous LSTM reading, closed (reverted)
- `data/processed/scalograms/real_price`: scalograms without log, discarded (log real price is used)
