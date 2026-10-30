import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


LEVEL_COLUMNS = ("log_price_mean", "log_price_std")


def level_columns(metadata):
    """Daily mean and std of the log real price in the scalogram metadata."""
    if not set(LEVEL_COLUMNS) <= set(metadata.columns):
        raise ValueError(f"No {LEVEL_COLUMNS} columns in metadata: {list(metadata.columns)}")
    return LEVEL_COLUMNS


def with_level(latents, metadata, train_idx, columns=None):
    """Append the daily price mean and std to the latents, z-scored with training-day statistics.
    The scalograms carry only the intraday shape, so the price enters the clustering here."""
    columns = columns or level_columns(metadata)
    level = metadata[list(columns)].to_numpy(dtype=np.float64)
    mean, std = level[train_idx].mean(axis=0), level[train_idx].std(axis=0)
    return np.hstack([latents.astype(np.float64), (level - mean) / std])


def separation_metrics(x, labels, seed=42, silhouette_sample=5000):
    """Silhouette (higher is better), Davies-Bouldin (lower is better) and Calinski-Harabasz (higher is better).
    They need the clustering features, so they are computed when clustering; every other metric is in src/metrics.py."""
    return {
        "silhouette": float(silhouette_score(x, labels, sample_size=min(silhouette_sample, len(x)), random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
    }


def fit_clusters(features, train_idx, k, seed=42):
    """K-means fitted on the training days, assigns every day. Returns the labels and the model."""
    features = features.astype(np.float64)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(features[train_idx])
    return model.predict(features), model


def order_regimes(labels, values):
    """Relabel regimes 0..k-1 by ascending mean of values (the real price) so labels are comparable across models."""
    means = pd.Series(values).groupby(labels).mean()
    mapping = {old: new for new, old in enumerate(means.sort_values().index)}
    return np.vectorize(mapping.get)(labels)


def run_lengths(labels):
    """Length in days of each uninterrupted run of the same regime."""
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], change, [len(labels)]])
    return np.diff(bounds)


def regime_profile(context):
    """One row per regime: size, real price, reservoir level, ONI and share of El Nino days."""
    grouped = context.groupby("regime")
    profile = pd.DataFrame({
        "days": grouped.size(),
        "share_pct": 100 * grouped.size() / len(context),
        "mean_real_price": grouped["precio_real_cop_kwh"].mean(),
        "mean_reservoir": grouped["volumen_util_pct"].mean(),
        "mean_oni": grouped["oni"].mean(),
        "el_nino_pct": 100 * grouped["enso_phase"].apply(lambda s: (s == "El Nino").mean()),
    })
    return profile.reset_index()


def save_regimes(run_dir, table, metrics):
    """regimes.csv (the per-day table of runs.regime_table) and metrics.json (separation metrics)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_dir / "regimes.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
