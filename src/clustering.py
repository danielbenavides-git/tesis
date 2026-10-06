import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


LEVEL_COLUMNS = {
    "real_price_log": ("log_price_mean", "log_price_std"),
    "real_price": ("price_mean", "price_std"),
}


def level_columns(metadata):
    """Daily level columns present in the scalogram metadata (log or linear real price)."""
    for columns in LEVEL_COLUMNS.values():
        if set(columns) <= set(metadata.columns):
            return columns
    raise ValueError(f"No daily level columns in metadata: {list(metadata.columns)}")


def with_level(latents, metadata, train_idx, columns=None):
    """Append the daily price level to the latents, z-scored with training-day statistics.
    The scalograms carry only the intraday shape, so the level enters the clustering here."""
    columns = columns or level_columns(metadata)
    level = metadata[list(columns)].to_numpy(dtype=np.float64)
    mean, std = level[train_idx].mean(axis=0), level[train_idx].std(axis=0)
    return np.hstack([latents.astype(np.float64), (level - mean) / std])


def _gmm(k, seed):
    return GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4, n_init=5, random_state=seed)


def _kmeans(k, seed):
    return KMeans(n_clusters=k, n_init=10, random_state=seed)


CLUSTERERS = {"gmm": _gmm, "kmeans": _kmeans}


def separation_metrics(x, labels, seed=42, silhouette_sample=5000):
    """Silhouette (higher is better), Davies-Bouldin (lower is better) and Calinski-Harabasz (higher is better)."""
    return {
        "silhouette": float(silhouette_score(x, labels, sample_size=min(silhouette_sample, len(x)), random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
    }


def select_k(latents, train_idx, method="gmm", k_range=range(2, 9), seed=42, silhouette_sample=5000):
    """One row per k on the training days: BIC (GMM) or inertia (k-means) plus the separation metrics."""
    x = latents[train_idx].astype(np.float64)
    rows = []
    for k in k_range:
        model = CLUSTERERS[method](k, seed).fit(x)
        labels = model.predict(x)
        criterion = {"bic": model.bic(x)} if method == "gmm" else {"inertia": model.inertia_}
        rows.append({"k": k, **criterion, **separation_metrics(x, labels, seed, silhouette_sample)})
    return pd.DataFrame(rows)


def select_gmm(latents, train_idx, k_range=range(2, 9), seed=42, silhouette_sample=5000):
    return select_k(latents, train_idx, "gmm", k_range, seed, silhouette_sample)


def fit_clusters(latents, train_idx, k, method="gmm", seed=42):
    """Fit on the training days, assign every day. Returns labels, probabilities (None for k-means) and the model."""
    latents = latents.astype(np.float64)
    model = CLUSTERERS[method](k, seed).fit(latents[train_idx])
    probs = model.predict_proba(latents) if method == "gmm" else None
    return model.predict(latents), probs, model


def fit_gmm(latents, train_idx, k, seed=42):
    return fit_clusters(latents, train_idx, k, "gmm", seed)


def fit_kmeans(latents, train_idx, k, seed=42):
    return fit_clusters(latents, train_idx, k, "kmeans", seed)


def order_regimes(labels, values, probs=None):
    """Relabel regimes 0..k-1 by ascending mean of values (e.g. real price) so labels are comparable across models."""
    means = pd.Series(values).groupby(labels).mean()
    order = means.sort_values().index.to_numpy()
    mapping = {old: new for new, old in enumerate(order)}
    new_labels = np.vectorize(mapping.get)(labels)
    if probs is None:
        return new_labels
    return new_labels, probs[:, order]


def run_lengths(labels):
    """Length in days of each uninterrupted run of the same regime."""
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], change, [len(labels)]])
    return np.diff(bounds)


def summary_metrics(latents, labels, seed=42, silhouette_sample=5000):
    """Separation metrics on the clustering features plus regime persistence, over all days."""
    runs = run_lengths(labels)
    n_years = len(labels) / 365.25
    return {
        "regimes": int(len(np.unique(labels))),
        **separation_metrics(latents, labels, seed, silhouette_sample),
        "mean_run_days": float(runs.mean()),
        "median_run_days": float(np.median(runs)),
        "switches_per_year": float((len(runs) - 1) / n_years),
    }


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


def assign_regimes(latents, train_idx, order_values, k=None, k_range=range(2, 9), seed=42, method="gmm"):
    """Selection table over k_range, then fit on training days with k and order regimes by order_values.
    For the GMM, k defaults to the lowest BIC; k-means has no automatic choice, so k is required."""
    table = select_k(latents, train_idx, method, k_range, seed)
    if not k:
        if method != "gmm":
            raise ValueError("k-means needs k. Read the inertia elbow in the selection table and set it")
        k = table.loc[table["bic"].idxmin(), "k"]
    k = int(k)
    labels, probs, _ = fit_clusters(latents, train_idx, k, method, seed)
    if probs is None:
        return table, k, order_regimes(labels, order_values), None
    labels, probs = order_regimes(labels, order_values, probs)
    return table, k, labels, probs


def save_regimes(run_dir, context, labels, probs, metrics):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = context[["date"]].assign(regime=labels)
    if probs is not None:
        for j in range(probs.shape[1]):
            out[f"prob_{j}"] = probs[:, j]
    out.to_csv(run_dir / "regimes.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
