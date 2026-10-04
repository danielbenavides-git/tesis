import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


LEVEL_COLUMNS = ("log_price_mean", "log_price_std")


def with_level(latents, metadata, train_idx, columns=LEVEL_COLUMNS):
    """Append the daily price level to the latents, z-scored with training-day statistics.
    The scalograms carry only the intraday shape, so the level enters the GMM here."""
    level = metadata[list(columns)].to_numpy(dtype=np.float64)
    mean, std = level[train_idx].mean(axis=0), level[train_idx].std(axis=0)
    return np.hstack([latents.astype(np.float64), (level - mean) / std])


def _gmm(k, seed):
    return GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4, n_init=5, random_state=seed)


def select_gmm(latents, train_idx, k_range=range(2, 9), seed=42, silhouette_sample=5000):
    """BIC and silhouette for each number of regimes, both computed on the training days."""
    x = latents[train_idx].astype(np.float64)
    rows = []
    for k in k_range:
        gmm = _gmm(k, seed).fit(x)
        labels = gmm.predict(x)
        sil = silhouette_score(x, labels, sample_size=min(silhouette_sample, len(x)), random_state=seed)
        rows.append({"k": k, "bic": gmm.bic(x), "silhouette": sil})
    return pd.DataFrame(rows)


def fit_gmm(latents, train_idx, k, seed=42):
    """Fit on the training days, assign every day. Returns labels, probabilities and the fitted model."""
    latents = latents.astype(np.float64)
    gmm = _gmm(k, seed).fit(latents[train_idx])
    return gmm.predict(latents), gmm.predict_proba(latents), gmm


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
    runs = run_lengths(labels)
    n_years = len(labels) / 365.25
    return {
        "regimes": int(len(np.unique(labels))),
        "silhouette": float(silhouette_score(latents, labels, sample_size=min(silhouette_sample, len(labels)),
                                             random_state=seed)),
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


def assign_regimes(latents, train_idx, order_values, k=None, k_range=range(2, 9), seed=42):
    """Select k by lowest BIC unless k is given, fit the GMM on training days and order regimes by order_values."""
    table = select_gmm(latents, train_idx, k_range, seed)
    k = int(k) if k else int(table.loc[table["bic"].idxmin(), "k"])
    labels, probs, _ = fit_gmm(latents, train_idx, k, seed)
    labels, probs = order_regimes(labels, order_values, probs)
    return table, k, labels, probs


def save_regimes(run_dir, context, labels, probs, metrics):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = context[["date"]].assign(regime=labels)
    for j in range(probs.shape[1]):
        out[f"prob_{j}"] = probs[:, j]
    out.to_csv(run_dir / "regimes.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
