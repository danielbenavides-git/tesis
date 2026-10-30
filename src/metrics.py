"""Size-aware metrics of the most expensive regime, compared with the real prices and the El Nino days.

A day is expensive when its real price is above a percentile of the whole series: 66.7 for k = 3, 75 for k = 4 and
80 for k = 5. The daily view uses each day's price; the 7d view uses the backward 7-day mean of the price (the day and
the 6 days before it), so a smoothed model is compared with smoothed prices. A day is El Nino when the ONI of its
month is >= 0.5, and the two episodes are the El Nino days in 2015-2016 and in 2023-2024 (as in runs.regime_table).
Every metric of a regime assignment is computed here, from regimes.csv of each run and from the regimes saved by
notebook 06; the runs only save their separation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz).

Ranking (agreed 2026-10-29): criterion A orders configurations by expensive_f1_pct, the F1 between
top_expensive_precision_pct and expensive_recall_pct. Criterion C first keeps the configurations whose switches per
year are at most persistence_factor times those of the 7-day baseline with the same k, then orders them by A.
The baselines are shown as reference and are not ranked.
"""
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from src.clustering import regime_profile, run_lengths
from src.runs import nino_episodes

thresholds = {3: 66.7, 4: 75.0, 5: 80.0}
price_views = {"daily": 1, "7d": 7}
persistence_factor = 2.0


def ep_key(name):
    """2015-2016 -> 2015_16"""
    return f"{name[:4]}_{name[7:9]}"


def days_from_context(context):
    """One row per day with date, real_price, nino (1 if ONI >= 0.5) and episode (2015-2016, 2023-2024 or empty)."""
    date = context["date"]
    nino = (context["enso_phase"] == "El Nino").astype(int)
    episode = pd.Series("", index=context.index)
    for name, years in nino_episodes.items():
        episode[(nino == 1) & date.dt.year.isin(years)] = name
    return pd.DataFrame({"date": date, "real_price": context["precio_real_cop_kwh"], "nino": nino,
                         "episode": episode}).reset_index(drop=True)


def expensive_days(days, k, view):
    """Boolean array of expensive days and the threshold in COP/kWh. The percentile is taken over all days of the
    daily price (daily view) or of its backward 7-day mean (7d view)."""
    price = days.sort_values("date")["real_price"].rolling(price_views[view], min_periods=1).mean()
    price = price.reindex(days.index).to_numpy()
    threshold = float(np.percentile(price, thresholds[k]))
    return price > threshold, threshold


def data_reference(days, k_values):
    """Level 1, no model. One row per (k, view, group); group is all days or one episode."""
    nino = days["nino"].to_numpy() == 1
    in_episode = days["episode"].to_numpy() != ""
    price = days["real_price"].to_numpy()
    rows = []
    for k in k_values:
        for view in price_views:
            expensive, threshold = expensive_days(days, k, view)
            groups = {"all": np.ones(len(days), bool),
                      **{name: days["episode"].to_numpy() == name for name in nino_episodes}}
            for group, mask in groups.items():
                rows.append({
                    "k": k, "threshold_pct": thresholds[k], "view": view, "threshold_cop_kwh": round(threshold, 1),
                    "group": group, "days": int(mask.sum()), "nino_days": int((mask & nino).sum()),
                    "expensive_pct": 100 * expensive[mask].mean(),
                    "median_price": float(np.median(price[mask])),
                    "median_price_rest": float(np.median(price[~in_episode])),
                    "base_expensive_pct": 100 * expensive.mean(), "base_nino_pct": 100 * nino.mean(),
                })
    return pd.DataFrame(rows).round(2)


def regime_metrics(days, k, view):
    """Level 2. days has date, real_price, nino, episode and regime (the regimes.csv columns); the most expensive
    regime is the highest number. Returns one dict with the metrics of that regime against expensive_days(k, view)."""
    expensive, threshold = expensive_days(days, k, view)
    top = days["regime"].to_numpy() == days["regime"].max()
    nino = days["nino"].to_numpy() == 1
    top_share = 100 * top.mean()
    out = {"view": view, "threshold_cop_kwh": round(threshold, 1), "top_share_pct": top_share}
    for name in nino_episodes:
        ep, key = days["episode"].to_numpy() == name, ep_key(name)
        recall = 100 * top[ep].mean()
        out[f"expensive_{key}_pct"] = 100 * expensive[ep].mean()
        out[f"nino_recall_{key}_pct"] = recall
        out[f"nino_lift_{key}"] = recall / top_share
        out[f"expensive_recall_{key}_pct"] = 100 * top[ep & expensive].mean()
        out[f"cheap_in_top_{key}_pct"] = 100 * top[ep & ~expensive].mean()
    precision = 100 * nino[top].mean()
    out["top_nino_precision_pct"] = precision
    out["top_nino_lift"] = precision / (100 * nino.mean())
    out["top_expensive_precision_pct"] = 100 * expensive[top].mean()
    out["expensive_recall_pct"] = 100 * top[expensive].mean()
    p, r = out["top_expensive_precision_pct"], out["expensive_recall_pct"]
    out["expensive_f1_pct"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    runs = run_lengths(days.sort_values("date")["regime"].to_numpy())
    out["switches_per_year"] = (len(runs) - 1) / (len(days) / 365.25)
    out["mean_run_days"] = float(runs.mean())
    return out


table_columns = {
    "switches_per_year": "Switches / yr", "mean_run_days": "Mean run (d)",
    "expensive_f1_pct": "Expensive F1 %",
    "top_share_pct": "Top %",
    "nino_recall_2015_16_pct": "Recall 15-16 %", "nino_lift_2015_16": "Lift 15-16",
    "nino_recall_2023_24_pct": "Recall 23-24 %", "nino_lift_2023_24": "Lift 23-24",
    "top_nino_lift": "Nino lift in top",
    "top_expensive_precision_pct": "Top expensive %",
    "expensive_recall_pct": "Expensive recall %",
    "cheap_in_top_2015_16_pct": "Cheap in top 15-16 %", "cheap_in_top_2023_24_pct": "Cheap in top 23-24 %",
}
config_columns = ["model", "study", "z_dim", "lam", "beta", "features", "latents"]


def reference_table(reference, k):
    t = PrettyTable(["View", "Threshold (COP/kWh)", "Group", "Days", "Expensive %", "Median price", "Base El Nino %"])
    for _, r in reference[reference["k"] == k].iterrows():
        t.add_row([r["view"], f"{r['threshold_cop_kwh']:.1f}", r["group"], r["days"], f"{r['expensive_pct']:.1f}",
                   f"{r['median_price']:.1f}", f"{r['base_nino_pct']:.1f}"])
    return t


def mean_cell(group, column):
    mean, std = group[column].mean(), group[column].std()
    digits = 2 if "lift" in column else 1
    return f"{mean:.{digits}f}" + (f" +- {std:.{digits}f}" if len(group) > 1 else "")


def metrics_table(rows, k):
    """One row per model configuration with the mean over seeds (+- std with more than one seed), only the view that
    matches the input (daily for unsmoothed inputs, 7d for smoothed ones)."""
    data = rows[(rows["k"] == k) & rows["matched"]].fillna({c: "-" for c in config_columns})
    t = PrettyTable(["Model", "Study", "z_dim", "lam", "beta", "Features", "Latents", "View", "Seeds",
                     *table_columns.values()])
    for keys, group in data.groupby(config_columns + ["view"], sort=False):
        t.add_row([*keys, len(group), *[mean_cell(group, c) for c in table_columns]])
    return t


def ranking(rows, k):
    """One row per model configuration (matched view, mean and std over seeds) with its rank by criterion A and, if it
    passes the persistence filter, by criterion C. Baselines get no rank."""
    data = rows[(rows["k"] == k) & rows["matched"]].fillna({c: "-" for c in config_columns})
    keys = config_columns + ["view"]
    columns = list(table_columns)
    stats = data.groupby(keys, sort=False)[columns].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    stats["seeds"] = data.groupby(keys, sort=False).size()
    stats = stats.reset_index()
    baseline = stats["model"] == "level_only"
    limit = persistence_factor * stats.loc[baseline & (stats["features"] == "level 7d"), "switches_per_year_mean"].iloc[0]
    stats["k"], stats["switches_limit"] = k, limit
    models = stats[~baseline].sort_values("expensive_f1_pct_mean", ascending=False)
    stats.loc[models.index, "rank_a"] = range(1, len(models) + 1)
    passed = models[models["switches_per_year_mean"] <= limit]
    stats.loc[passed.index, "rank_c"] = range(1, len(passed) + 1)
    return stats


ranking_columns = ["switches_per_year", "mean_run_days", "expensive_f1_pct", "top_expensive_precision_pct",
                   "expensive_recall_pct", "top_share_pct", "nino_lift_2015_16", "nino_lift_2023_24", "top_nino_lift"]


def ranking_table(stats, rank=None, top=5):
    """Baselines when rank is None, otherwise the first `top` configurations by rank_a or rank_c."""
    if rank is None:
        data = stats[stats["model"] == "level_only"]
    else:
        data = stats.dropna(subset=[rank]).sort_values(rank).head(top)
    t = PrettyTable(["Rank", "Model", "Study", "z_dim", "lam", "beta", "Features", "Latents", "View", "Seeds",
                     *[table_columns[c] for c in ranking_columns]])
    for _, r in data.iterrows():
        cells = []
        for c in ranking_columns:
            digits = 2 if "lift" in c else 1
            cells.append(f"{r[c + '_mean']:.{digits}f}" + (f" +- {r[c + '_std']:.{digits}f}" if r["seeds"] > 1 else ""))
        t.add_row(["ref" if rank is None else int(r[rank]), *[r[c] for c in config_columns + ["view"]],
                   int(r["seeds"]), *cells])
    return t


def profile_rows(context, regimes, identity):
    """Regime profile (days, share, real price, reservoir, ONI, El Nino share of each regime) with the identity columns."""
    return regime_profile(context.assign(regime=regimes)).assign(**identity)
