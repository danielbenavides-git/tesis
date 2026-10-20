"""Steps shared by notebooks 02 and 03 after the model is trained: one run folder per k, clustering, figures,
tables and the registry row. The model is trained (or loaded) once and every k reuses its latents."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from src import experiments as exp
from src.clustering import fit_clusters, order_regimes, regime_profile, save_regimes, summary_metrics
from src.data import chronological_split, load_context
from src.plots import regime_timeline, selection_chart, tsne_chart

# Keys taken from the current notebook when load_from reuses an earlier run. Everything else (set, filters,
# epochs...) comes from the loaded run, so the config always describes the model that produced the latents.
RUN_KEYS = ("study", "clustering", "k_values", "k_range", "tag")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# El Nino episodes used to judge the most expensive regime: El Nino days (ONI >= 0.5) in these years.
nino_episodes = {"2015-2016": [2015, 2016], "2023-2024": [2023, 2024]}
regime_columns = ["date", "split", "regime", "real_price", "oni", "nino", "episode", "top_regime", "captured"]


def existing_runs(model, clustering, trained_in, scalogram_set):
    """{k: run folder} of earlier runs that clustered the latents of trained_in with this algorithm and set."""
    found = {}
    for path in sorted((exp.EXPERIMENTS_DIR / model / clustering).glob("*/*/config.json")):
        c = json.loads(path.read_text())
        if c.get("trained_in") == trained_in and c.get("scalogram_set") == scalogram_set:
            found.setdefault(c["n_regimes"], path.parent)
    return found


def prepare_runs(config, load_from=None, device="cpu"):
    """One run folder per k in config["k_values"], with its config saved. Runs go to
    experiments/<model>/<clustering>/<study>/, so each study keeps its own folder.
    With load_from (a run_id), the model, history and latents of that run are reused and training is skipped.
    A k that was already clustered with the same model, algorithm and set reuses its folder instead of creating
    a new one (its files are rewritten with the same results); only new k or a new algorithm get new folders.
    Returns (config, run_dirs, run_configs, trained); trained is None when the model has to be trained.
    trained_in is empty for runs without a model (no z_dim in the config, e.g. the level-only baseline)."""
    trained = None
    reused = {}
    if load_from:
        trained = exp.load_trained(load_from, device)
        config = {**trained["config"], **{key: config[key] for key in RUN_KEYS if key in config}}
        trained_in = trained["config"].get("trained_in", load_from)
        reused = existing_runs(config["model"], config["clustering"], trained_in, config["scalogram_set"])
    run_dirs = {
        k: reused.get(k) or exp.create_run(config["model"], config["clustering"], config["study"],
                                           config["scalogram_set"], k, config.get("tag", ""))
        for k in config["k_values"]
    }
    if not trained:
        trained_in = exp.run_id(run_dirs[config["k_values"][0]]) if "z_dim" in config else ""
    run_configs = {}
    for k, run_dir in run_dirs.items():
        if k in reused:
            run_configs[k] = json.loads((run_dir / "config.json").read_text())
        else:
            run_configs[k] = {**config, "n_regimes": k, "trained_in": trained_in}
            exp.save_config(run_dir, run_configs[k])
    if reused:
        print(f"Reused folders for k = {sorted(k for k in run_dirs if k in reused)}; "
              f"new folders for k = {sorted(k for k in run_dirs if k not in reused)}")
    return config, run_dirs, run_configs, trained


def save_all(run_dirs, save, *args, **kwargs):
    """Call save(run_dir, *args, **kwargs) for every run folder."""
    for run_dir in run_dirs.values():
        save(run_dir, *args, **kwargs)


def feature_checks(feature_sets, train_idx, k, method, context, seed=42):
    """Same algorithm and k on other clustering inputs, e.g. level only or latents only."""
    price = context["precio_real_cop_kwh"].to_numpy()
    t = PrettyTable(["Features", "Silhouette", "Davies-Bouldin", "Mean run", "Switches / yr", "Median price by regime"])
    for name, f in feature_sets.items():
        lab = order_regimes(fit_clusters(f, train_idx, k, method, seed)[0], price)
        m = summary_metrics(f, lab, seed=seed)
        med = context.assign(r=lab).groupby("r")["precio_real_cop_kwh"].median().round(0).tolist()
        t.add_row([name, f"{m['silhouette']:.3f}", f"{m['davies_bouldin']:.3f}", f"{m['mean_run_days']:.1f}",
                   f"{m['switches_per_year']:.0f}", med])
    return t


def cluster_runs(run_dirs, run_configs, features, context, train_idx, split_dates, selection,
                 coords, plot_idx, feature_sets):
    """For each k: fit on training days, order regimes by mean real price, save figures, tables,
    regimes.csv, metrics.json and the registry row. Returns {k: {metrics, profile, weekday, context}}."""
    price = context["precio_real_cop_kwh"].to_numpy()
    results = {}
    for k, run_dir in run_dirs.items():
        config = run_configs[k]
        method, seed = config["clustering"], config["seed"]
        labels, probs, _ = fit_clusters(features, train_idx, k, method, seed)
        if probs is None:
            labels = order_regimes(labels, price)
        else:
            labels, probs = order_regimes(labels, price, probs)
        ctx = context.assign(regime=labels)

        selection_chart(selection, k).save(exp.figure_path(run_dir, "k_selection"), scale_factor=2.0)
        tsne_chart(coords, labels[plot_idx], f"t-SNE of the latent space (k={k})").save(
            exp.figure_path(run_dir, "tsne_regimes"), scale_factor=2.0)
        regime_timeline(ctx, split_dates, title=f"Detected regimes over time (k={k})").save(
            exp.figure_path(run_dir, "regime_timeline"), scale_factor=2.0)

        profile = regime_profile(ctx)
        metrics = {"run_id": exp.run_id(run_dir), "model": config["model"], "scalogram_set": config["scalogram_set"],
                   "clustering": method, "k": k, **summary_metrics(features, labels, seed=seed)}
        save_regimes(run_dir, regime_table(ctx, split_dates), probs, metrics)
        exp.log_run(run_dir, config, metrics, registry=exp.EXPERIMENTS_DIR / "registry.csv")

        spread = ctx.groupby("regime")["precio_real_cop_kwh"].describe(percentiles=[0.1, 0.5, 0.9])
        weekday = pd.crosstab(labels, ctx["date"].dt.dayofweek, normalize="columns").mul(100).round(0)
        weekday.index.name, weekday.columns = "regime", WEEKDAYS
        exp.save_table(run_dir, "regime_profile", exp.regime_profile_table(profile), show=False)
        exp.save_table(run_dir, "clustering_metrics", exp.metrics_table(metrics), show=False)
        exp.save_table(run_dir, "price_spread", spread[["count", "10%", "50%", "90%", "min", "max"]].round(1), show=False)
        exp.save_table(run_dir, "weekday_share", weekday, show=False)
        exp.save_table(run_dir, "feature_checks", feature_checks(feature_sets, train_idx, k, method, context, seed),
                       show=False)

        results[k] = {"metrics": metrics, "profile": profile, "weekday": weekday, "context": ctx}
        print(f"k={k}: {exp.run_id(run_dir)}  silhouette={metrics['silhouette']:.3f}  "
              f"mean run={metrics['mean_run_days']:.1f} d  switches/yr={metrics['switches_per_year']:.1f}")
    return results


def nino_share_in_top(ctx, years):
    """Share (%) of El Nino days in the given years that fall in the most expensive regime."""
    days = ctx[(ctx["enso_phase"] == "El Nino") & ctx["date"].dt.year.isin(years)]
    return 100 * (days["regime"] == ctx["regime"].max()).mean()


def regime_table(ctx, split_dates):
    """One row per day, saved as regimes.csv so every El Nino number can be checked by hand:
    split (train, val, test), regime, real price (the one used to number the regimes), ONI of the month,
    nino (1 if ONI >= 0.5), episode (2015-2016, 2023-2024 or empty), top_regime (1 if the regime is the most
    expensive one) and captured (1 if an El Nino day fell in the most expensive regime)."""
    date = ctx["date"]
    nino = (ctx["enso_phase"] == "El Nino").astype(int)
    episode = pd.Series("", index=ctx.index)
    for name, years in nino_episodes.items():
        episode[(nino == 1) & date.dt.year.isin(years)] = name
    top = (ctx["regime"] == ctx["regime"].max()).astype(int)
    split = np.where(date < split_dates[0], "train", np.where(date < split_dates[1], "val", "test"))
    return pd.DataFrame({"date": date, "split": split, "regime": ctx["regime"], "real_price": ctx["precio_real_cop_kwh"],
                         "oni": ctx["oni"], "nino": nino, "episode": episode, "top_regime": top,
                         "captured": nino * top})[regime_columns]


def complete_regimes(run_dir, data_dir):
    """regimes.csv of a run with all the regime_table columns. Older runs only saved date and regime (and GMM
    probabilities): the other columns are rebuilt from the data and the run's own split ratios."""
    run_dir = Path(run_dir)
    saved = pd.read_csv(run_dir / "regimes.csv", parse_dates=["date"])
    if set(regime_columns) <= set(saved.columns):
        return saved
    config = json.loads((run_dir / "config.json").read_text())
    ctx = load_context(data_dir, pd.DataFrame({"start_datetime": saved["date"]}))
    ctx = ctx.assign(regime=saved["regime"].to_numpy())
    _, val_idx, test_idx = chronological_split(len(saved), config["train_ratio"], config["val_ratio"])
    table = regime_table(ctx, [saved["date"].iloc[val_idx[0]], saved["date"].iloc[test_idx[0]]])
    return pd.concat([table, saved.filter(regex=r"^prob_")], axis=1)


def k_sweep_table(results):
    """One row per k: separation, persistence and the most expensive regime (size, price, reservoir, El Nino)."""
    rows = []
    for k, r in results.items():
        m, top = r["metrics"], r["profile"].iloc[-1]
        rows.append({
            "k": k, "silhouette": m["silhouette"], "davies_bouldin": m["davies_bouldin"],
            "calinski_harabasz": m["calinski_harabasz"], "mean_run_days": m["mean_run_days"],
            "switches_per_year": m["switches_per_year"], "top_share_pct": top["share_pct"],
            "top_price": top["mean_real_price"], "top_reservoir": top["mean_reservoir"],
            "nino_2015_16_in_top_pct": nino_share_in_top(r["context"], nino_episodes["2015-2016"]),
            "nino_2023_24_in_top_pct": nino_share_in_top(r["context"], nino_episodes["2023-2024"]),
        })
    return pd.DataFrame(rows).set_index("k").round(3)


def tsne_grid(coords, plot_idx, results, per_row=3):
    """t-SNE colored by each k's regimes, per_row charts per row."""
    plots = [tsne_chart(coords, r["context"]["regime"].to_numpy()[plot_idx], f"k={k}").properties(width=260, height=240)
             for k, r in results.items()]
    rows = [alt.hconcat(*plots[i:i + per_row]) for i in range(0, len(plots), per_row)]
    return alt.vconcat(*rows).resolve_scale(color="independent")


def regime_strips(results):
    """Regime strip over time for every k, one row per k."""
    return alt.vconcat(*[
        alt.Chart(r["context"].assign(regime=r["context"]["regime"].astype(str))).mark_tick(thickness=1, size=12).encode(
            x=alt.X("date:T", title=""), y=alt.Y("regime:O", title=f"k={k}"), color=alt.Color("regime:N", legend=None),
        ).properties(width=800, height=18 * k)
        for k, r in results.items()
    ]).resolve_scale(color="independent")
