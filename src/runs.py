"""Steps shared by notebooks 03 and 05 after the features are ready: one run folder per k, clustering, figures,
the two tables, regimes.csv, metrics.json and the registry row. Every other metric is computed later by
scripts/06_regime_metrics.py from regimes.csv."""
import json

import altair as alt
import numpy as np
import pandas as pd

from src import experiments as exp
from src.clustering import fit_clusters, order_regimes, regime_profile, save_regimes, separation_metrics
from src.plots import regime_timeline, tsne_chart

# Keys taken from the current notebook when load_from reuses an earlier run. Everything else comes from the loaded run,
# so the config always describes the model that produced the latents.
RUN_KEYS = ("study", "k_values", "tag")
# El Nino episodes used to judge the most expensive regime: El Nino days (ONI >= 0.5) in these years.
nino_episodes = {"2015-2016": [2015, 2016], "2023-2024": [2023, 2024]}
regime_columns = ["date", "split", "regime", "real_price", "oni", "nino", "episode", "top_regime", "captured"]


def existing_runs(model, trained_in):
    """{k: run folder} of earlier runs that clustered the latents of trained_in."""
    found = {}
    for path in sorted((exp.EXPERIMENTS_DIR / model / "kmeans").glob("*/*/config.json")):
        c = json.loads(path.read_text())
        if c.get("trained_in") == trained_in:
            found.setdefault(c["n_regimes"], path.parent)
    return found


def prepare_runs(config, load_from=None, device="cpu"):
    """One run folder per k in config["k_values"], with its config saved, in experiments/<model>/kmeans/<study>/.
    With load_from (a run_id), the model, history and latents of that run are reused and training is skipped, and a k
    already clustered with that model reuses its folder. Returns (config, run_dirs, run_configs, trained); trained is
    None when the model has to be trained. trained_in is empty for runs without a model (the price baseline)."""
    trained, reused = None, {}
    if load_from:
        trained = exp.load_trained(load_from, device)
        config = {**trained["config"], **{key: config[key] for key in RUN_KEYS if key in config}}
        trained_in = trained["config"].get("trained_in", load_from)
        reused = existing_runs(config["model"], trained_in)
    run_dirs = {k: reused.get(k) or exp.create_run(config["model"], config["study"], config["scalogram_set"], k,
                                                   config.get("tag", ""))
                for k in config["k_values"]}
    if not trained:
        trained_in = exp.run_id(run_dirs[config["k_values"][0]]) if "z_dim" in config else ""
    run_configs = {}
    for k, run_dir in run_dirs.items():
        if k in reused:
            run_configs[k] = json.loads((run_dir / "config.json").read_text())
        else:
            run_configs[k] = {**config, "n_regimes": k, "trained_in": trained_in}
            exp.save_config(run_dir, run_configs[k])
    return config, run_dirs, run_configs, trained


def save_all(run_dirs, save, *args, **kwargs):
    """Call save(run_dir, *args, **kwargs) for every run folder."""
    for run_dir in run_dirs.values():
        save(run_dir, *args, **kwargs)


def cluster_runs(run_dirs, run_configs, features, context, train_idx, split_dates, coords, plot_idx):
    """For each k: k-means fitted on training days, regimes numbered by mean real price, then saved: regimes.csv,
    metrics.json (separation metrics), tables/regime_profile.txt, tables/clustering_metrics.txt, figures/tsne_regimes.png,
    figures/regime_timeline.png and the registry row. Returns {k: context with the regime column}."""
    price = context["precio_real_cop_kwh"].to_numpy()
    results = {}
    for k, run_dir in run_dirs.items():
        config, seed = run_configs[k], run_configs[k]["seed"]
        labels = order_regimes(fit_clusters(features, train_idx, k, seed)[0], price)
        ctx = context.assign(regime=labels)
        metrics = {"run_id": exp.run_id(run_dir), "k": k, **separation_metrics(features, labels, seed=seed)}
        save_regimes(run_dir, regime_table(ctx, split_dates), metrics)
        exp.save_table(run_dir, "regime_profile", exp.regime_profile_table(regime_profile(ctx)), show=False)
        exp.save_table(run_dir, "clustering_metrics", exp.metrics_table(metrics), show=False)
        tsne_chart(coords, labels[plot_idx], f"t-SNE of the clustering features (k={k})").save(
            exp.figure_path(run_dir, "tsne_regimes"), scale_factor=2.0)
        regime_timeline(ctx, split_dates, title=f"Detected regimes over time (k={k})").save(
            exp.figure_path(run_dir, "regime_timeline"), scale_factor=2.0)
        exp.log_run(run_dir, config)
        results[k] = ctx
    return results


def regime_table(ctx, split_dates):
    """One row per day, saved as regimes.csv: split (train, val, test), regime, real price (the one used to number the
    regimes), ONI of the month, nino (1 if ONI >= 0.5), episode (2015-2016, 2023-2024 or empty), top_regime (1 if the
    regime is the most expensive one) and captured (1 if an El Nino day fell in the most expensive regime)."""
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


def tsne_grid(coords, plot_idx, results, per_row=3):
    """t-SNE colored by each k's regimes, per_row charts per row."""
    plots = [tsne_chart(coords, ctx["regime"].to_numpy()[plot_idx], f"k={k}").properties(width=260, height=240)
             for k, ctx in results.items()]
    rows = [alt.hconcat(*plots[i:i + per_row]) for i in range(0, len(plots), per_row)]
    return alt.vconcat(*rows).resolve_scale(color="independent")


def regime_strips(results):
    """Regime strip over time for every k, one row per k."""
    return alt.vconcat(*[
        alt.Chart(ctx.assign(regime=ctx["regime"].astype(str))).mark_tick(thickness=1, size=12).encode(
            x=alt.X("date:T", title=""), y=alt.Y("regime:O", title=f"k={k}"), color=alt.Color("regime:N", legend=None),
        ).properties(width=800, height=18 * k)
        for k, ctx in results.items()
    ]).resolve_scale(color="independent")
