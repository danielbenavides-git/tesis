"""Steps shared by notebooks 02 and 03 after the model is trained: one run folder per k, clustering, figures,
tables and the registry row. The model is trained (or loaded) once and every k reuses its latents."""
import altair as alt
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from src import experiments as exp
from src.clustering import fit_clusters, order_regimes, regime_profile, save_regimes, summary_metrics
from src.plots import regime_timeline, selection_chart, tsne_chart

# Keys taken from the current notebook when load_from reuses an earlier run. Everything else (set, filters,
# epochs...) comes from the loaded run, so the config always describes the model that produced the latents.
RUN_KEYS = ("study", "clustering", "k_values", "k_range")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def prepare_runs(config, load_from=None, device="cpu"):
    """Create one run folder per k in config["k_values"] and save each config.
    With load_from (a run_id), the model, history and latents of that run are reused and training is skipped.
    Returns (config, run_dirs, run_configs, trained); trained is None when the model has to be trained."""
    trained = None
    if load_from:
        trained = exp.load_trained(load_from, device)
        config = {**trained["config"], **{key: config[key] for key in RUN_KEYS}}
    run_dirs = {
        k: exp.create_run(config["clustering"], config["scalogram_set"], config["clustering"], k,
                          root=exp.EXPERIMENTS_DIR / config["model"])
        for k in config["k_values"]
    }
    first = exp.run_id(run_dirs[config["k_values"][0]])
    trained_in = trained["config"].get("trained_in", load_from) if trained else first
    run_configs = {k: {**config, "n_regimes": k, "trained_in": trained_in} for k in run_dirs}
    for k, run_dir in run_dirs.items():
        exp.save_config(run_dir, run_configs[k])
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
        save_regimes(run_dir, context, labels, probs, metrics)
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
            "nino_2015_16_in_top_pct": nino_share_in_top(r["context"], [2015, 2016]),
            "nino_2023_24_in_top_pct": nino_share_in_top(r["context"], [2023, 2024]),
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
