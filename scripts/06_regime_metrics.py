"""
Metrics of every regime assignment of the k-means runs with k = 3, 4 and 5. The definitions are in src/metrics.py.

Reads regimes.csv and config.json of every run in `runs`, and the regimes that notebook 06 saves for the inputs only it
has (scaled latents and smoothed latents): latent_smoothing/regimes.csv (one column per input) and inputs.csv (what each
column is). Each assignment is measured with both views of an expensive day (daily price and backward 7-day mean);
`matched` marks the view that fits the input: daily for unsmoothed inputs, 7d for smoothed ones.

Runs of every set in `scalogram_sets` are read. Every assignment is measured on the same days: the dates that all those
sets have (the weekly set has no scalogram for the first 6 days, so all models and baselines start on 2000-01-07).

The script stops if a trained model of `runs` is missing from notebook 06, so no table mixes complete and missing inputs.

Outputs (in metrics/):
    data_reference.csv    no model: expensive days and prices of each episode, per k and view
    regime_metrics.csv    one row per assignment and view
    regime_metrics.txt    one section per k: the data reference and the matched view of every configuration
    regime_profiles.csv   one row per assignment and regime: days, share, real price, reservoir, ONI, El Nino share
    ranking.csv           one row per configuration and k: mean and std over seeds, rank by criterion A and C
    ranking.txt           one section per k: the baselines, the top 5 by criterion A and the top 5 by criterion C

Usage (run notebook 06 first):
    python scripts/06_regime_metrics.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from src import experiments as exp
from src.data import load_context
from src.metrics import (data_reference, days_from_context, metrics_table, persistence_factor, profile_rows, ranking,
                         ranking_table, reference_table, regime_metrics)

k_values = [3, 4, 5]
scalogram_sets = ["real_price_log", "real_price_log_weekly"]  # the first one gives the daily context
runs = ["level_only/kmeans/baseline", "temporal_vae/kmeans/hyperparameters", "temporal_vae/kmeans/lam_beta_grid",
        "temporal_vae/kmeans/dimensions", "temporal_vae/kmeans/small_filters", "temporal_vae/kmeans/weekly"]
smoothing_dir = exp.EXPERIMENTS_DIR / "temporal_vae/kmeans/latent_smoothing"
out_dir = root / "metrics"


def run_identity(config, run_dir):
    if config["model"] == "level_only":
        smoothed = config.get("level_window", 1) > 1
        features, latents = ("level 7d" if smoothed else "level"), "-"
    else:
        smoothed, features, latents = False, "latents + level", "raw"
    return {"model": config["model"], "study": config["study"], "run_id": exp.run_id(run_dir),
            "z_dim": config.get("z_dim"), "lam": config.get("lam"), "beta": config.get("beta"),
            "seed": config.get("seed") if "z_dim" in config else None, "k": config["n_regimes"],
            "features": features, "latents": latents, "smoothed": smoothed}


def set_dates(scalogram_set):
    metadata = pd.read_csv(root / "data/processed/scalograms" / scalogram_set / "scalogram_metadata.csv",
                           parse_dates=["start_datetime", "end_datetime"])
    return metadata, metadata["start_datetime"].dt.normalize()


def on_common_dates(regimes, name):
    """regimes (a Series indexed by date) on the common dates; stops if one of them has no regime."""
    regimes = regimes.reindex(common_dates)
    if regimes.isna().any():
        sys.exit(f"{name} has no regime on {int(regimes.isna().sum())} of the common dates")
    return regimes.astype(int).to_numpy()


metadata, dates = set_dates(scalogram_sets[0])
common_dates = pd.DatetimeIndex(dates)
for scalogram_set in scalogram_sets[1:]:
    common_dates = common_dates.intersection(set_dates(scalogram_set)[1])
context = load_context(root / "data/processed", metadata)
context = context[context["date"].isin(common_dates)].reset_index(drop=True)
days = days_from_context(context)

assignments, trained, no_regimes = [], set(), []
for study in runs:
    for path in sorted((exp.EXPERIMENTS_DIR / study).glob("*/config.json")):
        config = json.loads(path.read_text())
        if config.get("scalogram_set") not in scalogram_sets or config.get("n_regimes") not in k_values:
            continue
        if not (path.parent / "regimes.csv").exists():
            no_regimes.append(exp.run_id(path.parent))
            continue
        regimes = pd.read_csv(path.parent / "regimes.csv", parse_dates=["date"]).set_index("date")["regime"]
        assignments.append((run_identity(config, path.parent), on_common_dates(regimes, exp.run_id(path.parent))))
        if config.get("trained_in"):
            trained.add(config["trained_in"])
if no_regimes:
    sys.exit("These runs have no regimes.csv (unfinished run?), finish or remove them first:\n  " + "\n  ".join(no_regimes))

if not (smoothing_dir / "inputs.csv").exists():
    sys.exit(f"{(smoothing_dir / 'inputs.csv').relative_to(root)} not found: run notebook 06 first")
inputs = pd.read_csv(smoothing_dir / "inputs.csv")
missing = sorted(trained - set(inputs["run_id"]))
if missing:
    sys.exit("Notebook 06 has not read these trained models, run it again first:\n  " + "\n  ".join(missing))
smoothed_regimes = pd.read_csv(smoothing_dir / "regimes.csv", parse_dates=["date"]).set_index("date")
for _, row in inputs.iterrows():
    identity = row.drop("column").to_dict()
    if identity["k"] in k_values:
        regimes = on_common_dates(smoothed_regimes[row["column"]], f"{row['column']} of notebook 06 ({row['run_id']})")
        assignments.append((identity, regimes))

rows, profiles = [], []
for identity, regimes in assignments:
    run_days = days.assign(regime=regimes)
    profiles.append(profile_rows(context, regimes, identity))
    for view in ("daily", "7d"):
        rows.append({**identity, **regime_metrics(run_days, identity["k"], view),
                     "matched": (view == "7d") == bool(identity["smoothed"])})
results = pd.DataFrame(rows)

out_dir.mkdir(exist_ok=True)
reference = data_reference(days, k_values)
reference.to_csv(out_dir / "data_reference.csv", index=False)
results.round(3).to_csv(out_dir / "regime_metrics.csv", index=False)
pd.concat(profiles, ignore_index=True).round(3).to_csv(out_dir / "regime_profiles.csv", index=False)

sections = [f"k = {k}, expensive day = real price above percentile of the series\n"
            f"{reference_table(reference, k)}\n\n{metrics_table(results, k)}" for k in k_values]
(out_dir / "regime_metrics.txt").write_text("\n\n\n".join(sections) + "\n", encoding="utf-8")

rankings = [ranking(results, k) for k in k_values]
pd.concat(rankings, ignore_index=True).round(3).to_csv(out_dir / "ranking.csv", index=False)
sections = []
for k, stats in zip(k_values, rankings):
    limit = stats["switches_limit"].iloc[0]
    sections.append(
        f"k = {k}\nBaselines (reference, not ranked)\n{ranking_table(stats)}\n\n"
        f"Top 5 by criterion A (expensive F1)\n{ranking_table(stats, 'rank_a')}\n\n"
        f"Top 5 by criterion C (switches / yr <= {persistence_factor:g} x 7-day baseline = {limit:.1f}, then expensive F1)\n"
        f"{ranking_table(stats, 'rank_c')}")
(out_dir / "ranking.txt").write_text("\n\n\n".join(sections) + "\n", encoding="utf-8")
print(f"{len(assignments)} assignments on {len(days)} days from {days['date'].min():%Y-%m-%d}, {len(results)} rows -> metrics/")
