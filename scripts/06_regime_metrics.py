"""
Metrics of every regime assignment of the k-means runs with k = 3, 4 and 5. The definitions are in src/metrics.py.

Reads regimes.csv and config.json of every run in `runs`, and the regimes that notebook 06 saves for the inputs only it
has (smoothed latents): latent_smoothing/regimes.csv (one column per input) and inputs.csv (what each
column is). Each assignment is measured with both views of an expensive day (daily price and backward 7-day mean);
`matched` marks the view that fits the input: daily for unsmoothed inputs, 7d for smoothed ones (the `smoothed`
column of inputs.csv; for the weekly models, `latents + level 7d` is smoothed). Each one is also measured on every
period of src.metrics.periods (all days and the train, val and test days of src.data.date_split).

Runs of every set in `scalogram_sets` are read. Every assignment is measured on the same days: the dates that all those
sets have (the weekly set has no scalogram for the first 6 days, so all models and baselines start on 2000-01-07).

The script stops if a trained model of `runs` is missing from notebook 06, so no table mixes complete and missing inputs.

Outputs (in metrics/):
    data_reference.csv    no model: expensive days and prices of each episode, per k and view
    regime_metrics.csv    one row per assignment, view and period (read by notebook 07, which shows the top
                          configurations)
    regime_profiles.csv   one row per assignment and regime: days, share, real price, reservoir, ONI, El Nino share

and, in every Temporal VAE run folder, tables/baseline_comparison.txt: the run as notebook 03 clustered it, the
notebook 06 inputs of the same model and k, and the daily and 7-day baselines with the same k, each in its matched view,
in one table per period in comparison_periods.

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
from src.data import test_start, val_start
from src.metrics import (comparison_table, data_reference, days_from_context, periods, profile_rows, regime_metrics,
                         thresholds)

k_values = [3, 4, 5]
scalogram_sets = ["real_price_log", "real_price_log_weekly"]  # the first one gives the daily context
runs = ["level_only/kmeans/baseline", "temporal_vae/kmeans/hyperparameters", "temporal_vae/kmeans/lam_beta_grid",
        "temporal_vae/kmeans/dimensions", "temporal_vae/kmeans/weekly"]
smoothing_dir = exp.EXPERIMENTS_DIR / "temporal_vae/kmeans/latent_smoothing"
out_dir = root / "metrics"
comparison_periods = ["all", "val", "test"]


def run_identity(config, run_dir):
    if config["model"] == "level_only":
        smoothed = config.get("level_window", 1) > 1
        features = "level 7d" if smoothed else "level"
    else:
        smoothed, features = False, "latents + level"
    return {"model": config["model"], "study": config["study"], "run_id": exp.run_id(run_dir),
            "z_dim": config.get("z_dim"), "lam": config.get("lam"), "beta": config.get("beta"),
            "seed": config.get("seed") if "z_dim" in config else None, "k": config["n_regimes"],
            "features": features, "smoothed": smoothed}


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

assignments, trained, no_regimes, vae_runs = [], set(), [], []
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
            vae_runs.append((exp.run_id(path.parent), config))
if no_regimes:
    sys.exit("These runs have no regimes.csv (unfinished run?), finish or remove them first:\n  " + "\n  ".join(no_regimes))

if not (smoothing_dir / "inputs.csv").exists():
    sys.exit(f"{(smoothing_dir / 'inputs.csv').relative_to(root)} not found: run notebook 06 first")
try:
    inputs = pd.read_csv(smoothing_dir / "inputs.csv")
except pd.errors.EmptyDataError:  # notebook 06 found no trained model
    inputs = pd.DataFrame(columns=["column", "run_id", "k"])
missing = sorted(trained - set(inputs["run_id"]))
if missing:
    sys.exit("Notebook 06 has not read these trained models, run it again first:\n  " + "\n  ".join(missing))
smoothed_regimes = (pd.read_csv(smoothing_dir / "regimes.csv", parse_dates=["date"]).set_index("date")
                    if len(inputs) else None)
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
        for period in periods:
            rows.append({**identity, **regime_metrics(run_days, identity["k"], view, period),
                         "matched": (view == "7d") == bool(identity["smoothed"])})
results = pd.DataFrame(rows)

out_dir.mkdir(exist_ok=True)
reference = data_reference(days, k_values)
reference.to_csv(out_dir / "data_reference.csv", index=False)
results.round(3).to_csv(out_dir / "regime_metrics.csv", index=False)
pd.concat(profiles, ignore_index=True).round(3).to_csv(out_dir / "regime_profiles.csv", index=False)


def run_comparison(run_id, config):
    """tables/baseline_comparison.txt of one Temporal VAE run (see the module docstring)."""
    k = config["n_regimes"]
    matched = results[results["matched"] & (results["k"] == k)]
    as_run = matched["features"] == "latents + level"
    parts = [("Run (notebook 03)", matched[as_run & (matched["run_id"] == run_id)]),
             ("Notebook 06", matched[~as_run & (matched["run_id"] == config["trained_in"])]),
             ("Daily baseline", matched[matched["features"] == "level"]),
             ("7-day baseline", matched[matched["features"] == "level 7d"])]
    rows = pd.concat([part.assign(row=name) for name, part in parts], ignore_index=True)
    labels = {"row": "Row", "features": "Features", "view": "View"}
    first, last = days["date"].min(), days["date"].max()
    header = (f"{run_id}\nk = {k}, {len(days)} days from {first:%Y-%m-%d} to {last:%Y-%m-%d}; "
              f"train before {val_start}, val from {val_start}, test from {test_start}\n"
              f"Expensive day: real price above percentile {thresholds[k]} of the days of the same period; "
              f"the 7d view uses the backward 7-day mean of the price\n"
              f"Capture: share of the El Nino days of the episode in the most expensive regime; "
              f"lift: capture / top regime days % (1 = chance)\n"
              f"Precision, recall and F1 of the expensive days in the most expensive regime\n"
              f"Switches per year are counted inside each period; - = the episode has no days in the period\n")
    tables = []
    for period in comparison_periods:
        part = rows[rows["period"] == period]
        n_days = len(days) if period == "all" else int((days["period"] == period).sum())
        limits = ", ".join(f"{view} view {part.loc[part['view'] == view, 'threshold_cop_kwh'].iloc[0]:.1f}"
                           for view in ("daily", "7d") if (part["view"] == view).any())
        tables.append(f"Period: {period} ({n_days} days), expensive above {limits} COP/kWh\n"
                      + comparison_table(part, labels).get_string())
    path = exp.EXPERIMENTS_DIR / run_id / "tables" / "baseline_comparison.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n\n".join(tables) + "\n", encoding="utf-8")


for run_id, config in vae_runs:
    run_comparison(run_id, config)
print(f"{len(vae_runs)} runs got tables/baseline_comparison.txt")
print(f"{len(assignments)} assignments on {len(days)} days from {days['date'].min():%Y-%m-%d}, {len(results)} rows -> metrics/")
