"""
Metrics of the calendar-week runs (scalogram set real_price_log_calweek) with k = 3, 4 and 5, one row per week.
The definitions are those of src/metrics.py; scripts/06_regime_metrics.py measures the daily and weekly-by-day runs,
and nothing here mixes with them.

The calendar-week test uses one scalogram per week, Monday 00:00 to Sunday 23:00, with no overlap between weeks
(scripts/04_generate_scalograms.py --window calweek, 1386 weeks from 2000-01-03 to 2026-07-26). It is measured by
week, not by day:
    - A week belongs to the period (train, val or test) of its Monday (src.data.date_split).
    - Its price is the mean real price of its 7 days, and it is expensive when that price is above the 66.7, 75 or 80
      percentile of the weeks of the same period (k = 3, 4 and 5).
    - It is El Nino when the ONI of the month of its Thursday is >= 0.5; the episodes are the El Nino weeks of
      2014-2016 (named 2015-2016) and 2023-2024 (src.data.load_context and src.metrics.days_from_context).
    - Switches per year are counted with 52.18 weeks per year, so a regime can switch at most about 52 times a year.
    - The persistence filter of the top models compares with the weekly baseline (reference = "level").

Runs read (each folder of `runs` whose config has scalogram_set = real_price_log_calweek):
    level_only/calweek                 weekly baseline: mean and std of the log price of the week (level_baseline.ipynb)
    vae_lstm/calweek                   VAE-LSTM latents + mean and std of the week (04_vae_lstm_calweek.ipynb)
    vae_lstm/calweek_latents_only      the same models, clustered on the latents only (04_vae_lstm_calweek.ipynb)

Outputs (in metrics/calweek/):
    data_reference.csv    no model: expensive weeks and prices of each period and episode, per k
    regime_metrics.csv    one row per run and period (read by top_models.ipynb with metrics_file set to this file)
    regime_profiles.csv   one row per run and regime: weeks, share, real price, reservoir, ONI, El Nino share

and, in every VAE-LSTM run folder, tables/baseline_comparison.txt: the run, the other clustering input of the same
model and k (latents + level or latents only) and the weekly baseline with the same k, in one table for all weeks, one
for val and one for test.

The script stops if a run has no regimes.csv or a VAE-LSTM model was clustered with only one of the two inputs, so no
table mixes complete and missing runs.

Usage (run level_baseline.ipynb and 04_vae_lstm_calweek.ipynb first):
    python scripts/07_calweek_metrics.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from src import experiments as exp
from src.data import load_context, test_start, val_start
from src.metrics import (comparison_table, data_reference, days_from_context, periods, profile_rows, regime_metrics,
                         thresholds, weeks_per_year)

k_values = [3, 4, 5]
scalogram_set = "real_price_log_calweek"
runs = ["level_only/calweek", "vae_lstm/calweek", "vae_lstm/calweek_latents_only"]
view = "week"
out_dir = root / "metrics" / "calweek"
comparison_periods = ["all", "val", "test"]


def run_identity(config, run_dir):
    """Columns that identify a run. features is what k-means clustered: level (the baseline), latents + level or
    latents (clustering_input in config.json)."""
    if config["model"] == "level_only":
        features = "level"
    else:
        features = config.get("clustering_input", "latents + level")
    return {"model": config["model"], "study": config["study"], "run_id": exp.run_id(run_dir),
            "trained_in": config.get("trained_in") or None,
            "z_dim": config.get("z_dim"), "lam": config.get("lam"), "beta": config.get("beta"),
            "seed": config.get("seed") if "z_dim" in config else None, "k": config["n_regimes"],
            "features": features, "smoothed": False}


metadata = pd.read_csv(root / "data/processed/scalograms" / scalogram_set / "scalogram_metadata.csv",
                       parse_dates=["start_datetime", "end_datetime"])
context = load_context(root / "data/processed", metadata)
weeks = days_from_context(context)
week_dates = pd.DatetimeIndex(weeks["date"])

assignments, no_regimes = [], []
for study in runs:
    for path in sorted((exp.EXPERIMENTS_DIR / study).glob("*/config.json")):
        config = json.loads(path.read_text())
        if config.get("scalogram_set") != scalogram_set or config.get("n_regimes") not in k_values:
            continue
        if not (path.parent / "regimes.csv").exists():
            no_regimes.append(exp.run_id(path.parent))
            continue
        regimes = pd.read_csv(path.parent / "regimes.csv", parse_dates=["date"]).set_index("date")["regime"]
        regimes = regimes.reindex(week_dates)
        if regimes.isna().any():
            sys.exit(f"{exp.run_id(path.parent)} has no regime on {int(regimes.isna().sum())} weeks")
        assignments.append((run_identity(config, path.parent), regimes.astype(int).to_numpy()))
if no_regimes:
    sys.exit("These runs have no regimes.csv (unfinished run?), finish or remove them first:\n  " + "\n  ".join(no_regimes))
if not any(identity["model"] == "level_only" for identity, _ in assignments):
    sys.exit("No weekly baseline in experiments/level_only/calweek: run level_baseline.ipynb with the calweek set first")

# every trained model must have been clustered with both inputs and the same k
inputs_of = {}
for identity, _ in assignments:
    if identity["trained_in"]:
        inputs_of.setdefault((identity["trained_in"], identity["k"]), set()).add(identity["features"])
incomplete = sorted(f"{model} k={k}: only {sorted(f)}" for (model, k), f in inputs_of.items() if len(f) < 2)
if incomplete:
    sys.exit("These models were clustered with only one input, run 04_vae_lstm_calweek.ipynb again:\n  "
             + "\n  ".join(incomplete))

rows, profiles = [], []
for identity, regimes in assignments:
    run_weeks = weeks.assign(regime=regimes)
    profiles.append(profile_rows(context, regimes, identity))
    for period in periods:
        rows.append({**identity, **regime_metrics(run_weeks, identity["k"], view, period, weeks_per_year),
                     "matched": True})
results = pd.DataFrame(rows)

out_dir.mkdir(parents=True, exist_ok=True)
data_reference(weeks, k_values, views=(view,)).rename(columns={"days": "weeks", "nino_days": "nino_weeks"}).to_csv(
    out_dir / "data_reference.csv", index=False)
results.round(3).to_csv(out_dir / "regime_metrics.csv", index=False)
pd.concat(profiles, ignore_index=True).rename(columns={"days": "weeks"}).round(3).to_csv(
    out_dir / "regime_profiles.csv", index=False)


def run_comparison(identity):
    """tables/baseline_comparison.txt of one VAE-LSTM run (see the module docstring)."""
    k, run_id = identity["k"], identity["run_id"]
    same_k = results[results["k"] == k]
    this = same_k["run_id"] == run_id
    parts = [(f"This run ({identity['features']})", same_k[this]),
             ("Same model, other input", same_k[~this & (same_k["trained_in"] == identity["trained_in"])]),
             ("Weekly baseline", same_k[same_k["model"] == "level_only"])]
    table_rows = pd.concat([part.assign(row=name) for name, part in parts], ignore_index=True)
    labels = {"row": "Row", "features": "Features"}
    header = (f"{run_id}\nk = {k}, {len(weeks)} calendar weeks from {weeks['date'].min():%Y-%m-%d} to "
              f"{weeks['date'].max():%Y-%m-%d}; a week belongs to the period of its Monday: train before {val_start}, "
              f"val from {val_start}, test from {test_start}\n"
              f"Expensive week: mean real price of the week above percentile {thresholds[k]} of the weeks of the "
              f"same period\n"
              f"El Nino week: ONI of the month of its Thursday >= 0.5\n"
              f"Capture: share of the El Nino weeks of the episode in the most expensive regime; "
              f"lift: capture / top regime weeks % (1 = chance)\n"
              f"Precision, recall and F1 of the expensive weeks in the most expensive regime\n"
              f"Switches per year are counted inside each period, with {weeks_per_year:.2f} weeks per year; "
              f"- = the episode has no weeks in the period\n")
    tables = []
    for period in comparison_periods:
        part = table_rows[table_rows["period"] == period]
        n_weeks = len(weeks) if period == "all" else int((weeks["period"] == period).sum())
        limit = part["threshold_cop_kwh"].iloc[0]
        tables.append(f"Period: {period} ({n_weeks} weeks), expensive above {limit:.1f} COP/kWh\n"
                      + comparison_table(part, labels, unit="weeks").get_string())
    path = exp.EXPERIMENTS_DIR / run_id / "tables" / "baseline_comparison.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n\n".join(tables) + "\n", encoding="utf-8")


vae_runs = [identity for identity, _ in assignments if identity["trained_in"]]
for identity in vae_runs:
    run_comparison(identity)
print(f"{len(vae_runs)} runs got tables/baseline_comparison.txt")
print(f"{len(assignments)} runs on {len(weeks)} weeks from {weeks['date'].min():%Y-%m-%d}, {len(results)} rows -> "
      f"{out_dir.relative_to(root)}/")
