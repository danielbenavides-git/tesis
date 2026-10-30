"""
Metrics of every regime assignment of the k-means runs with k = 3, 4 and 5. The definitions are in src/metrics.py.

Reads regimes.csv and config.json of every run in `runs`, and the regimes that notebook 06 saves for the inputs only it
has (scaled latents and smoothed latents): latent_smoothing/regimes.csv (one column per input) and inputs.csv (what each
column is). Each assignment is measured with both views of an expensive day (daily price and backward 7-day mean);
`matched` marks the view that fits the input: daily for unsmoothed inputs, 7d for smoothed ones.

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
scalogram_set = "real_price_log"
runs = ["level_only/kmeans/baseline", "temporal_vae/kmeans/hyperparameters", "temporal_vae/kmeans/lam_beta_grid",
        "temporal_vae/kmeans/z16"]
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


metadata = pd.read_csv(root / "data/processed/scalograms" / scalogram_set / "scalogram_metadata.csv",
                       parse_dates=["start_datetime", "end_datetime"])
context = load_context(root / "data/processed", metadata)
days = days_from_context(context)

assignments, trained = [], set()
for study in runs:
    for path in sorted((exp.EXPERIMENTS_DIR / study).glob("*/config.json")):
        config = json.loads(path.read_text())
        if config.get("scalogram_set") != scalogram_set or config.get("n_regimes") not in k_values:
            continue
        regimes = pd.read_csv(path.parent / "regimes.csv")["regime"].to_numpy()
        assignments.append((run_identity(config, path.parent), regimes))
        if config.get("trained_in"):
            trained.add(config["trained_in"])

if not (smoothing_dir / "inputs.csv").exists():
    sys.exit(f"{(smoothing_dir / 'inputs.csv').relative_to(root)} not found: run notebook 06 first")
inputs = pd.read_csv(smoothing_dir / "inputs.csv")
missing = sorted(trained - set(inputs["run_id"]))
if missing:
    sys.exit("Notebook 06 has not read these trained models, run it again first:\n  " + "\n  ".join(missing))
smoothed_regimes = pd.read_csv(smoothing_dir / "regimes.csv")
for _, row in inputs.iterrows():
    identity = row.drop("column").to_dict()
    if identity["k"] in k_values:
        assignments.append((identity, smoothed_regimes[row["column"]].to_numpy()))

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
print(f"{len(assignments)} assignments, {len(results)} rows -> metrics/")
