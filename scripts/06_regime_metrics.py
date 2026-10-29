"""
Size-aware metrics of the most expensive regime for the k-means runs with k = 3, 4 and 5.

Reads the regimes.csv and config.json of every run in `runs` and the rows of metrics/smoothing_metrics.csv
(written by notebook 06) for the inputs that only notebook 06 has (scaled latents and smoothed latents), and computes for each run both views of an expensive day (daily price and backward 7-day
mean). `matched` marks the view that fits the input: daily for unsmoothed inputs, 7d for smoothed ones.
The definitions are in src/metrics.py.

Outputs (in metrics/):
    data_reference.csv    level 1, no model: expensive days and prices of each episode, per k and view
    regime_metrics.csv    level 2: one row per run, input, k and view
    regime_metrics.txt    one section per k: the data reference and the matched view of every configuration
    ranking.csv           one row per configuration and k: mean and std over seeds, rank by criterion A and C
    ranking.txt           one section per k: the baselines, the top 5 by criterion A and the top 5 by criterion C

Usage (run notebook 06 first so its rows are included):
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
from src.metrics import (data_reference, days_from_context, metrics_table, persistence_factor, ranking,
                         ranking_table, reference_table, regime_metrics)

k_values = [3, 4, 5]
scalogram_set = "real_price_log"
runs = ["level_only/kmeans/baseline", "temporal_vae/kmeans/hyperparameters", "temporal_vae/kmeans/lam_beta_grid"]
out_dir = root / "metrics"


def run_identity(config, run_dir):
    if config["model"] == "level_only":
        smoothed = config.get("level_window", 1) > 1
        features, latents = ("level 7d" if smoothed else "level"), "-"
    else:
        smoothed, features, latents = False, "latents + level", "raw"
    return {"model": config["model"], "study": config["study"], "run_id": exp.run_id(run_dir),
            "lam": config.get("lam"), "beta": config.get("beta"),
            "seed": config.get("seed") if "z_dim" in config else None, "k": config["n_regimes"],
            "features": features, "latents": latents, "smoothed": smoothed}


metadata = pd.read_csv(root / "data/processed/scalograms" / scalogram_set / "scalogram_metadata.csv",
                       parse_dates=["start_datetime", "end_datetime"])
days = days_from_context(load_context(root / "data/processed", metadata))
reference = data_reference(days, k_values)

rows = []
for study in runs:
    for path in sorted((exp.EXPERIMENTS_DIR / study).glob("*/config.json")):
        config = json.loads(path.read_text())
        if config.get("scalogram_set") != scalogram_set or config.get("n_regimes") not in k_values:
            continue
        regimes = pd.read_csv(path.parent / "regimes.csv", parse_dates=["date"], keep_default_na=False)
        run_days = days.assign(regime=regimes["regime"].to_numpy())
        identity = run_identity(config, path.parent)
        for view in ("daily", "7d"):
            rows.append({**identity, **regime_metrics(run_days, identity["k"], view),
                         "matched": (view == "7d") == identity["smoothed"]})
results = pd.DataFrame(rows)

smoothing_csv = out_dir / "smoothing_metrics.csv"
if smoothing_csv.exists():
    smoothing = pd.read_csv(smoothing_csv)
    if "switches_per_year" not in smoothing.columns:
        sys.exit(f"{smoothing_csv.relative_to(root)} was written by an older version: run notebook 06 again first")
    # level 7d and raw latents + level are already in the runs above; only the inputs new to notebook 06 are added
    new = (smoothing["features"] != "level 7d") & ~((smoothing["features"] == "latents + level") & (smoothing["latents"] == "raw"))
    results = pd.concat([results, smoothing[new]], ignore_index=True)
else:
    print(f"{smoothing_csv.relative_to(root)} not found: run notebook 06 to include the smoothed inputs")

out_dir.mkdir(exist_ok=True)
reference.to_csv(out_dir / "data_reference.csv", index=False)
results.round(3).to_csv(out_dir / "regime_metrics.csv", index=False)
sections = []
for k in k_values:
    sections.append(f"k = {k}, expensive day = real price above percentile of the series\n"
                    f"{reference_table(reference, k)}\n\n{metrics_table(results, k)}")
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
print(f"{len(results)} rows -> {(out_dir / 'regime_metrics.csv').relative_to(root)}, data_reference.csv, regime_metrics.txt")
