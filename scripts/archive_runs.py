"""
Move the runs, notebooks and files that are no longer part of the thesis to archive/, keeping their paths
(experiments/conv_vae/ -> archive/experiments/conv_vae/). Whole folders are moved, including the files that are not in
git (artifacts/ with weights and latents, figures/). Then rebuilds experiments/registry.csv from the runs that remain and
appends what was moved to archive/README.md, the only file of archive/ that is in git.

Run it once from the repository root, after committing everything else:
    python scripts/archive_runs.py
It prints the list and asks for confirmation before moving anything. Paths that no longer exist are skipped.
"""
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from src.experiments import rebuild_registry

hp = "experiments/temporal_vae/kmeans/hyperparameters"
base = "experiments/level_only/kmeans/baseline"
smooth = "experiments/temporal_vae/kmeans/latent_smoothing"
moves = {
    "experiments/conv_vae": "ConvVAE runs (GMM, k sweep, log vs linear price): the ConvVAE is no longer used",
    "experiments/temporal_vae/kmeans/vae_model": "Temporal VAE with 16 dims and blocks every 7 days, replaced by study z16",
    f"{hp}/005_real_price_log_kmeans_k6_z4_s14": "k = 6; experiments use k = 3, 4 and 5",
    f"{base}/001_real_price_log_kmeans_k2": "k = 2; experiments use k = 3, 4 and 5",
    f"{base}/005_real_price_log_kmeans_k6": "k = 6; experiments use k = 3, 4 and 5",
    f"{base}/006_real_price_log_kmeans_k2_7d": "k = 2; experiments use k = 3, 4 and 5",
    f"{base}/010_real_price_log_kmeans_k6_7d": "k = 6; experiments use k = 3, 4 and 5",
    "experiments/comparisons.csv": "output of notebook 04, replaced by metrics/",
    "experiments/comparisons_summary.csv": "output of notebook 04, replaced by metrics/",
    f"{smooth}/smoothing_results.csv": "old notebook 06 output; notebook 06 now saves regimes, metrics/ has the metrics",
    f"{smooth}/smoothing_summary.csv": "old notebook 06 output; notebook 06 now saves regimes, metrics/ has the metrics",
    f"{smooth}/regime_profiles.csv": "old notebook 06 output, replaced by metrics/regime_profiles.csv",
    "metrics/smoothing_metrics.csv": "old notebook 06 output; scripts/06_regime_metrics.py now reads the regimes",
    "notebooks/02_conv_vae.ipynb": "ConvVAE notebook: the ConvVAE is no longer used",
    "notebooks/04_comparisons.ipynb": "pairwise comparisons, replaced by metrics/ and scripts/06_regime_metrics.py",
    "check_extraction.py": "diagnostic of the continuous LSTM reading, closed (reverted)",
    "data/processed/scalograms/real_price": "scalograms without log, discarded (log real price is used)",
}
# hyperparameters/001 (k = 2) is kept: it is the trained_in folder of the z4_s14 model (its weights and latents).

existing = {p: why for p, why in moves.items() if (root / p).exists()}
if not existing:
    sys.exit("Nothing to move")
print("Will move to archive/:")
for p in existing:
    print(f"  {p}")
if input("Continue? [y/N] ").strip().lower() != "y":
    sys.exit("Nothing moved")

for p in existing:
    target = root / "archive" / p
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root / p), str(target))
n_runs = rebuild_registry()

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                            check=True).stdout.strip()
except (OSError, subprocess.CalledProcessError):
    commit = "unknown"
readme = root / "archive" / "README.md"
header = "" if readme.exists() else (
    "# Archive\n\nRuns, notebooks and files that are no longer part of the thesis. Only this file is in git; the rest "
    "stays on the computer that archived it and in the git history (last commit before archiving, below).\n")
lines = [f"\n## {date.today().isoformat()}, from commit {commit}\n"] + [f"- `{p}`: {why}\n" for p, why in existing.items()]
with readme.open("a", encoding="utf-8") as f:
    f.write(header + "".join(lines))
print(f"Moved {len(existing)} paths. registry.csv rebuilt with {n_runs} runs. See archive/README.md")
