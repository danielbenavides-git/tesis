import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"

REGISTRY_COLUMNS = ["run_id", "study", "model", "k", "z_dim", "lam", "beta", "seed", "epochs", "trained_in", "git_commit"]


def create_run(model, study, scalogram_set, k, tag="", root=EXPERIMENTS_DIR):
    """New folder experiments/<model>/kmeans/<study>/NNN_<scalogram_set>_kmeans_k<k>[_<tag>]/.
    Its tables/, figures/ and artifacts/ (weights and latents, not in git) are created when something is saved
    in them, so no empty folder is left (a run without a model has no artifacts/).
    NNN is consecutive within each study folder and never reused. tag is an optional label for the folder name."""
    study_dir = Path(root) / model / "kmeans" / study
    study_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[:3]) for p in study_dir.iterdir() if p.is_dir() and p.name[:3].isdigit()]
    name = f"{max(numbers, default=0) + 1:03d}_{scalogram_set}_kmeans_k{k}" + (f"_{tag}" if tag else "")
    run_dir = study_dir / name
    run_dir.mkdir()
    return run_dir


def _subdir(run_dir, name):
    path = Path(run_dir) / name
    path.mkdir(exist_ok=True)
    return path


def run_id(run_dir):
    """Path of the run inside experiments/, e.g. conv_vae/kmeans/k_sweep/001_real_price_log_kmeans_k2."""
    run_dir = Path(run_dir).resolve()
    try:
        return run_dir.relative_to(EXPERIMENTS_DIR.resolve()).as_posix()
    except ValueError:
        return f"{run_dir.parent.name}/{run_dir.name}"


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def save_config(run_dir, config):
    _write_json(Path(run_dir) / "config.json", config)


def save_history(run_dir, history):
    _write_json(Path(run_dir) / "history.json", history)


def save_model(run_dir, model):
    torch.save(model.state_dict(), _subdir(run_dir, "artifacts") / "model.pt")


def save_latents(run_dir, latents):
    np.save(_subdir(run_dir, "artifacts") / "latents.npy", latents)


def load_trained(run_id, device="cpu", root=EXPERIMENTS_DIR):
    """Config, history, model weights and latents saved by an earlier run, to reuse its model without training."""
    run_dir = Path(root) / run_id
    return {
        "config": json.loads((run_dir / "config.json").read_text()),
        "history": json.loads((run_dir / "history.json").read_text()),
        "state_dict": torch.load(run_dir / "artifacts" / "model.pt", map_location=device),
        "latents": np.load(run_dir / "artifacts" / "latents.npy"),
    }


def save_table(run_dir, name, table, show=True):
    """Write a PrettyTable or DataFrame to tables/<name>.txt and print it unless show=False."""
    text = table.to_string() if isinstance(table, pd.DataFrame) else table.get_string()
    if show:
        print(text)
    (_subdir(run_dir, "tables") / f"{name}.txt").write_text(text + "\n", encoding="utf-8")


def figure_path(run_dir, name):
    return str(_subdir(run_dir, "figures") / f"{name}.png")


def git_commit(root=ROOT):
    """Short HEAD hash, with -dirty if src/ or scripts/ have uncommitted changes.
    Notebooks are not checked: their config cell is edited before every run and is saved in config.json."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def registry_row(run_dir, config, commit=""):
    """Registry row of one run: what was run, not its results (those are in metrics/)."""
    seed = config.get("seed", "") if "z_dim" in config else ""
    return {"run_id": run_id(run_dir), "study": config.get("study", ""), "model": config["model"],
            "k": config["n_regimes"], "z_dim": config.get("z_dim", ""), "lam": config.get("lam", ""),
            "beta": config.get("beta", ""), "seed": seed, "epochs": config.get("epochs", ""),
            "trained_in": config.get("trained_in", ""), "git_commit": commit}


def log_run(run_dir, config, registry=None):
    """Add (or replace) this run's row in experiments/registry.csv."""
    registry = Path(registry) if registry else EXPERIMENTS_DIR / "registry.csv"
    row = registry_row(run_dir, config, git_commit())
    table = pd.read_csv(registry, dtype=str, keep_default_na=False) if registry.exists() else pd.DataFrame()
    table = table.reindex(columns=REGISTRY_COLUMNS)
    table = table[table["run_id"] != row["run_id"]]
    table = pd.concat([table, pd.DataFrame([row]).astype(str)], ignore_index=True)[REGISTRY_COLUMNS]
    table.sort_values("run_id").to_csv(registry, index=False)
    return row


def rebuild_registry(registry=None):
    """Rewrite experiments/registry.csv with one row per run folder under experiments/. The git commit of each run is
    kept from the old registry when it was there."""
    registry = Path(registry) if registry else EXPERIMENTS_DIR / "registry.csv"
    commits = {}
    if registry.exists():
        old = pd.read_csv(registry, dtype=str, keep_default_na=False)
        if "git_commit" in old:
            commits = dict(zip(old["run_id"], old["git_commit"]))
    rows = []
    for path in sorted(EXPERIMENTS_DIR.glob("*/kmeans/*/*/config.json")):
        config = json.loads(path.read_text())
        rows.append(registry_row(path.parent, config, commits.get(run_id(path.parent), "")))
    pd.DataFrame(rows, columns=REGISTRY_COLUMNS).to_csv(registry, index=False)
    return len(rows)


def split_table(context, splits):
    t = PrettyTable(["Split", "Days", "From", "To"])
    for name, idx in splits.items():
        t.add_row([name, len(idx), context["date"].iloc[idx[0]].date(), context["date"].iloc[idx[-1]].date()])
    return t


def hyperparameter_table(config, n_params):
    t = PrettyTable(["Hyperparameter", "Value"])
    t.align["Hyperparameter"] = "l"
    rows = [
        ("Latent dimension", config["z_dim"]),
        ("Encoder filters", " -> ".join(str(f) for f in config["filters"])),
        ("Decoder filters", " -> ".join(str(f) for f in reversed(config["filters"]))),
        ("Beta (KL weight)", config["beta"]),
        ("Smoothness weight (lambda)", config.get("lam", "-")),
        ("KL warm-up epochs", config["kl_warmup_epochs"]),
        ("Optimizer, learning rate", f"{config['optimizer']}, {config['lr']:g}"),
        ("Batch size", config["batch_size"]),
        ("Epochs", config["epochs"]),
        ("Trainable parameters", f"{n_params:,}"),
    ]
    if "seq_len" in config:
        rows[-1:-1] = [
            ("Sequence length, stride (days)", f"{config['seq_len']}, {config['stride']}"),
            ("Recurrent layer, hidden size", f"{config['rnn_type'].upper()}, {config['rnn_hidden']}"),
        ]
    for row in rows:
        t.add_row(row)
    return t


def regime_profile_table(profile):
    t = PrettyTable(["Regime", "Days", "Share %", "Real price", "Reservoir", "Mean ONI", "El Nino %"])
    for _, row in profile.iterrows():
        t.add_row([int(row["regime"]), int(row["days"]), f"{row['share_pct']:.1f}", f"{row['mean_real_price']:.1f}",
                   f"{row['mean_reservoir']:.3f}", f"{row['mean_oni']:.2f}", f"{row['el_nino_pct']:.1f}"])
    return t


def metrics_table(metrics):
    t = PrettyTable(["Metric", "Value"])
    t.align["Metric"] = "l"
    for key, value in metrics.items():
        t.add_row([key, f"{value:.3f}" if isinstance(value, float) else value])
    return t
