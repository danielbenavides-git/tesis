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

REGISTRY_COLUMNS = [
    "run_id", "date", "model", "scalogram_set", "clustering", "k", "filters", "z_dim", "beta", "lam",
    "kl_warmup_epochs", "epochs", "silhouette", "davies_bouldin", "calinski_harabasz",
    "mean_run_days", "switches_per_year", "git_commit",
]


def create_run(model, scalogram_set, clustering, k, root=EXPERIMENTS_DIR):
    """New folder experiments/<model>/NNN_<scalogram_set>_<clustering>_k<k>/ with tables/, figures/ and
    artifacts/ (weights and latents, not in git).
    NNN is consecutive within each model and never reused."""
    model_dir = Path(root) / model
    model_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[:3]) for p in model_dir.iterdir() if p.is_dir() and p.name[:3].isdigit()]
    k_label = f"k{k}" if k else "kbic"
    run_dir = model_dir / f"{max(numbers, default=0) + 1:03d}_{scalogram_set}_{clustering}_{k_label}"
    (run_dir / "tables").mkdir(parents=True)
    (run_dir / "figures").mkdir()
    (run_dir / "artifacts").mkdir()
    return run_dir


def run_id(run_dir):
    """Path of the run inside experiments/, e.g. conv_vae/kmeans/001_real_price_kmeans_k2."""
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


def save_metrics(run_dir, metrics):
    _write_json(Path(run_dir) / "metrics.json", metrics)


def save_model(run_dir, model):
    torch.save(model.state_dict(), Path(run_dir) / "artifacts" / "model.pt")


def save_latents(run_dir, latents):
    np.save(Path(run_dir) / "artifacts" / "latents.npy", latents)


def save_table(run_dir, name, table):
    """Print a PrettyTable and write it to tables/<name>.txt."""
    print(table)
    (Path(run_dir) / "tables" / f"{name}.txt").write_text(table.get_string() + "\n", encoding="utf-8")


def figure_path(run_dir, name):
    return str(Path(run_dir) / "figures" / f"{name}.png")


def git_commit(root=ROOT):
    """Short HEAD hash, with -dirty if src/ or scripts/ have uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def log_run(run_dir, config, metrics, registry=None):
    """Add (or replace) this run's row in experiments/registry.csv and return it."""
    run_dir = Path(run_dir)
    registry = Path(registry) if registry else run_dir.parent.parent / "registry.csv"
    row = {
        "run_id": run_id(run_dir),
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": config["model"],
        "scalogram_set": config["scalogram_set"],
        "clustering": config["clustering"],
        "k": metrics["k"],
        "filters": "-".join(str(f) for f in config["filters"]),
        "z_dim": config["z_dim"],
        "beta": config["beta"],
        "lam": config.get("lam", ""),
        "kl_warmup_epochs": config["kl_warmup_epochs"],
        "epochs": config["epochs"],
        **{key: round(metrics[key], 4) for key in
           ("silhouette", "davies_bouldin", "calinski_harabasz", "mean_run_days", "switches_per_year")},
        "git_commit": git_commit(),
    }
    table = pd.read_csv(registry, dtype=str) if registry.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)
    table = table[table["run_id"] != row["run_id"]]
    table = pd.concat([table, pd.DataFrame([row]).astype(str)], ignore_index=True)[REGISTRY_COLUMNS]
    table.to_csv(registry, index=False)
    return row


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


def selection_table(selection):
    criterion = "bic" if "bic" in selection else "inertia"
    t = PrettyTable(["k", "BIC" if criterion == "bic" else "Inertia", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])
    for _, row in selection.iterrows():
        t.add_row([int(row["k"]), f"{row[criterion]:,.0f}", f"{row['silhouette']:.3f}",
                   f"{row['davies_bouldin']:.3f}", f"{row['calinski_harabasz']:,.0f}"])
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
