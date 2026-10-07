import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from .clustering import regime_profile
from .data import load_context
from .experiments import EXPERIMENTS_DIR

METRICS = {
    "silhouette": ("Silhouette", "separation", True),
    "davies_bouldin": ("Davies-Bouldin", "separation", False),
    "calinski_harabasz": ("Calinski-Harabasz", "separation", True),
    "mean_run_days": ("Mean run (days)", "persistence", True),
    "switches_per_year": ("Switches per year", "persistence", False),
}
GROUPS = ("separation", "persistence")
IGNORED_CONFIG_KEYS = ("model", "clustering", "scalogram_set", "n_regimes", "k_range")


def load_run(run_id, root=EXPERIMENTS_DIR):
    run_dir = Path(root) / run_id
    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"{run_id} has no metrics.json (run not finished or wrong id)")
    return {
        "run_id": run_id,
        "dir": run_dir,
        "config": json.loads((run_dir / "config.json").read_text()),
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
        "regimes": pd.read_csv(run_dir / "regimes.csv", parse_dates=["date"]),
    }


def run_setup(run):
    """What defines a run besides its hyperparameters: model, clustering, scalogram set and k."""
    c, m = run["config"], run["metrics"]
    return {"model": c["model"], "clustering": c["clustering"], "scalogram_set": c["scalogram_set"], "k": m["k"]}


def differences(a, b):
    """Setup fields and hyperparameters that differ between the two runs, as {key: (value_a, value_b)}."""
    left = {**run_setup(a), **{k: v for k, v in a["config"].items() if k not in IGNORED_CONFIG_KEYS}}
    right = {**run_setup(b), **{k: v for k, v in b["config"].items() if k not in IGNORED_CONFIG_KEYS}}
    return {key: (left.get(key), right.get(key)) for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


def _winner(value_a, value_b, higher_is_better, id_a, id_b):
    if np.isclose(value_a, value_b, rtol=1e-9, atol=0):
        return "tie"
    a_better = value_a > value_b if higher_is_better else value_a < value_b
    return id_a if a_better else id_b


def metric_table(a, b):
    """One row per metric with both values, the better direction and the winning run."""
    rows = []
    for key, (label, group, higher) in METRICS.items():
        va, vb = float(a["metrics"][key]), float(b["metrics"][key])
        rows.append({"metric": key, "label": label, "group": group, "better": "higher" if higher else "lower",
                     "value_a": va, "value_b": vb, "difference_b_minus_a": vb - va,
                     "winner": _winner(va, vb, higher, a["run_id"], b["run_id"])})
    return pd.DataFrame(rows)


def _group_result(table, group, id_a, id_b):
    sub = table[table["group"] == group]
    wins_a, wins_b = int((sub["winner"] == id_a).sum()), int((sub["winner"] == id_b).sum())
    winner = id_a if wins_a > wins_b else id_b if wins_b > wins_a else "tie"
    return wins_a, wins_b, winner


def summarize(a, b, table):
    """Winner per metric group and overall. The overall winner wins both groups, or one group with the other
    tied. If each run wins one group, the split is resolved by total metric wins and marked as split."""
    id_a, id_b = a["run_id"], b["run_id"]
    summary = {"run_a": id_a, "run_b": id_b,
               "changed": "; ".join(f"{k}: {va} -> {vb}" for k, (va, vb) in sorted(differences(a, b).items()))}
    group_winners = {}
    for group in GROUPS:
        wins_a, wins_b, winner = _group_result(table, group, id_a, id_b)
        summary.update({f"{group}_wins_a": wins_a, f"{group}_wins_b": wins_b, f"{group}_winner": winner})
        group_winners[group] = winner
    total_a, total_b = int((table["winner"] == id_a).sum()), int((table["winner"] == id_b).sum())
    summary.update({"total_wins_a": total_a, "total_wins_b": total_b})

    decided = {w for w in group_winners.values() if w != "tie"}
    if len(decided) == 1:
        overall, basis = decided.pop(), "groups"
    elif total_a != total_b:
        overall = id_a if total_a > total_b else id_b
        basis = "split groups, total wins" if decided else "total wins"
    else:
        overall, basis = "tie", "tie"
    summary.update({"overall_winner": overall, "basis": basis})

    won = {run: [METRICS[m][0] for m in table.loc[table["winner"] == run, "metric"]] for run in (id_a, id_b)}
    if overall == "tie":
        summary["verdict"] = "Tie: neither run wins more metrics."
    else:
        loser = id_b if overall == id_a else id_a
        summary["verdict"] = (f"{overall} is better ({basis}). It wins {', '.join(won[overall]) or 'no metric'}"
                              f"; {loser} wins {', '.join(won[loser]) or 'no metric'}.")
    return summary


def checks(a, b):
    """Warnings that limit the comparison."""
    notes = []
    sa, sb = run_setup(a), run_setup(b)
    if sa["k"] != sb["k"]:
        notes.append(f"Different k ({sa['k']} vs {sb['k']}): separation metrics tend to change with k by themselves.")
    if sa["scalogram_set"] != sb["scalogram_set"]:
        notes.append("Different scalogram sets: the input images and the daily level differ.")
    if sa["model"] != sb["model"]:
        notes.append("Different models: silhouette, Davies-Bouldin and Calinski-Harabasz are measured in each "
                     "model's own latent space, so only run length and switches per year compare cleanly.")
    return notes


def latents_match(a, b):
    """Compare latents.npy of two runs of the same model. None when either file is missing locally."""
    pa, pb = a["dir"] / "latents.npy", b["dir"] / "latents.npy"
    if run_setup(a)["model"] != run_setup(b)["model"] or not (pa.exists() and pb.exists()):
        return None
    la, lb = np.load(pa), np.load(pb)
    if la.shape != lb.shape:
        return {"same": False, "max_abs_diff": None}
    diff = float(np.abs(la - lb).max())
    return {"same": diff < 1e-5, "max_abs_diff": diff}


def label_agreement(a, b):
    """Share of days in the same regime. Regimes are ordered by mean price, so labels match when k is equal."""
    if run_setup(a)["k"] != run_setup(b)["k"]:
        return None
    merged = a["regimes"][["date", "regime"]].merge(b["regimes"][["date", "regime"]], on="date", suffixes=("_a", "_b"))
    return float((merged["regime_a"] == merged["regime_b"]).mean())


def profile_side_by_side(a, b, data_dir):
    """Regime profiles of both runs joined on regime number."""
    dates = a["regimes"]["date"]
    context = load_context(data_dir, pd.DataFrame({"start_datetime": dates}))
    profiles = []
    for run, tag in ((a, "a"), (b, "b")):
        ctx = context.merge(run["regimes"][["date", "regime"]], on="date", how="left")
        profiles.append(regime_profile(ctx).set_index("regime").add_suffix(f"_{tag}"))
    return profiles[0].join(profiles[1], how="outer").reset_index()


def compare(id_a, id_b, data_dir, root=EXPERIMENTS_DIR):
    a, b = load_run(id_a, root), load_run(id_b, root)
    table = metric_table(a, b)
    return {"a": a, "b": b, "table": table, "summary": summarize(a, b, table), "checks": checks(a, b),
            "latents": latents_match(a, b), "agreement": label_agreement(a, b),
            "profile": profile_side_by_side(a, b, data_dir)}


def _fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:,.3f}" if isinstance(value, float) else str(value)


def report(result):
    """Print the differences, metric table, regime profiles, checks and verdict of one comparison."""
    a, b, s = result["a"], result["b"], result["summary"]
    print(f"A = {a['run_id']}\nB = {b['run_id']}\n")

    t = PrettyTable(["Setting", "A", "B"])
    for key, (va, vb) in sorted(differences(a, b).items()):
        t.add_row([key, va, vb])
    print("What changes between the runs")
    print(t if t.rows else "Nothing (same setup and hyperparameters)")

    t = PrettyTable(["Metric", "Group", "Better", "A", "B", "Winner"])
    for _, row in result["table"].iterrows():
        winner = "A" if row["winner"] == a["run_id"] else "B" if row["winner"] == b["run_id"] else "tie"
        t.add_row([row["label"], row["group"], row["better"], _fmt(row["value_a"]), _fmt(row["value_b"]), winner])
    print("\nClustering metrics")
    print(t)

    p = result["profile"]
    t = PrettyTable(["Regime", "Days A", "Days B", "Price A", "Price B", "Reservoir A", "Reservoir B",
                     "ONI A", "ONI B", "El Nino % A", "El Nino % B"])
    formats = (("days", "{:,.0f}"), ("mean_real_price", "{:.1f}"), ("mean_reservoir", "{:.3f}"),
               ("mean_oni", "{:.2f}"), ("el_nino_pct", "{:.1f}"))
    for _, row in p.iterrows():
        cells = [int(row["regime"])]
        for col, fmt in formats:
            cells += ["-" if pd.isna(row[f"{col}_{tag}"]) else fmt.format(row[f"{col}_{tag}"]) for tag in ("a", "b")]
        t.add_row(cells)
    print("\nRegime profiles")
    print(t)

    if result["agreement"] is not None:
        print(f"\nDays assigned to the same regime: {result['agreement']:.1%}")
    if result["latents"] is not None:
        state = "identical" if result["latents"]["same"] else "different"
        print(f"Latents of the two runs: {state} (max abs diff {_fmt(result['latents']['max_abs_diff'])})")
    elif run_setup(a)["model"] == run_setup(b)["model"]:
        print("Latents of the two runs: not checked (latents.npy missing locally)")
    for note in result["checks"]:
        print(f"Note: {note}")
    print(f"\nVerdict: {s['verdict']}")


def summary_table(summaries):
    t = PrettyTable(["Run A", "Run B", "Separation", "Persistence", "Total wins A-B", "Overall winner"])
    t.align = "l"
    for s in summaries:
        t.add_row([s["run_a"], s["run_b"],
                   f"{s['separation_wins_a']}-{s['separation_wins_b']} ({_short(s, 'separation_winner')})",
                   f"{s['persistence_wins_a']}-{s['persistence_wins_b']} ({_short(s, 'persistence_winner')})",
                   f"{s['total_wins_a']}-{s['total_wins_b']}", _short(s, "overall_winner")])
    return t


def _short(s, key):
    return "A" if s[key] == s["run_a"] else "B" if s[key] == s["run_b"] else "tie"


def _upsert(path, new, key_cols):
    if path.exists():
        old = pd.read_csv(path)
        keys = set(map(tuple, new[key_cols].astype(str).to_numpy()))
        old = old[[tuple(r) not in keys for r in old[key_cols].astype(str).to_numpy()]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def save_comparisons(results, root=EXPERIMENTS_DIR):
    """Write comparisons.csv (one row per pair and metric) and comparisons_summary.csv (one row per pair).
    Rows of a pair that was compared before are replaced."""
    root = Path(root)
    date = datetime.now().isoformat(timespec="seconds")
    long = pd.concat([r["table"].assign(run_a=r["a"]["run_id"], run_b=r["b"]["run_id"], date=date)
                      for r in results], ignore_index=True)
    long = long[["date", "run_a", "run_b", "metric", "group", "better", "value_a", "value_b",
                 "difference_b_minus_a", "winner"]]
    summary = pd.DataFrame([{"date": date, **r["summary"],
                             "same_regime_share": r["agreement"],
                             "latents_identical": None if r["latents"] is None else r["latents"]["same"],
                             "notes": " ".join(r["checks"])} for r in results])
    _upsert(root / "comparisons.csv", long, ["run_a", "run_b", "metric"])
    return _upsert(root / "comparisons_summary.csv", summary, ["run_a", "run_b"])
