"""
Show which days fall in each regime of a run, read only from its regimes.csv.

For each regime: days, share of all days, mean real price, El Nino days and the days of each El Nino episode
(2015-2016 and 2023-2024). Then the share of each episode captured by the most expensive regime, the switches
per year and, with --missed, the El Nino days of the episodes that fell outside the most expensive regime.

regimes.csv has one row per day with these columns:
    date, split, regime, real_price, oni,
    nino        1 if the ONI of the month is >= 0.5
    episode     2015-2016, 2023-2024 or empty
    top_regime  1 if the regime is the most expensive one
    captured    1 if nino = 1 and top_regime = 1

Usage:
    python scripts/inspect_regimes.py experiments/temporal_vae/kmeans/vae_model/003_real_price_log_kmeans_k4
    python scripts/inspect_regimes.py path/to/regimes.csv --missed
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from prettytable import PrettyTable

root = Path(__file__).resolve().parent.parent
episodes = ("2015-2016", "2023-2024")


def read_regimes(arg):
    path = Path(arg)
    if not path.exists():
        path = root / "experiments" / arg
    if path.is_dir():
        path = path / "regimes.csv"
    table = pd.read_csv(path, parse_dates=["date"], keep_default_na=False)
    missing = {"split", "real_price", "nino", "episode", "top_regime", "captured"} - set(table.columns)
    if missing:
        sys.exit(f"{path} has no {sorted(missing)} columns")
    return path, table


def date_ranges(dates):
    """Consecutive dates grouped as 'first to last (n days)', single days as the date alone."""
    dates = pd.Series(sorted(dates))
    groups = (dates.diff() != pd.Timedelta(days=1)).cumsum()
    return [f"{g.iloc[0].date()}" if len(g) == 1 else f"{g.iloc[0].date()} to {g.iloc[-1].date()} ({len(g)} days)"
            for _, g in dates.groupby(groups)]


def report(path, table, missed):
    n, k = len(table), table["regime"].nunique()
    top = int(table.loc[table["top_regime"] == 1, "regime"].iloc[0])
    run = path.parent.resolve()
    experiments = (root / "experiments").resolve()
    print(f"Run: {run.relative_to(experiments).as_posix() if run.is_relative_to(experiments) else run}")
    print(f"Days: {n}, {table['date'].min().date()} to {table['date'].max().date()}, k = {k}\n")

    t = PrettyTable(["Regime", "Days", "Share %", "Mean real price", "El Nino days", "El Nino % of the regime",
                     *[f"{e} days" for e in episodes]])
    for regime, g in table.groupby("regime"):
        t.add_row([f"{regime}{' (most expensive)' if regime == top else ''}", len(g), f"{100 * len(g) / n:.1f}",
                   f"{g['real_price'].mean():.1f}", int(g["nino"].sum()), f"{100 * g['nino'].mean():.1f}",
                   *[int((g["episode"] == e).sum()) for e in episodes]])
    print(t)

    t = PrettyTable(["Episode", "El Nino days", "In the most expensive regime", "Captured %"])
    for e in episodes:
        days = table[table["episode"] == e]
        t.add_row([e, len(days), int(days["captured"].sum()), f"{100 * days['captured'].mean():.1f}"])
    print(t)

    runs = (table["regime"] != table["regime"].shift()).sum()
    print(f"Switches per year: {(runs - 1) / (n / 365.25):.1f}   Mean run: {n / runs:.1f} days")

    if missed:
        for e in episodes:
            out = table[(table["episode"] == e) & (table["captured"] == 0)]
            print(f"\n{e}: {len(out)} El Nino days outside the most expensive regime")
            for regime, g in out.groupby("regime"):
                print(f"  regime {regime}: " + "; ".join(date_ranges(g["date"])))


def main():
    parser = argparse.ArgumentParser(description="Regimes and El Nino days of one run")
    parser.add_argument("run", nargs="?", help="run folder, its regimes.csv, or a run id inside experiments/")
    parser.add_argument("--missed", action="store_true", help="list the episode days outside the most expensive regime")
    args = parser.parse_args()
    if args.run:
        report(*read_regimes(args.run), args.missed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
