"""
Real spot price of El Nino days compared with the rest of the days.

A day is El Nino when the ONI of its month is >= 0.5 (the same rule used for the regimes). Each row of the output
is one group of El Nino days:
- the two episodes used in the thesis: El Nino days in 2015-2016 and in 2023-2024
- every El Nino episode in the data, as a run of consecutive El Nino months
- all El Nino days together

For each group: days, first and last day, median, mean, standard deviation, 10th and 90th percentiles of the daily
real price. The reference is every day that is not El Nino (whole series): its median, mean and standard deviation
are repeated in each row. The pct_ columns give the share of the group's days above each reference value.

Inputs:
    data/processed/XM/precio_bolsa_diario_real.csv
    data/processed/NOAA/oni.csv

Output:
    data/processed/NOAA/el_nino_price_stats.csv

Usage:
    python scripts/05_el_nino_price_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
proc = root / "data" / "processed"
thesis_episodes = {"2015-2016": [2015, 2016], "2023-2024": [2023, 2024]}


def load_days():
    price = pd.read_csv(proc / "XM" / "precio_bolsa_diario_real.csv", parse_dates=["date"])
    oni = pd.read_csv(proc / "NOAA" / "oni.csv", parse_dates=["date"])
    days = price[["date", "precio_real_cop_kwh"]].rename(columns={"precio_real_cop_kwh": "price"})
    days["month"] = days["date"].dt.to_period("M")
    oni["month"] = oni["date"].dt.to_period("M")
    days = days.merge(oni[["month", "oni"]], on="month", how="left")
    days["nino"] = days["oni"] >= 0.5
    return days


def nino_groups(days):
    """{name: (type, mask)} for the thesis episodes, every run of consecutive El Nino months and all El Nino days."""
    groups = {}
    for name, years in thesis_episodes.items():
        groups[name] = ("thesis episode", days["nino"] & days["date"].dt.year.isin(years))
    months = days.drop_duplicates("month")[["month", "nino"]].reset_index(drop=True)
    run = (months["nino"] != months["nino"].shift()).cumsum()
    for _, block in months[months["nino"]].groupby(run[months["nino"]]):
        first, last = block["month"].iloc[0], block["month"].iloc[-1]
        groups[f"{first} to {last}"] = ("consecutive El Nino months", days["month"].between(first, last))
    groups["all El Nino days"] = ("all", days["nino"])
    return groups


def main():
    days = load_days()
    rest = days.loc[~days["nino"], "price"]
    ref = {"median": rest.median(), "mean": rest.mean(), "std": rest.std()}
    all_median = days["price"].median()

    rows = []
    for name, (kind, mask) in nino_groups(days).items():
        p = days.loc[mask, "price"]
        rows.append({
            "group": name,
            "type": kind,
            "first_day": days.loc[mask, "date"].min().date(),
            "last_day": days.loc[mask, "date"].max().date(),
            "days": len(p),
            "median_price": p.median(),
            "mean_price": p.mean(),
            "std_price": p.std(),
            "p10_price": p.quantile(0.1),
            "p90_price": p.quantile(0.9),
            "rest_days": len(rest),
            "rest_median_price": ref["median"],
            "rest_mean_price": ref["mean"],
            "rest_std_price": ref["std"],
            "median_ratio_to_rest": p.median() / ref["median"],
            "pct_above_rest_median": 100 * (p > ref["median"]).mean(),
            "pct_above_rest_mean": 100 * (p > ref["mean"]).mean(),
            "pct_above_rest_mean_plus_1std": 100 * (p > ref["mean"] + ref["std"]).mean(),
            "pct_above_all_days_median": 100 * (p > all_median).mean(),
        })

    out = pd.DataFrame(rows)
    out[out.select_dtypes("number").columns] = out.select_dtypes("number").round(2)
    path = proc / "NOAA" / "el_nino_price_stats.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path.relative_to(root)}: {len(out)} rows")
    print(f"Rest of the days: {len(rest)}, median {ref['median']:.1f}, mean {ref['mean']:.1f}, std {ref['std']:.1f}")
    print(out[["group", "days", "median_price", "mean_price", "pct_above_rest_mean",
               "pct_above_rest_mean_plus_1std"]].to_string(index=False))


if __name__ == "__main__":
    main()
