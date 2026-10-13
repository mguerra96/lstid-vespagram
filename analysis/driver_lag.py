"""
driver_lag.py

How long after a geomagnetic driver changes does LSTID occurrence respond?

For a representative subset of drivers (one from each physical layer), slide a
fixed short trailing-mean window back in time (lag 0..LAG_MAX h) and, at each
lag, fit the same confound-adjusted logistic model used in driver_regression.py
(driver + season + local-time + F10.7). The predictive skill (AUC) as a
function of lag reveals the delay between driver forcing and ionospheric
response -- and whether the upstream solar-wind drivers lead the downstream
magnetospheric-response indices.

Output (in --out-dir):
  - driver_lag_auc.png : AUC vs lag, one curve per driver, peak lag marked.
  - driver_lag.csv     : AUC and standardized |coef| per (driver, lag).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driver_activity import load_records
from driver_regression import (DRIVER_COLS, build_base_design,
                               load_driver_on_grid, _zscore)

# representative subset: one driver per physical layer
SUBSET      = ["Hp30", "SME", "IE", "E_KL", "Newell", "Bz", "SymH", "Pressure"]
WINDOW_H    = 1     # short trailing-mean window while we vary the lag
LAG_MAX_H   = 12    # sweep lag 0..LAG_MAX_H hours
LAG_STEP_H  = 1

CONTROLS = ["sin_m", "cos_m", "sin_2m", "cos_2m",
            "sin_h", "cos_h", "sin_2h", "cos_2h", "f107"]


def sweep_lags(records):
    base, grid, years = build_base_design(records)
    lags = list(range(0, LAG_MAX_H + 1, LAG_STEP_H))

    rows = []
    for name in SUBSET:
        for lag in lags:
            col = load_driver_on_grid(name, grid, years,
                                      window_hours=WINDOW_H, lag_hours=lag)
            d = base[["y"] + CONTROLS].copy()
            d[name] = col
            d = d.dropna()

            X   = sm.add_constant(d[[name] + CONTROLS])
            res = sm.Logit(d["y"], X).fit(disp=False)
            auc = roc_auc_score(d["y"], res.predict(X))
            rows.append({"driver": name, "lag_h": lag,
                         "coef_abs": abs(res.params[name]), "auc": auc})
    return pd.DataFrame(rows)


def plot_lag_curves(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    peaks = []
    for i, name in enumerate(SUBSET):
        d = results[results["driver"] == name].sort_values("lag_h")
        color = cmap(i % 10)
        ax.plot(d["lag_h"], d["auc"], marker="o", markersize=4,
                color=color, label=name)
        # mark the peak-AUC lag
        best = d.loc[d["auc"].idxmax()]
        ax.scatter([best["lag_h"]], [best["auc"]], s=90, facecolors="none",
                   edgecolors=color, linewidths=2, zorder=5)
        peaks.append((name, int(best["lag_h"]), best["auc"]))

    ax.set_xlabel(f"Lag [h]  (driver averaged over a {WINDOW_H}h window ending lag hours "
                  f"before the LSTID)")
    ax.set_ylabel("Predictive skill (model AUC)")
    ax.set_title("LSTID response delay per driver  "
                 "(circle = peak-skill lag; adjusted for season / local time / F10.7)")
    ax.set_xticks(range(0, LAG_MAX_H + 1, LAG_STEP_H))
    ax.grid(linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "driver_lag_auc.png", dpi=150)
    plt.close(fig)
    return peaks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LSTID response-delay (lag) study for geomagnetic drivers.")
    parser.add_argument("--pkl", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "results/vespagram/lstid_hours_highmed.pkl",
                        help="LSTID records pickle (default: vespagram high+medium hours, written by catalogue_records.py; ROI: results/roi/lstid_hours.pkl, or the raw "
                             "per-front results/roi/lstid_rois.pkl)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder (default: a climatology/ folder beside --pkl, "
                             "so results stay with the run that produced them)")
    return parser.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.pkl.parent / "climatology"
    records = load_records(args.pkl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = sweep_lags(records)
    results.to_csv(args.out_dir / "driver_lag.csv", index=False)
    peaks = plot_lag_curves(results, args.out_dir)

    print(f"Peak-skill lag per driver (window={WINDOW_H}h):")
    for name, lag, auc in peaks:
        print(f"  {name:10s} lag={lag:2d}h  AUC={auc:.4f}")
    print(f"\nSaved -> {args.out_dir / 'driver_lag_auc.png'}")
    print(f"Saved -> {args.out_dir / 'driver_lag.csv'}")


if __name__ == "__main__":
    main()
