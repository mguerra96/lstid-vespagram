"""
driver_regression.py

Rank geomagnetic/auroral drivers by how well each predicts LSTID occurrence,
controlling for the confounds that would otherwise steal credit: season
(equinoctial geomagnetic activity), local time (diurnal cycle), and the solar
cycle (F10.7).

Approach
--------
Unit of analysis: one row per hour of the whole 2015-2025 span (the same
regular grid used by driver_activity.py). Outcome y = 1 if >= 1 LSTID ROI was
detected that hour, else 0.

The electrojet indices (SME/SML/IE/IU/IL/SMU) are near-duplicates of one
another, so a single model containing all of them is uninterpretable
(multicollinearity). Instead we fit ONE logistic regression per driver:

    logit P(LSTID) ~ z(driver_3h_mean)
                     + sin/cos(month) + sin/cos(2*month)   # season: annual +
                                                           #   semiannual (equinoctial)
                     + sin/cos(hour)  + sin/cos(2*hour)    # local time: diurnal +
                                                           #   semidiurnal (UT)
                     + z(F10.7_daily)                      # solar cycle

All continuous predictors are z-scored, so the driver's coefficient is the
change in log-odds per 1 standard deviation of that driver AFTER removing the
season / local-time / solar-cycle signal -- directly comparable across drivers.

Outputs (in --out-dir):
  - driver_regression_ranking.png : drivers ranked by standardized coefficient
    with 95% CIs (adjusted effect) alongside pseudo-R2 / AUC.
  - driver_regression.csv         : coef, 95% CI, p-value, pseudo-R2, AUC per driver.
"""

import sys
import pickle
import argparse
import functools
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supermag import read_features

ROLLING_HOURS = 3    # trailing window for the driver average
GRID_MINUTES  = 60   # cadence of the regular time grid LSTID ref_times are snapped to
                     # (MATLAB reference used 30 min; occurrence and exposure are
                     # evaluated on this same grid so their counts are directly
                     # comparable, avoiding the ratio blow-ups a misaligned grid causes)

# All drivers, in every analysis script, are read from the single processed,
# quality-controlled feature table (uniform 1-min cadence, 2015-2025, outliers
# cleaned). The raw per-provider parquets are not used: their IMAGE files in
# particular contain spikes (IE up to ~13000 nT vs the QC'd ~3400 nT) that
# corrupt IE's z-score and its activity bins.
F107_COL = "f107"

# regression driver name -> (feature-table column, transform). `transform` is
# applied to the raw column before the trailing mean; None = identity.
#   abs: |clock angle| in [0, pi] -- 0 = northward IMF (quiet), pi = fully
#     southward (geoeffective). Averaging/z-scoring the raw wrapped angle would
#     be meaningless, so we rectify to its magnitude first.
# HWM14 meridional wind is deliberately excluded: it is a deterministic model
# climatology ~79% determined by hour-of-day, hence collinear with the diurnal
# controls; its standardized coefficient would be inflated and incomparable.
DRIVER_COLS = {
    "Hp30":     ("hp30",            None),
    "SymH":     ("sym_h",           None),
    "IU":       ("image_iu",        None),
    "IL":       ("image_il",        None),
    "IE":       ("image_ie",        None),
    "SMU":      ("smu",             None),
    "SML":      ("sml",             None),
    "SME":      ("sme",             None),
    "Pressure": ("sw_pressure",     None),
    "E_KL":     ("e_kl",            None),
    "ClockAng": ("imf_clock_angle", "abs"),
    "Newell":   ("newell_dphi_dt",  None),
    "Bz":       ("imf_bz_gsm",      None),
}

def load_records(pkl_path):
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    if not records:
        raise ValueError(f"No LSTID records found in {pkl_path}")
    return records


@functools.lru_cache(maxsize=None)
def _feature(col):
    """One column of the feature table with its time, sorted by time (read once)."""
    return read_features(["time", col]).sort("time")


def _load_driver_series(name):
    """One driver's timeseries from the feature table, transform applied,
    as a sorted polars frame ['time', <name>]."""
    col, transform = DRIVER_COLS[name]
    df = _feature(col)
    if transform == "abs":
        df = df.with_columns(pl.col(col).abs().alias(col))
    return df.rename({col: name})


def _round_to_grid(times, minutes=GRID_MINUTES):
    """Round datetimes to the nearest `minutes`, matching the MATLAB
    reference's roundedtime (seconds zeroed, minutes rounded to nearest 30)."""
    ts = pl.Series(times).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    epoch_min = ts.dt.epoch(time_unit="us") / 1_000_000 / 60
    rounded_min = (epoch_min / minutes).round(0) * minutes
    return (rounded_min * 60 * 1_000_000).cast(pl.Int64).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")


def _rolling_on_grid(series, col, grid, window_hours=ROLLING_HOURS):
    """`window_hours`-trailing rolling mean of series[col], sampled (nearest)
    at each timestamp of `grid` (in any order; times without a time zone are
    UTC, as in the catalogues). Returns a numpy array aligned with `grid`."""
    step     = series["time"][1] - series["time"][0]
    window_n = max(1, int(round(dt.timedelta(hours=window_hours) / step)))
    rolled   = series.with_columns(
        pl.col(col).rolling_mean(window_size=window_n, min_samples=1).alias("_roll")
    )
    t = pl.Series("time", grid)
    if t.dtype.time_zone is None:
        t = t.dt.replace_time_zone("UTC")
    grid_df = (pl.DataFrame({"time": t.cast(rolled["time"].dtype)})
                 .with_row_index("_i").sort("time"))
    joined = grid_df.join_asof(rolled.select(["time", "_roll"]), on="time",
                              strategy="nearest")
    return joined.sort("_i")["_roll"].to_numpy()


def _f107_series():
    """F10.7 solar flux from the feature table, as a sorted frame ['time','f107']."""
    return _feature(F107_COL)


def build_base_design(records):
    """
    Build the per-hour outcome + confound-control table (everything EXCEPT the
    driver columns), plus the grid so callers can add driver columns
    themselves (e.g. at different lags).

    Returns (df, grid) where df has one row per grid hour with columns:
      time          - grid timestamp (UTC)
      y             - 1 if >=1 LSTID detected that hour, else 0
      sin_m,cos_m   - cyclic season, annual harmonic (month)
      sin_2m,cos_2m - cyclic season, semiannual harmonic (equinoctial double-peak)
      sin_h,cos_h   - cyclic local time, diurnal harmonic (UT hour)
      sin_2h,cos_2h - cyclic local time, semidiurnal harmonic
      f107          - z-scored daily F10.7
    """
    # regular hourly grid spanning the rounded ROI times (same as driver_activity)
    rounded = _round_to_grid([r["ref_time"] for r in records])
    t_min, t_max = rounded.min(), rounded.max()
    grid = pl.datetime_range(t_min, t_max, interval=f"{GRID_MINUTES}m",
                             time_zone="UTC", eager=True)
    grid_us = grid.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")

    # outcome: which grid slots had >=1 LSTID
    detected = set(rounded)
    y = np.array([1 if t in detected else 0 for t in grid_us], dtype=int)

    df = pd.DataFrame({"time": grid_us.to_numpy(), "y": y})
    months = grid_us.dt.month().to_numpy()
    hours  = grid_us.dt.hour().to_numpy()
    # annual + semiannual (equinoctial) season harmonics
    df["sin_m"]  = np.sin(2 * np.pi * months / 12)
    df["cos_m"]  = np.cos(2 * np.pi * months / 12)
    df["sin_2m"] = np.sin(2 * np.pi * 2 * months / 12)
    df["cos_2m"] = np.cos(2 * np.pi * 2 * months / 12)
    # diurnal + semidiurnal local-time harmonics
    df["sin_h"]  = np.sin(2 * np.pi * hours / 24)
    df["cos_h"]  = np.cos(2 * np.pi * hours / 24)
    df["sin_2h"] = np.sin(2 * np.pi * 2 * hours / 24)
    df["cos_2h"] = np.cos(2 * np.pi * 2 * hours / 24)

    # F10.7 on the grid, z-scored
    f107 = _rolling_on_grid(_f107_series(), F107_COL, grid, window_hours=1)
    df["f107"] = _zscore(f107)

    return df, grid


def load_driver_on_grid(name, grid, window_hours=ROLLING_HOURS, lag_hours=0):
    """
    z-scored `window_hours`-trailing mean of driver `name`, sampled on `grid`,
    optionally lagged by `lag_hours` (the grid slot at time t receives the
    driver value current at t - lag_hours).
    """
    series = _load_driver_series(name)
    # a lag means: sample the driver at (grid time - lag). Shift the grid back.
    q_grid = grid if lag_hours == 0 else grid.dt.offset_by(f"-{int(lag_hours * 60)}m")
    vals = _rolling_on_grid(series, name, q_grid, window_hours=window_hours)
    return _zscore(vals)


def build_design(records):
    """Full design table: base (outcome + controls) plus every driver's
    z-scored 3h trailing mean column. See build_base_design for the controls."""
    df, grid = build_base_design(records)
    for name in DRIVER_COLS:
        df[name] = load_driver_on_grid(name, grid)
    return df


def _zscore(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x - mu


def fit_driver_models(df):
    """One logistic regression per driver. Returns a DataFrame of results."""
    controls = ["sin_m", "cos_m", "sin_2m", "cos_2m",
                "sin_h", "cos_h", "sin_2h", "cos_2h", "f107"]
    rows = []
    for name in DRIVER_COLS:
        cols = [name] + controls + ["y"]
        d = df[cols].dropna()
        X = sm.add_constant(d[[name] + controls])
        res = sm.Logit(d["y"], X).fit(disp=False)

        coef   = res.params[name]
        lo, hi = res.conf_int().loc[name]
        pval   = res.pvalues[name]
        # McFadden pseudo-R2 and AUC of the full model
        pseudo_r2 = res.prsquared
        auc = roc_auc_score(d["y"], res.predict(X))

        rows.append({"driver": name, "coef": coef, "ci_lo": lo, "ci_hi": hi,
                     "pvalue": pval, "pseudo_r2": pseudo_r2, "auc": auc,
                     "n": len(d), "n_pos": int(d["y"].sum())})
    out = pd.DataFrame(rows)
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_ranking(results, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})

    y = np.arange(len(results))[::-1]   # strongest at top
    err = np.vstack([results["coef"] - results["ci_lo"],
                     results["ci_hi"] - results["coef"]])
    ax1.errorbar(results["coef"], y, xerr=err, fmt="o", color="tab:blue",
                 capsize=4, markersize=6)
    ax1.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels(results["driver"])
    ax1.set_xlabel("Standardized logistic coefficient\n"
                   "(log-odds per 1 SD, adjusted for season / local time / F10.7)")
    ax1.set_title("Driver effect on LSTID occurrence (95% CI)")
    ax1.grid(axis="x", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)

    ax2.barh(y, results["auc"], color="tab:green", alpha=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(results["driver"])
    ax2.set_xlim(0.5, max(0.75, results["auc"].max() * 1.02))
    ax2.set_xlabel("Model AUC")
    ax2.set_title("Full-model discrimination")
    ax2.grid(axis="x", linewidth=0.5, alpha=0.4)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_dir / "driver_regression_ranking.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rank geomagnetic drivers of LSTID occurrence via logistic regression.")
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

    df      = build_design(records)
    results = fit_driver_models(df)

    csv_path = args.out_dir / "driver_regression.csv"
    results.to_csv(csv_path, index=False)
    plot_ranking(results, args.out_dir)

    print(f"Base rate: {df['y'].mean():.1%} of {len(df)} hours had >=1 LSTID\n")
    print(results.to_string(index=False,
          columns=["driver", "coef", "ci_lo", "ci_hi", "pvalue", "pseudo_r2", "auc"],
          float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {args.out_dir / 'driver_regression_ranking.png'}")


if __name__ == "__main__":
    main()
