"""
quiet_check.py

Contact sheets of QUIET-TIME vespagram trains (high + medium confidence, Hp30
3 h trailing mean < HP30_MAX at the train midpoint), to see what drives the
large vespagram/ROI ratio at low activity:
  sheet_vespa_only.png   trains with no overlapping ROI train (the extra ones)
  sheet_both.png         trains the ROI method also found (reference)
Each panel: 8 h keogram window centred on the sampled train (boxed), all
vespagram trains of the day, and ROI fronts as orange dashed boxes.

  python quiet_check.py [--n 12] [--seed 0]
Output: results/vespagram/quiet_check/
"""

import sys
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

import inspect_confidence as ic
from paths import FEATURES

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "analysis"))
from catalogue_records import roi_trains   # ROI fronts merged into trains (compare.py rule)
import pickle

OUT      = ROOT / "results/vespagram/quiet_check"
HP30_MAX = 2.0
TOL      = dt.timedelta(minutes=15)       # same matching tolerance as compare.py
ROI_COLOR = "#e07b00"


def hp30_at(times):
    f = (pl.read_parquet(FEATURES, columns=["time", "hp30"]).sort("time")
         .with_columns(pl.col("hp30").rolling_mean(window_size=180, min_samples=1).alias("hp3h")))
    tz = f["time"].dtype
    q = pl.DataFrame({"time": times, "k": np.arange(len(times))}) \
          .with_columns(pl.col("time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").cast(tz)).sort("time")
    return q.join_asof(f.select("time", "hp3h"), on="time", strategy="nearest").sort("k")["hp3h"].to_numpy()


def classify(tr, roi):
    starts = np.array([r["start_time"] for r in roi], dtype="datetime64[us]")
    ends   = np.array([r["end_time"] for r in roi], dtype="datetime64[us]")
    tol = np.timedelta64(int(TOL.total_seconds()), "s")
    out = []
    for s, e in zip(tr["t_start"].to_numpy(), tr["t_end"].to_numpy()):
        out.append(bool(np.any((starts < e + tol) & (ends > s - tol))))
    return out


def roi_boxes(ax, day, fronts):
    mid = dt.datetime.combine(day, dt.time())
    for r in fronts:
        a = (r["start_time"] - mid).total_seconds() / 3600
        b = (r["end_time"] - mid).total_seconds() / 3600
        if b < -3 or a > 27:
            continue
        lo, hi = sorted((r["start_lat"], r["end_lat"]))
        ax.add_patch(Rectangle((a, lo), b - a, hi - lo, fill=False, ec=ROI_COLOR, lw=1.4, ls="--"))


def sheet(rows, title, path, cache, fronts_by_day, ncol=3):
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.2 * nrow), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        ic.panel(ax, r["date"], r["train_id"], cache)
        for dd in (-1, 0, 1):
            roi_boxes(ax, r["date"], fronts_by_day.get(r["date"] + dt.timedelta(days=dd), []))
        ax.set_title(ax.get_title(loc="left") + f"  ·  Hp30 {r['hp3h']:.1f}", fontsize=8, loc="left")
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("lat [°N]", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time [h UT]", fontsize=8)
    fig.suptitle(title, fontsize=11, y=0.995)
    handles = [Patch(fc=c, ec="k", lw=0.5, label=f"{k}: {d}") for k, (c, d) in ic.TIERS.items()]
    handles.append(Line2D([], [], color=ROI_COLOR, ls="--", lw=1.4, label="ROI front"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--more", type=int, default=0,
                    help="instead: K extra vespagram-only sheets (sheet_vespa_only_2.. ), "
                         "none repeating the trains of sheet_vespa_only.png")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    tr = pl.read_parquet(ic.TRAINS).filter(pl.col("confidence") != "low")
    mid = (tr["t_start"] + (tr["t_end"] - tr["t_start"]) / 2).to_list()
    tr = tr.with_columns(pl.Series("hp3h", hp30_at(mid)))
    quiet = tr.filter(pl.col("hp3h") < HP30_MAX)
    roi = roi_trains()
    quiet = quiet.with_columns(pl.Series("roi_match", classify(quiet, roi)))
    print(f"quiet high+medium trains (Hp30 3 h < {HP30_MAX:g}): {quiet.height} of {tr.height}; "
          f"ROI also found {quiet['roi_match'].sum()}, vespagram only "
          f"{(~quiet['roi_match']).sum()}")
    print(quiet.group_by("roi_match").agg(pl.len(), pl.col("amplitude").median().alias("amp_med"),
                                          pl.col("semblance").median().alias("sem_med"),
                                          (pl.col("confidence") == "high").mean().alias("share_high"),
                                          pl.col("speed_m_s").median().alias("speed_med"),
                                          (pl.col("speed_m_s") < 350).mean().alias("share_lt350"))
          .sort("roi_match"))
    act = tr.filter(pl.col("hp3h") >= HP30_MAX)
    print(f"active-time trains (Hp30 >= {HP30_MAX:g}): median speed {act['speed_m_s'].median():.0f} m/s, "
          f"share < 350 m/s {(act['speed_m_s'] < 350).mean():.2f}")

    fronts = pickle.load(open(ROOT / "results/roi/lstid_rois.pkl", "rb"))
    by_day = {}
    for r in fronts:
        by_day.setdefault(r["date"], []).append(r)

    cache = {}
    if args.more:
        # one permutation: its first n trains are the original sheet, the next ones the extras
        sub = quiet.filter(~pl.col("roi_match"))
        shown = set(sub.sample(min(args.n, sub.height), seed=args.seed)
                       .select("date", "train_id").iter_rows())
        rest = sub.filter(~pl.struct("date", "train_id").map_elements(
            lambda s: (s["date"], s["train_id"]) in shown, return_dtype=pl.Boolean))
        rest = rest.sample(fraction=1.0, shuffle=True, seed=args.seed + 1)
        for k in range(args.more):
            rows = rest[k * args.n:(k + 1) * args.n].sort("date", "train_id")
            if rows.height == 0:
                break
            sheet(list(rows.iter_rows(named=True)),
                  f"Quiet time (Hp30 3 h < {HP30_MAX:g}), high+medium trains found by the vespagram "
                  f"only  —  extra set {k + 1}: {rows.height} more of {sub.height}.  Boxed = sampled train",
                  OUT / f"sheet_vespa_only_{k + 2}.png", cache, by_day)
        return

    for match, name, label in [(False, "vespa_only", "found by the vespagram only"),
                               (True, "both", "also found by ROI (reference)")]:
        sub = quiet.filter(pl.col("roi_match") == match)
        rows = sub.sample(min(args.n, sub.height), seed=args.seed).sort("date", "train_id")
        sheet(list(rows.iter_rows(named=True)),
              f"Quiet time (Hp30 3 h < {HP30_MAX:g}), high+medium trains {label}  —  "
              f"{rows.height} random of {sub.height}.  Boxed = sampled train",
              OUT / f"sheet_{name}.png", cache, by_day)


if __name__ == "__main__":
    main()
