"""
storm_overview.py

Overview sheets of the vespagram result on selected days: full-day keogram with
the LSTID trains (label colour = confidence tier, colour scale fitted to the
day) and the SYM-H / SME traces underneath. Each panel title also gives how many
hours of the day the trains cover, to compare with what the eye sees.

  python storm_overview.py                    storm days: the 2 lowest daily SYM-H
                                              minima per year (>= 5 days apart)
  python storm_overview.py --random 40        random days, uniform over ALL keogram
                                              days (with or without trains)

Output (results/vespagram/storms/ or results/vespagram/random_days/):
  days.csv                selected days (SYM-H min, SME max, Hp30 max, LSTID hours)
  overview_N.png          the sheets
  daily/<date>_vespa.png  the standard vespagram figure for each day
"""

import argparse
from dataclasses import replace

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import vespagram as v
from detect_roi import discover_dates
from inspect_confidence import draw_train, TIERS, ROOT
from paths import FEATURES

VMIN_SCALE = 0.06    # colour scale is max(this, 98th percentile of |X| in the day) [TECu]


def daily_stats(features):
    return (features.group_by(pl.col("time").dt.date().alias("date"))
            .agg(pl.col("sym_h").min().alias("symh_min"),
                 pl.col("sme").max().alias("sme_max"),
                 pl.col("hp30").max().alias("hp30_max")))


def select_storms(stats, per_year=2, min_gap_days=5):
    stats = stats.sort("symh_min")
    out = []
    for y in sorted(stats["date"].dt.year().unique()):
        picked = []
        for r in stats.filter(pl.col("date").dt.year() == y).iter_rows(named=True):
            if all(abs((r["date"] - p["date"]).days) >= min_gap_days for p in picked):
                picked.append(r)
            if len(picked) == per_year:
                break
        out += picked
    return pl.DataFrame(out).sort("date")


def random_days(stats, n, seed):
    days = discover_dates(v.VespaConfig().keo_folder, None)
    pick = sorted(np.random.default_rng(seed).choice(len(days), size=n, replace=False))
    sel = pl.DataFrame({"date": [days[i] for i in pick]})
    return sel.join(stats, on="date", how="left").sort("date")


def covered_hours(trains):
    """Hours of [0, 24) UT covered by at least one of the trains (union)."""
    ivs = sorted((max(tr["t_start_s"], 0), min(tr["t_end_s"], 86400)) for tr in trains)
    total, end = 0.0, 0.0
    for a, b in ivs:
        a = max(a, end)
        if b > a:
            total += b - a
            end = b
    return total / 3600


def day_panel(fig, spec, day, info, features, out_daily):
    cfg = v.VespaConfig(input_date=day)
    X, lats, t, results, dets, trains, members = v.detect(cfg)
    inday = (t >= 0) & (t < 86400)
    vmax = max(VMIN_SCALE, float(np.nanpercentile(np.abs(X[:, inday]), 98)))

    gs = spec.subgridspec(3, 1, height_ratios=[3.2, 0.8, 0.8], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.imshow(X, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest",
              extent=[t[0] / 3600, t[-1] / 3600, lats.min(), lats.max()])
    for tr, idx in zip(trains, members):
        draw_train(ax, tr, [dets[j] for j in idx], lats, cfg, boxed=False)
    ax.set_xlim(0, 24)
    ax.set_ylim(lats.min(), lats.max())
    ax.set_ylabel("lat [°N]", fontsize=8)
    ax.tick_params(labelsize=7, labelbottom=False)
    n  = {k: sum(tr["confidence"] == k for tr in trains) for k in TIERS}
    h_all = covered_hours(trains)
    h_hm  = covered_hours([tr for tr in trains if tr["confidence"] != "low"])
    fmt = lambda x, f: "–" if x is None else format(x, f)       # driver data can be missing
    ax.set_title(f"{day}   SYM-H min {fmt(info['symh_min'], '.0f')} nT · "
                 f"SME max {fmt(info['sme_max'], '.0f')} nT · Hp30 max {fmt(info['hp30_max'], '.1f')}\n"
                 f"trains: " + ", ".join(f"{n[k]} {k}" for k in TIERS) +
                 f"   |   LSTID present {h_all:.1f} h ({h_all / 24:.0%}), high+medium {h_hm:.1f} h"
                 f"   (colour ±{vmax:.2f} TECu)", fontsize=8, loc="left")

    f = features.filter(pl.col("time").dt.date() == day)
    h = (f["time"].dt.hour() + f["time"].dt.minute() / 60).to_numpy()
    for k, (col, label, color) in enumerate([("sym_h", "SYM-H [nT]", "#5b3f8c"),
                                             ("sme", "SME [nT]", "#b5532a")]):
        a = fig.add_subplot(gs[1 + k], sharex=ax)
        a.plot(h, f[col].to_numpy(), color=color, lw=1)
        a.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
        a.tick_params(labelsize=7, labelbottom=k == 1)
        a.grid(lw=0.4, alpha=0.4)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if col == "sym_h":
            a.axhline(0, color="#999999", lw=0.6)
    a.set_xticks(range(0, 25, 3))
    a.set_xlabel("time [h UT]", fontsize=8)

    # standard full figure for the day
    v.write_day(replace(cfg, out_dir=out_daily), X, lats, t, results, dets, trains, members)
    return h_all, h_hm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-year", type=int, default=2)
    ap.add_argument("--random", type=int, default=0, help="N random days instead of storms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-sheet", type=int, default=6)
    args = ap.parse_args(argv)

    features = pl.read_parquet(FEATURES, columns=["time", "sym_h", "sme", "hp30"])
    stats = daily_stats(features)
    if args.random:
        out, days = ROOT / "results/vespagram/random_days", random_days(stats, args.random, args.seed)
        what = f"{args.random} random days (seed {args.seed}, uniform over all keogram days)"
    else:
        out, days = ROOT / "results/vespagram/storms", select_storms(stats, args.per_year)
        what = f"storm days ({args.per_year} strongest per year by SYM-H)"
    out.mkdir(parents=True, exist_ok=True)

    rows, hours = list(days.iter_rows(named=True)), []
    ncol = 2
    for s, start in enumerate(range(0, len(rows), args.per_sheet), 1):
        chunk = rows[start:start + args.per_sheet]
        nrow = int(np.ceil(len(chunk) / ncol))
        fig = plt.figure(figsize=(9 * ncol, 5.4 * nrow))
        outer = fig.add_gridspec(nrow, ncol, hspace=0.38, wspace=0.12)
        for i, info in enumerate(chunk):
            hours.append(day_panel(fig, outer[i // ncol, i % ncol], info["date"], info, features,
                                   out / "daily"))
        fig.suptitle(f"Vespagram LSTID trains on {what}  —  sheet {s}", fontsize=12, y=0.995)
        fig.legend(handles=[Patch(fc=c, ec="k", lw=0.5, label=f"{k}: {d}") for k, (c, d) in TIERS.items()],
                   loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.98))
        fig.savefig(out / f"overview_{s}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"sheet {s} -> {out / f'overview_{s}.png'}")

    h = np.array(hours)
    days.with_columns(pl.Series("lstid_hours", h[:, 0]), pl.Series("lstid_hours_highmed", h[:, 1])) \
        .write_csv(out / "days.csv")
    print(f"LSTID present: {h[:, 0].mean():.1f} h/day ({h[:, 0].mean() / 24:.0%}) all tiers, "
          f"{h[:, 1].mean():.1f} h/day ({h[:, 1].mean() / 24:.0%}) high+medium; "
          f"days with no train: {(h[:, 0] == 0).sum()}/{len(h)}")


if __name__ == "__main__":
    main()
