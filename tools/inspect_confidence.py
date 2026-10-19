"""
inspect_confidence.py

Contact sheets for eyeballing the vespagram catalogue by confidence tier. For
each tier, N random trains; each panel is an 8 h keogram window (full latitude)
centred on the sampled train, with every train of that day drawn: crests solid,
troughs dashed, label coloured by tier. The sampled train is boxed.

  python tools/inspect_confidence.py [--n 12] [--seed 0]
Output: results/vespagram/confidence_check/sheet_<tier>.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the repository root
import vespagram as v

ROOT   = Path(__file__).resolve().parent.parent
TRAINS = ROOT / "results/vespagram/trains.parquet"
OUT    = ROOT / "results/vespagram/confidence_check"

# tier -> (label colour, rule, from the thresholds of vespagram.train_confidence)
_C = v.VespaConfig()
TIERS = {"high":   ("#a6d96a", f"score ≥ {_C.confidence_high:.2f}"),
         "medium": ("#fee08b", f"score {_C.confidence_medium:.2f}–{_C.confidence_high:.2f}"),
         "low":    ("#d9d9d9", f"score < {_C.confidence_medium:.2f}")}
HALF_H = 4.0
VMAX   = 0.06
NCOL   = 3


def draw_train(ax, tr, members, lats, cfg, boxed):
    for d in members:
        lo, hi = v._owned_slice(d, lats.min(), lats.max(), cfg)
        la = np.linspace(lo, hi, 8)
        dd = (la - d["lat_c"]) * v.M_PER_DEG
        for tc, ls in [(c, "-") for c in d["crests_s"]] + [(c, "--") for c in d["troughs_s"]]:
            ax.plot((tc - dd * d["slowness"]) / 3600, la, color="k", lw=1.1 if boxed else 0.8,
                    ls=ls, alpha=1 if boxed else 0.7)
    if boxed:
        ax.add_patch(Rectangle((tr["t_start_s"] / 3600, lats.min()),
                               (tr["t_end_s"] - tr["t_start_s"]) / 3600, lats.max() - lats.min(),
                               fill=False, ec="k", lw=1.6))
    tm = 0.5 * (tr["t_start_s"] + tr["t_end_s"]) / 3600
    ax.text(tm, lats.max() - 0.3,
            f"#{tr['train_id']} {tr['speed_m_s']:.0f} m/s · {tr['period_s'] / 60:.0f} min · "
            f"score {tr['confidence_score']:.2f}",
            fontsize=7, ha="center", va="top", clip_on=True,
            bbox=dict(boxstyle="round,pad=0.2", fc=TIERS[tr["confidence"]][0], ec="k" if boxed else "none",
                      lw=0.6, alpha=0.95))


def panel(ax, day, train_id):
    cfg = v.VespaConfig(input_date=day)
    X, lats, t, results, dets, trains, members = v.detect(cfg)
    k = next(i for i, tr in enumerate(trains) if tr["train_id"] == train_id)
    tc = 0.5 * (trains[k]["t_start_s"] + trains[k]["t_end_s"]) / 3600
    t0 = min(max(tc - HALF_H, t[0] / 3600), t[-1] / 3600 - 2 * HALF_H)
    ax.imshow(X, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, aspect="auto", interpolation="nearest",
              extent=[t[0] / 3600, t[-1] / 3600, lats.min(), lats.max()])
    for i, (tr, idx) in enumerate(zip(trains, members)):
        draw_train(ax, tr, [dets[j] for j in idx], lats, cfg, boxed=i == k)
    ax.set_xlim(t0, t0 + 2 * HALF_H)
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xticks(np.arange(np.ceil(t0), t0 + 2 * HALF_H + 0.01, 2))
    ax.tick_params(labelsize=7)
    tr = trains[k]
    ax.set_title(f"{day}  #{train_id}  semblance {tr['semblance']:.2f}, coverage {tr['coverage']:.2f}, "
                 f"{tr['n_detections']} detections", fontsize=8, loc="left")


def sheet(tier, rows, n_total, path):
    nrow = int(np.ceil(len(rows) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(6 * NCOL, 3.1 * nrow), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        panel(ax, r["date"], r["train_id"])
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("lat [°N]", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time [h UT]", fontsize=8)
    color, rule = TIERS[tier]
    fig.suptitle(f"Confidence: {tier.upper()}  ({rule})  —  {len(rows)} random of {n_total} trains.  "
                 f"Boxed = sampled train; label colour = tier of each train", fontsize=11, y=0.995)
    fig.legend(handles=[Patch(fc=c, ec="k", lw=0.5, label=f"{k}: {d}") for k, (c, d) in TIERS.items()],
               loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  {tier:6s} -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    tr = pl.read_parquet(TRAINS)
    for tier in TIERS:
        sub = tr.filter(pl.col("confidence") == tier)
        rows = sub.sample(min(args.n, sub.height), seed=args.seed).sort("date", "train_id")
        sheet(tier, list(rows.iter_rows(named=True)), sub.height, OUT / f"sheet_{tier}.png")


if __name__ == "__main__":
    main()
