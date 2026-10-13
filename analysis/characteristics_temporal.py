"""
characteristics_temporal.py

LSTID wave characteristics (propagation speed, peak amplitude) as a function of
time -- binned by year and by UT hour. One violin plot per (parameter, grouping)
combination, reusing the violin style from driver_characteristics.py (red mean
dot, black median square, value labels, per-bin n counts).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driver_characteristics import load_records, draw_violins

MIN_PER_BIN = 5   # bins with fewer ROIs than this are dropped (too thin to plot)

# param key -> (label, unit, scale, take_abs)
PARAMETERS = {
    "speed_m_s":      ("Propagation speed", "m/s",  1, False),
    "peak_amplitude": ("Peak amplitude (abs)", "TECu", 1, True),
}

# grouping name -> (per-record key function, axis label)
GROUPINGS = {
    "year": (lambda r: r["ref_time"].year, "Year"),
    "hour": (lambda r: r["ref_time"].hour, "Hour (UT)"),
}


def plot_temporal_characteristic(records, param_key, group_key, out_dir):
    label, unit, scale, take_abs = PARAMETERS[param_key]
    keyfn, xlabel = GROUPINGS[group_key]

    vals = np.array([r[param_key] for r in records], dtype=float) * scale
    if take_abs:
        vals = np.abs(vals)
    keys = np.array([keyfn(r) for r in records])

    valid = ~np.isnan(vals)
    vals, keys = vals[valid], keys[valid]

    groups, tick_labels = [], []
    for k in sorted(np.unique(keys)):
        g = vals[keys == k]
        if len(g) >= MIN_PER_BIN:
            groups.append(g)
            tick_labels.append(str(int(k)))

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(groups)), 6))
    draw_violins(ax, groups, tick_labels, rotate=45 if group_key == "year" else 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{label} [{unit}]")
    ax.set_title(f"LSTID {label.lower()} by {xlabel.lower()}  (n={len(vals)})")

    fig.tight_layout()
    fig.savefig(out_dir / f"characteristics_{group_key}_{param_key}.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LSTID speed/amplitude violin plots by year and by hour.")
    parser.add_argument("--pkl", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "results/vespagram/lstid_trains_highmed.pkl",
                        help="LSTID records pickle (default: vespagram high+medium trains, written by catalogue_records.py; ROI: results/roi/lstid_trains.pkl, or the raw "
                             "per-front results/roi/lstid_rois.pkl)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder for the plots (default: a climatology/ folder "
                             "beside --pkl, so results stay with the run that produced them)")
    return parser.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.pkl.parent / "climatology"
    records = load_records(args.pkl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for group_key in GROUPINGS:
        for param_key in PARAMETERS:
            plot_temporal_characteristic(records, param_key, group_key, args.out_dir)
            print(f"  {group_key} / {param_key}: saved -> "
                 f"{args.out_dir / f'characteristics_{group_key}_{param_key}.png'}")

    print(f"Saved temporal-characteristics plots -> {args.out_dir}  (n={len(records)} LSTID ROIs)")


if __name__ == "__main__":
    main()
