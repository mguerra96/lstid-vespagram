"""
driver_characteristics.py

LSTID wave characteristics (period, propagation speed, peak amplitude) as a
function of geomagnetic/auroral activity level (SME, IE), using each ROI's
3h trailing mean of the driver at its ref_time. One violin plot per
(driver, parameter) combination, with activity binned into fixed-width bins.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driver_activity import DRIVERS, _cap_bounds
from driver_regression import ROLLING_HOURS, load_records, _load_driver_series, _rolling_on_grid

BIN_WIDTH     = 200   # nT, fixed-width activity bins for the violin plots
MIN_PER_BIN   = 5     # bins with fewer ROIs than this are dropped (too thin to plot)

PARAMETERS = {
    "period_s":       ("Period", "min", 1 / 60),
    "speed_m_s":      ("Propagation speed", "m/s", 1),
    "peak_amplitude": ("Peak amplitude (abs)", "TECu", 1),
}


def compute_driver_at_rois(records, name):
    """3h trailing mean of driver `name` at each ROI's ref_time (nearest
    sample, no grid snapping -- this is per-ROI, not an occurrence/exposure
    comparison, so there's no need to align to a shared regular grid)."""
    return _rolling_on_grid(_load_driver_series(name), name, [r["ref_time"] for r in records])


def draw_violins(ax, groups, tick_labels, rotate=45, whisker_pct=(5, 95)):
    """Draw violins for `groups` (list of value arrays) at integer positions,
    with a red mean dot and black median square per violin, their values
    labelled to the right, and per-violin n= counts along the bottom. Shared
    by the driver-activity and temporal characteristic plots.

    Whiskers span the `whisker_pct` percentiles (default 5th-95th) rather than
    the data min/max, so a single extreme value doesn't stretch them."""
    positions = np.arange(len(groups))
    # showextrema=False: draw our own percentile whiskers instead of min/max caps
    parts = ax.violinplot(groups, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("tab:blue")
        body.set_alpha(0.6)

    # percentile whiskers: vertical line lo..hi with short end caps
    lo_p, hi_p = whisker_pct
    for x, g in zip(positions, groups):
        lo, hi = np.percentile(g, [lo_p, hi_p])
        ax.plot([x, x], [lo, hi], color="tab:blue", linewidth=1.2, zorder=2)
        for yv in (lo, hi):
            ax.plot([x - 0.05, x + 0.05], [yv, yv], color="tab:blue",
                    linewidth=1.2, zorder=2)

    means   = [np.mean(g) for g in groups]
    medians = [np.median(g) for g in groups]
    ax.scatter(positions, means, marker="o", color="tab:red", zorder=3,
              s=30, label="Mean")
    ax.scatter(positions, medians, marker="s", color="black", zorder=3,
              s=30, label="Median")

    x_offset = 0.12
    all_vals = [v for g in groups for v in g]
    min_gap  = (max(all_vals) - min(all_vals)) * 0.025   # min label separation
    for x, m, md in zip(positions, means, medians):
        if abs(m - md) < min_gap:
            m_y, md_y = m + min_gap / 2, md - min_gap / 2
        else:
            m_y, md_y = m, md
        ax.text(x + x_offset, m_y, f"{m:.3g}", ha="left", va="center",
                fontsize=7, color="tab:red")
        ax.text(x + x_offset, md_y, f"{md:.3g}", ha="left", va="center",
                fontsize=7, color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=rotate)
    for x, g in zip(positions, groups):
        ax.text(x, ax.get_ylim()[0], f"n={len(g)}", ha="center", va="bottom",
                fontsize=7, color="dimgray")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8)


def plot_driver_characteristic(name, driver_values, records, param_key, out_dir):
    _, cap = DRIVERS[name]
    label, unit, scale = PARAMETERS[param_key]

    param_values = np.array([r[param_key] for r in records], dtype=float) * scale
    if param_key == "peak_amplitude":
        param_values = np.abs(param_values)
    valid = ~np.isnan(driver_values) & ~np.isnan(param_values)
    driver_values, param_values = driver_values[valid], param_values[valid]

    low, high = _cap_bounds(cap)
    clipped = np.clip(driver_values, low, high)
    bin_centers = np.round(clipped / BIN_WIDTH) * BIN_WIDTH

    levels = np.unique(bin_centers)
    groups, tick_labels = [], []
    for lv in levels:
        vals = param_values[bin_centers == lv]
        if len(vals) < MIN_PER_BIN:
            continue
        groups.append(vals)
        if high is not None and lv >= high:
            tick_labels.append(f"{high:g}+")
        elif low is not None and lv <= low:
            tick_labels.append(f"{low:g}-")
        else:
            tick_labels.append(f"{lv:g}")

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    draw_violins(ax, groups, tick_labels)
    ax.set_xlabel(f"{name} [nT], {ROLLING_HOURS}h trailing mean, binned to {BIN_WIDTH:g}")
    ax.set_ylabel(f"{label} [{unit}]")
    ax.set_title(f"LSTID {label.lower()} vs {name}  (n={len(param_values)})")

    fig.tight_layout()
    fig.savefig(out_dir / f"driver_characteristics_{name}_{param_key}.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LSTID period/speed/amplitude vs geomagnetic driver activity (violin plots).")
    parser.add_argument("--pkl", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "results/vespagram/lstid_trains_highmed.pkl",
                        help="LSTID records pickle (default: vespagram high+medium trains, written by catalogue_records.py; ROI: results/roi/lstid_trains.pkl, or the raw "
                             "per-front results/roi/lstid_rois.pkl)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder for driver-characteristics plots (default: a "
                             "climatology/ folder beside --pkl, so results stay with "
                             "the run that produced them)")
    parser.add_argument("--drivers", nargs="*", choices=["SME", "IE"], default=["SME", "IE"],
                        help="which drivers to plot against (default: SME IE)")
    return parser.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.pkl.parent / "climatology"
    records = load_records(args.pkl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.drivers:
        driver_values = compute_driver_at_rois(records, name)
        for param_key in PARAMETERS:
            plot_driver_characteristic(name, driver_values, records, param_key, args.out_dir)
            print(f"  {name} / {param_key}: saved -> "
                 f"{args.out_dir / f'driver_characteristics_{name}_{param_key}.png'}")

    print(f"Saved driver-characteristics plots -> {args.out_dir}  (n={len(records)} LSTID ROIs)")


if __name__ == "__main__":
    main()
