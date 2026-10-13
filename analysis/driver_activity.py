"""
driver_activity.py

LSTID occurrence and detection rate as a function of geomagnetic/auroral
activity level, using the mean of each driver over the 3h preceding each
ROI's ref_time.

For each driver:
  - occurrence: histogram of LSTID ROI counts, binned by the driver's 3h
    trailing mean at ref_time.
  - exposure: histogram of the driver's 3h trailing mean evaluated on a
    regular hourly grid spanning the whole dataset (i.e. how often each
    activity level occurred at all, whether or not an LSTID was detected).
  - detection rate [%] = occurrence / exposure * 100 per bin.

Drivers: Hp30, SymH (OMNI), IU/IL/IE (IMAGE), SMU/SML/SME (SuperMAG), all from
the quality-controlled feature table (see driver_regression.py).
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driver_regression import (GRID_MINUTES, load_records, _round_to_grid,
                               _load_driver_series, _rolling_on_grid)

# name (a key of driver_regression.DRIVER_COLS) -> (bin width, cap)
# cap: values beyond this magnitude are clipped into a single overflow bin,
# keeping the rare-extreme tail from spreading LSTID/exposure counts across
# many near-empty bins. Either a single number (symmetric clip to [-cap, cap])
# or a (low, high) tuple for an asymmetric clip (only one side truncated).
DRIVERS = {
    "Hp30": (1,  (None, 9)),
    "SymH": (10, (-150, None)),
    "IU":   (50, 500),
    "IL":   (50, 1000),
    "IE":   (50, 1000),
    "SMU":  (50, 500),
    "SML":  (50, 1000),
    "SME":  (50, 1000),
    "Bz":   (2,  20),
}


def compute_occurrence_and_exposure(records, name):
    """
    Returns (occ_values, detected_values, exposure_values), all the 3h
    trailing mean of driver `name` evaluated on the SAME regular
    GRID_MINUTES-cadence time grid (spanning the full record span):
      occ_values      - grid values at each LSTID ROI's ref_time, rounded to
                        the nearest grid point (one entry per ROI; two ROIs in
                        the same grid slot both appear, for the raw occurrence
                        histogram)
      detected_values - grid values at each DISTINCT grid slot that had >=1
                        LSTID ROI (deduplicated by time); used as the
                        detection-rate numerator so multiple ROIs landing in
                        the same time slot don't push the rate past 100%
      exposure_values - grid values at every point of the regular grid (the
                        "at all" baseline, i.e. how often each activity level
                        occurred whether or not an LSTID was detected)
    """
    rounded_occ = _round_to_grid([r["ref_time"] for r in records])
    grid = pl.datetime_range(rounded_occ.min(), rounded_occ.max(), interval=f"{GRID_MINUTES}m",
                             time_zone="UTC", eager=True)

    exposure_values = _rolling_on_grid(_load_driver_series(name), name, grid).astype(float)
    grid_map        = dict(zip(grid, exposure_values))
    occ_values      = np.array([grid_map.get(t, np.nan) for t in rounded_occ], dtype=float)
    detected_values = np.array([grid_map.get(t, np.nan) for t in set(rounded_occ)], dtype=float)

    return occ_values, detected_values, exposure_values


def _cap_bounds(cap):
    """Normalize a DRIVERS cap entry to a (low, high) tuple. A bare number
    means a symmetric clip to [-cap, cap]; a (low, high) tuple clips each
    side independently (either side may be None to leave it unclipped)."""
    if cap is None:
        return None, None
    if isinstance(cap, tuple):
        return cap
    return -cap, cap


def _bin_values(values, bin_width, cap=None):
    """Round to the nearest bin_width, dropping NaNs. If cap is set, values
    are clipped first so everything beyond it collapses into one overflow
    bin at the cap instead of spreading across many near-empty bins in the
    rare-extreme tail."""
    values = values[~np.isnan(values)]
    low, high = _cap_bounds(cap)
    if low is not None or high is not None:
        values = np.clip(values, low, high)
    return np.round(values / bin_width) * bin_width


MIN_EXPOSURE = 5   # bins with fewer hourly exposure samples than this are dropped from
                    # the detection-rate panel: too few samples for the ratio to be meaningful
                    # (e.g. a single extreme-storm hour can otherwise yield a >100% "rate")


def plot_driver_activity(name, occ_values, detected_values, exposure_values, out_dir):
    bin_width, cap = DRIVERS[name]

    occ_binned  = _bin_values(occ_values, bin_width, cap)
    det_binned  = _bin_values(detected_values, bin_width, cap)
    exp_binned  = _bin_values(exposure_values, bin_width, cap)

    all_levels = np.union1d(np.unique(occ_binned), np.unique(exp_binned))
    occ_counts = np.array([(occ_binned == lv).sum() for lv in all_levels])
    det_counts = np.array([(det_binned == lv).sum() for lv in all_levels])
    exp_counts = np.array([(exp_binned == lv).sum() for lv in all_levels])

    # rate = fraction of grid slots at this activity level with >=1 LSTID
    # detected (det_counts, deduplicated by time slot), NOT raw ROI count
    # (occ_counts) -- two ROIs landing in the same slot must not push past 100%.
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(exp_counts >= MIN_EXPOSURE, det_counts / exp_counts * 100, np.nan)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.bar(all_levels, occ_counts, width=bin_width * 0.9, color="tab:blue",
           label="LSTID occurrence (count)")
    ax1.set_xlabel(f"{name} [{'nT' if name != 'Hp30' else 'index'}], 3h trailing mean, "
                   f"binned to {bin_width:g}  "
                   f"(rate omitted where exposure < {MIN_EXPOSURE} samples)")
    ax1.set_ylabel("LSTID ROI count", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(all_levels, rate, color="tab:red", marker="o", markersize=4,
             linewidth=1.5, label="Detection rate")
    ax2.set_ylabel("Detection rate [%]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(bottom=0)

    cap_low, cap_high = _cap_bounds(cap)
    if cap_low is not None or cap_high is not None:
        # relabel the clipped overflow bin(s) so the cap reads as "9+"/"-150-"
        # rather than a plain tick value indistinguishable from a real bin
        ticks = ax1.get_xticks()
        labels = []
        for t in ticks:
            if cap_high is not None and t >= cap_high:
                labels.append(f"{cap_high:g}+")
            elif cap_low is not None and t <= cap_low:
                labels.append(f"{cap_low:g}-")
            else:
                labels.append(f"{t:g}")
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)

    ax1.set_title(f"LSTID occurrence & detection rate vs {name}  (3h trailing mean, "
                  f"n={len(occ_values[~np.isnan(occ_values)])})", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"driver_activity_{name}.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LSTID occurrence and detection rate vs geomagnetic driver activity.")
    parser.add_argument("--pkl", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "results/vespagram/lstid_hours_highmed.pkl",
                        help="LSTID records pickle (default: vespagram high+medium hours, written by catalogue_records.py; ROI: results/roi/lstid_hours.pkl, or the raw "
                             "per-front results/roi/lstid_rois.pkl)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder for driver-activity plots (default: a "
                             "climatology/ folder beside --pkl, so results stay with "
                             "the run that produced them)")
    parser.add_argument("--drivers", nargs="*", choices=list(DRIVERS), default=list(DRIVERS),
                        help="which drivers to plot (default: all)")
    return parser.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.pkl.parent / "climatology"
    records = load_records(args.pkl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.drivers:
        occ_values, detected_values, exposure_values = compute_occurrence_and_exposure(records, name)
        plot_driver_activity(name, occ_values, detected_values, exposure_values, args.out_dir)
        print(f"  {name}: saved -> {args.out_dir / f'driver_activity_{name}.png'}")

    print(f"Saved driver-activity plots -> {args.out_dir}  (n={len(records)} LSTID ROIs)")


if __name__ == "__main__":
    main()
