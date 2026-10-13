"""
climatology.py

LSTID ROI climatology: hourly, monthly, and yearly distribution of detected
LSTID events, plus a month x hour heatmap of seasonal-diurnal structure.

Reads the combined lstid_rois.pkl produced by detect_roi.py's batch mode
(one record per LSTID ROI, tagged with ref_time). Binning uses ref_time in UT.
"""

import argparse
import calendar
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# the driver overlays come from the quality-controlled feature table
from driver_regression import load_records, _feature

# overlay short name -> feature-table column
OVERLAY_SOURCES = {
    "wind": "wind_meridional",
    "sme":  "sme",
    "ie":   "image_ie",
}
# hourly-panel indices: the IMAGE electrojets on the keogram meridian. Their UT cycle
# is the magnetic-local-time cycle of the auroral source region upstream of the
# array. A global index like SME has a much flatter UT cycle, partly shaped by
# where the network's stations are, so it does not describe the local source.
HOURLY_SOURCES = {
    "iu": "image_iu",    # eastward electrojet (afternoon-dusk sector), positive
    "il": "image_il",    # westward electrojet (midnight-morning sector), negative
}
F107_COL = "f107"
HP30_COL = "hp30"
SZA_COL  = "sza"

def _minmax(a):
    """Scale an array to 0-1 by its own min/max (NaN-safe)."""
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _add_cell_grid(ax, n_rows, n_cols):
    """Draw thin gridlines at heatmap cell boundaries (imshow with origin='lower')."""
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.4)
    ax.tick_params(which="minor", length=0)


def _mean_month_hour(col, years):
    """Mean of feature-table column `col` by (month, hour), averaged over
    `years` and over all days/minutes within each hour. Returns a (12, 24)
    array matching the LSTID month x hour heatmap's cell layout."""
    df = _feature(col).with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("time").dt.month().alias("month"),
        pl.col("time").dt.hour().alias("hour"),
    ).filter(pl.col("year").is_in(years))
    grid = (df.group_by(["month", "hour"]).agg(pl.col(col).mean().alias("v"))
              .sort(["month", "hour"]))

    values = np.full((12, 24), np.nan)
    for m, h, v in grid.iter_rows():
        values[m - 1, h] = v
    return values


def load_index_month_hour(years, name):
    """Mean of overlay variable `name` (one of OVERLAY_SOURCES) by (month, hour)."""
    return _mean_month_hour(OVERLAY_SOURCES[name], years)


def load_sza_month_hour(years):
    """Mean solar zenith angle [deg] by (month, hour) over `years`. The SZA=90 deg
    contour of this grid is the mean solar terminator (day/night boundary) as a
    function of month and UT hour."""
    return _mean_month_hour(SZA_COL, years)


def load_f107_by_year(years):
    """Mean F10.7 solar flux per year. Returns a dict {year: mean}."""
    df = _feature(F107_COL).with_columns(pl.col("time").dt.year().alias("year")) \
                              .filter(pl.col("year").is_in(years))
    grid = df.group_by("year").agg(pl.col(F107_COL).mean().alias("v"))
    return dict(zip(grid["year"], grid["v"]))


def load_hp30_by_month(years):
    """Mean Hp30 per calendar month, averaged over `years`. Returns a
    length-12 array indexed [0]=Jan .. [11]=Dec."""
    df = _feature(HP30_COL).with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("time").dt.month().alias("month"),
    ).filter(pl.col("year").is_in(years))
    grid = df.group_by("month").agg(pl.col(HP30_COL).mean().alias("v")).sort("month")

    values = np.full(12, np.nan)
    for m, v in grid.iter_rows():
        values[m - 1] = v
    return values


def load_index_by_hour(years, name):
    """
    Mean of overlay variable `name` (instantaneous, no trailing smoothing) by
    UT hour, averaged over `years`. Returns a length-24 array indexed by hour.
    `name` is a key of OVERLAY_SOURCES or HOURLY_SOURCES; the hourly panel uses
    the IMAGE IU/IL electrojets.
    """
    col = {**OVERLAY_SOURCES, **HOURLY_SOURCES}[name]
    df = _feature(col).with_columns(
        pl.col("time").dt.year().alias("year"),
        pl.col("time").dt.hour().alias("hour"),
    ).filter(pl.col("year").is_in(years))
    grid = df.group_by("hour").agg(pl.col(col).mean().alias("v")).sort("hour")

    values = np.full(24, np.nan)
    for h, v in grid.iter_rows():
        values[h] = v
    return values


def _month_hour_heatmap(heat, n_records, out_path, overlay=None, overlay_label=None,
                        sza=None):
    """Draw the month x hour ROI-count heatmap, optionally with one contour
    overlay (a (12, 24) array, e.g. from load_index_month_hour) on top, and/or
    the solar terminator (the SZA=90 deg contour of a (12, 24) mean-SZA grid)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("Hour (UT)")
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    ax.set_ylabel("Month")
    title = f"LSTID count by month x hour (n={n_records})"
    if overlay_label:
        title += f"  +  {overlay_label} contour"
    ax.set_title(title)
    _add_cell_grid(ax, n_rows=12, n_cols=24)
    fig.colorbar(im, ax=ax).set_label("LSTID count")

    has_legend = False
    if overlay is not None:
        # smooth: raw month x hour means of bursty indices (SME/IE) are noisy
        # enough to fragment contours into many small closed loops otherwise
        smoothed = gaussian_filter(overlay, sigma=1, mode="wrap")
        cs = ax.contour(np.arange(24), np.arange(12), smoothed, levels=6, colors="white",
                        linewidths=1.2, linestyles="solid")
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
        ax.plot([], [], color="white", linewidth=1.2, label=overlay_label)
        has_legend = True

    if sza is not None:
        # solar terminator = SZA 90 deg boundary (sunrise/sunset in UT per month)
        ax.contour(np.arange(24), np.arange(12), sza, levels=[90],
                   colors="orange", linewidths=2.0, linestyles="--")
        ax.plot([], [], color="orange", linewidth=2.0, linestyle="--",
                label="Solar terminator (SZA=90 deg)")
        has_legend = True

    if has_legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_climatology(records, out_dir, overlays=None, f107_by_year=None,
                     hp30_by_month=None, iu_by_hour=None, il_by_hour=None,
                     sza_month_hour=None):
    """overlays: optional dict of {name: (12, 24) array} to contour onto
    separate copies of the month x hour heatmap (see load_index_month_hour);
    one output file per overlay, plus one plain version with no overlay.
    f107_by_year/hp30_by_month/iu_by_hour/il_by_hour: optional driver arrays
    (see load_f107_by_year, load_hp30_by_month, load_index_by_hour)
    drawn as twin-axis lines on the yearly/monthly/hourly bar charts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hours  = np.array([r["ref_time"].hour for r in records])
    months = np.array([r["ref_time"].month for r in records])
    years  = np.array([r["ref_time"].year for r in records])

    # ── Hourly distribution ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(24), np.bincount(hours, minlength=24), color="tab:blue")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("Hour (UT)")
    ax.set_ylabel("LSTID count", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title(f"Hourly distribution of LSTIDs (n={len(records)})")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    if iu_by_hour is not None or il_by_hour is not None:
        # min-max normalize each index to 0-1 so their diurnal SHAPES are
        # comparable on one axis. IL is negative (westward electrojet), so it is
        # plotted as -IL: up = stronger electrojet for both curves.
        ax2 = ax.twinx()
        if iu_by_hour is not None:
            ax2.plot(np.arange(24), _minmax(iu_by_hour), color="tab:red", marker="o",
                    markersize=4, linewidth=1.5, label="IMAGE IU, eastward (norm.)")
        if il_by_hour is not None:
            ax2.plot(np.arange(24), _minmax(-il_by_hour), color="tab:purple", marker="s",
                    markersize=4, linewidth=1.5, label="IMAGE −IL, westward (norm.)")
        ax2.set_ylabel("Index (min-max normalized)", color="dimgray")
        ax2.tick_params(axis="y", labelcolor="dimgray")
        ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "climatology_hourly.png", dpi=150)
    plt.close(fig)

    # ── Monthly distribution ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(1, 13), np.bincount(months, minlength=13)[1:], color="tab:orange")
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    ax.set_xlabel("Month")
    ax.set_ylabel("LSTID count", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title(f"Monthly distribution of LSTIDs (n={len(records)})")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    if hp30_by_month is not None:
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, 13), hp30_by_month, color="tab:red", marker="o",
                markersize=4, linewidth=1.5, label="Hp30 (mean)")
        ax2.set_ylabel("Hp30 [index]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "climatology_monthly.png", dpi=150)
    plt.close(fig)

    # ── Yearly distribution ──
    year_range = np.arange(years.min(), years.max() + 1)
    counts     = np.array([(years == y).sum() for y in year_range])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(year_range, counts, color="tab:green")
    ax.set_xticks(year_range)
    ax.set_xticklabels(year_range, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("LSTID count", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title(f"Yearly distribution of LSTIDs (n={len(records)})")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    if f107_by_year is not None:
        f107_vals = [f107_by_year.get(y, np.nan) for y in year_range]
        ax2 = ax.twinx()
        ax2.plot(year_range, f107_vals, color="tab:red", marker="o",
                markersize=5, linewidth=1.5, label="F10.7 (mean)")
        ax2.set_ylabel("F10.7 [sfu]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "climatology_yearly.png", dpi=150)
    plt.close(fig)

    # ── Month x hour heatmap(s) ──
    heat = np.zeros((12, 24), dtype=int)
    for h, m in zip(hours, months):
        heat[m - 1, h] += 1

    _month_hour_heatmap(heat, len(records), out_dir / "climatology_month_hour_heatmap.png",
                        sza=sza_month_hour)

    overlay_labels = {
        "wind": f"Meridional wind [m/s] ({OVERLAY_SOURCES['wind']})",
        "sme":  "SuperMAG SME [nT]",
        "ie":   "IMAGE IE [nT]",
    }
    for name, grid in (overlays or {}).items():
        label = overlay_labels.get(name, name)
        _month_hour_heatmap(heat, len(records),
                            out_dir / f"climatology_month_hour_heatmap_{name}.png",
                            overlay=grid, overlay_label=label, sza=sza_month_hour)

    # ── Month x year heatmap ──
    heat_my = np.zeros((12, len(year_range)), dtype=int)
    for y, m in zip(years, months):
        heat_my[m - 1, y - year_range[0]] += 1

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(year_range)), 6))
    im = ax.imshow(heat_my, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(np.arange(len(year_range)))
    ax.set_xticklabels(year_range, rotation=45)
    ax.set_xlabel("Year")
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    ax.set_ylabel("Month")
    ax.set_title(f"LSTID count by month x year (n={len(records)})")
    _add_cell_grid(ax, n_rows=12, n_cols=len(year_range))
    fig.colorbar(im, ax=ax).set_label("LSTID count")
    fig.tight_layout()
    fig.savefig(out_dir / "climatology_month_year_heatmap.png", dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LSTID climatological analysis.")
    parser.add_argument("--pkl", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "results/vespagram/lstid_trains_highmed.pkl",
                        help="LSTID records pickle (default: vespagram high+medium trains, written by catalogue_records.py; ROI: results/roi/lstid_trains.pkl, or the raw "
                             "per-front results/roi/lstid_rois.pkl)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder for climatology plots "
                             "(default: a climatology/ folder beside --pkl, so results "
                             "stay with the run that produced them)")
    parser.add_argument("--overlays", nargs="*", choices=list(OVERLAY_SOURCES),
                        default=list(OVERLAY_SOURCES),
                        help="which variables to contour onto separate month x hour "
                             "heatmaps (default: all of %(choices)s; pass with no "
                             "values, or omit the flag with --no-overlays, to skip)")
    parser.add_argument("--no-overlays", action="store_true",
                        help="skip all contour overlays")
    parser.add_argument("--no-driver-panels", action="store_true",
                        help="skip the F10.7/Hp30/IMAGE IU+IL line overlays on the "
                             "yearly/monthly/hourly bar charts")
    parser.add_argument("--no-terminator", action="store_true",
                        help="skip the solar terminator (SZA=90 deg) on the "
                             "month x hour heatmaps")
    return parser.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.pkl.parent / "climatology"
    records = load_records(args.pkl)
    years   = sorted({r["ref_time"].year for r in records})

    overlays = {}
    if not args.no_overlays:
        for name in args.overlays:
            overlays[name] = load_index_month_hour(years, name)

    f107_by_year = hp30_by_month = iu_by_hour = il_by_hour = None
    if not args.no_driver_panels:
        f107_by_year  = load_f107_by_year(years)
        hp30_by_month = load_hp30_by_month(years)
        iu_by_hour    = load_index_by_hour(years, "iu")
        il_by_hour    = load_index_by_hour(years, "il")

    sza_month_hour = None if args.no_terminator else load_sza_month_hour(years)

    plot_climatology(records, args.out_dir, overlays=overlays,
                     f107_by_year=f107_by_year, hp30_by_month=hp30_by_month,
                     iu_by_hour=iu_by_hour, il_by_hour=il_by_hour,
                     sza_month_hour=sza_month_hour)
    print(f"Saved climatology plots -> {args.out_dir}  (n={len(records)} LSTID records)")


if __name__ == "__main__":
    main()
