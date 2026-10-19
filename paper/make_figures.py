"""
paper/make_figures.py

Figures 1-6 of the Data Descriptor, written to paper/figures/figN.{pdf,png}.

Inputs: the released catalogue (results/vespagram/trains.parquet, detections.parquet,
lstid_hours_highmed.pkl), the keograms (KEO_DIR in paths.py: the archive, or a
folder rebuilt from TEC files by paper/keogram_from_tec.py), the coverage
table (results/vespagram/daily_coverage.csv), the validation numbers
(paper/numbers.json, results/climatology_compare/ratio.csv), the driver series
(FEATURES in paths.py) and the cached summaries in
paper/figures/data/ (tec_sample.py; monthly sunspot number from SILSO).

  python paper/make_figures.py            all figures
  python paper/make_figures.py 1 6        selected figures
"""

import sys
import json
import pickle
import calendar
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import vespagram as v
from detect_roi import Config as RoiConfig, load_grid
from paths import FEATURES
from supermag import read_features

OUT      = HERE / "figures"
DATA     = OUT / "data"
VESPA    = ROOT / "results/vespagram"

MM   = 1 / 25.4
WIDE = 183 * MM                      # double-column width
TIER = {"high": "#0072B2", "medium": "#E69F00", "low": "#999999"}
HM_C, HF_C = "#009E73", "#CC79A7"    # high+medium aggregate; HF-INT
SYMH_C, SME_C = "#5b3f8c", "#b5532a"
KEO_LIM = 0.15                       # TECu, band-passed keograms
CFG = v.VespaConfig()

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, n):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig{n}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"fig{n}.pdf")


def tag(ax, s, x=-0.01, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


# ── catalogue ──────────────────────────────────────────────────────────────────

def catalogue():
    """Train catalogue (every row is one wave: trains crossing midnight are settled
    across days when the catalogue is built)."""
    tr = pd.read_parquet(VESPA / "trains.parquet")
    tr["dur_h"] = (tr["t_end"] - tr["t_start"]).dt.total_seconds() / 3600
    return tr


def detections():
    return pd.read_parquet(VESPA / "detections.parquet")


def trains_in(tr, t0, t1):
    return tr[(tr["t_end"] > t0) & (tr["t_start"] < t1)].sort_values("t_start")


RIDGE = "#FFD400"  # vespagram ridges: visible on the Blues colormap


def draw_fronts(ax, trains, dets):
    """Crest (solid) and trough (dashed) lines of each train, drawn piecewise over the
    latitude slice of each sub-array detection, as in vespagram._draw_train. Time
    axis in matplotlib date units."""
    lat_min, lat_max = 37.5, 57.5
    for _, tr in trains.iterrows():
        m = dets[(dets["date"] == tr["date"]) & (dets["train_id"] == tr["train_id"])]
        midnight = pd.Timestamp(tr["date"])
        for _, d in m.iterrows():
            dd = dict(d)
            dd["lat_c"] = 0.5 * (d["lat_lo"] + d["lat_hi"])
            lo, hi = v._owned_slice(dd, lat_min, lat_max, CFG)
            la = np.linspace(lo, hi, 6)
            off = (la - dd["lat_c"]) * v.M_PER_DEG * d["slowness"]
            for tc, ls in [(c, "-") for c in d["crests_s"]] + [(c, (0, (2.5, 1.5))) for c in d["troughs_s"]]:
                tt = midnight + pd.to_timedelta(tc - off, unit="s")
                ax.plot(mdates.date2num(tt), la, color="k", lw=0.7, ls=ls)


def train_strip(ax, trains, t0, t1):
    """One bar per train (trains never overlap in time), coloured by confidence and
    labelled with its speed."""
    for _, tr in trains.iterrows():
        s, e = max(tr["t_start"], t0), min(tr["t_end"], t1)
        x0, x1 = mdates.date2num(s), mdates.date2num(e)
        ax.barh(0, x1 - x0, left=x0, height=0.72, color=TIER[tr["confidence"]], lw=0)
        if e - s < pd.Timedelta(minutes=40):      # cut by the window edge: no room for a label
            continue
        ax.text(0.5 * (x0 + x1), 0, f"{tr['speed_m_s']:.0f}", ha="center", va="center",
                fontsize=5.5, color="k" if tr["confidence"] != "high" else "white")
    ax.set_ylim(0.5, -0.5)
    ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))
    ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)


def tier_legend(ax, loc="upper left", **kw):
    h = [plt.Rectangle((0, 0), 1, 1, color=TIER[k]) for k in TIER]
    return ax.legend(h, [k.capitalize() for k in TIER], loc=loc, ncol=3, handlelength=1.2,
                     columnspacing=1.0, **kw)


def bandpassed(t0, t1):
    """Band-passed, decimated keogram stitched over [t0, t1) from daily grids, each
    filtered with the padding of the catalogue."""
    blocks, times = [], []
    day = t0.normalize()
    while day < t1:
        X, lats, t = v.load_and_prepare(v.VespaConfig(input_date=day.date()))
        k = (t >= 0) & (t < 86400)
        blocks.append(X[:, k]); times.append(day + pd.to_timedelta(t[k], unit="s"))
        day += pd.Timedelta(days=1)
    X, tt = np.concatenate(blocks, axis=1), np.concatenate(times)
    k = (tt >= t0) & (tt < t1)
    return X[:, k], lats, pd.DatetimeIndex(tt[k])


def keogram(ax, X, lats, times, lim=KEO_LIM):
    x = mdates.date2num(times)
    dx = x[1] - x[0]
    im = ax.imshow(X, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", origin="upper",
                   extent=[x[0] - dx / 2, x[-1] + dx / 2, lats.min(), lats.max()],
                   interpolation="nearest", rasterized=True)
    ax.set_ylabel("Latitude (°N)")
    ax.set_yticks([40, 45, 50, 55])
    return im


def time_axis(ax, t0, t1, step_h=3, labels=True):
    ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, step_h)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    if not labels:
        ax.tick_params(labelbottom=False)


def day_labels(ax, t0, t1):
    """Date of each day at its midnight tick (or at the start of the window)."""
    d = t0.normalize()
    while d < t1:
        x = max(d, t0)
        ax.annotate(f"{x.day} {x:%b %Y}", (mdates.date2num(x), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -15), textcoords="offset points", fontsize=6.5, ha="left", va="top")
        d += pd.Timedelta(days=1)


def midnights(ax, t0, t1):
    for d in [d for d in pd.date_range(t0.normalize(), t1, freq="D") if t0 < d < t1]:
        ax.axvline(mdates.date2num(d), color="k", lw=0.6, ls=":")


def colorbar(fig, im, ax, label, ticks=None):
    cax = ax.inset_axes([1.01, 0.0, 0.012, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.set_label(label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    return cb


# ── Figure 1: processing chain ────────────────────────────────────────────────

SPEEDS = [300, 500, 700, 1000, 1500]


def speed_axis(ax, labels=True, speeds=SPEEDS):
    ax.set_yticks([1e3 / u for u in speeds], [str(u) for u in speeds] if labels else [])
    for u in (CFG.lstid_min_speed, CFG.lstid_max_speed):
        ax.axhline(1e3 / u, color="0.45", lw=0.4, ls=(0, (3, 2)))


def vespa(ax, r, t0):
    shown = np.where(np.nan_to_num(r["A"]) >= CFG.beam_rms_min, r["S"], np.nan)
    x = mdates.date2num(t0 + pd.to_timedelta(r["t"], unit="s"))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower",
                   extent=[x[0], x[-1], r["p"][0] * 1e3, r["p"][-1] * 1e3],
                   interpolation="nearest", rasterized=True)
    ax.set_facecolor("0.9")
    for d in r["dets"]:
        kept = "train_id" in d
        xx = mdates.date2num(t0 + pd.to_timedelta(d["ridge_t"], unit="s"))
        ax.scatter(xx, d["ridge_p"] * 1e3, s=0.6, lw=0, c=RIDGE if kept else "#D55E00",
                   rasterized=True, zorder=3)
    return im


def fig1():
    day = pd.Timestamp("2024-08-12")
    t0, t1 = day, day + pd.Timedelta(days=1)
    cfg = v.VespaConfig(input_date=day.date())
    Z, lats_raw, traw, _ = load_grid(RoiConfig(keo_folder=cfg.keo_folder, input_date=cfg.input_date,
                                               border_pad_hours=0))
    X, lats, t, results, *_ = v.detect(cfg)
    for r in results:
        r["t"] = t
    tr = trains_in(catalogue(), t0, t1)
    de = detections()

    fig = plt.figure(figsize=(WIDE, 155 * MM))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 0.6, 1.4], hspace=0.42, figure=fig)
    ge = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3], height_ratios=[0.32, 1], hspace=0.06)

    ax = fig.add_subplot(gs[0])
    im = keogram(ax, Z, lats_raw, day + pd.to_timedelta(traw, unit="s"), lim=0.3)
    time_axis(ax, t0, t1, labels=False)
    colorbar(fig, im, ax, "dTEC (TECu)")
    tag(ax, "a", x=-0.07)
    ax.set_title("Detrended TEC keogram, 0.05° × 30 s", loc="left")

    ax = fig.add_subplot(gs[1])
    k = (t >= 0) & (t < 86400)
    im = keogram(ax, X[:, k], lats, day + pd.to_timedelta(t[k], unit="s"))
    time_axis(ax, t0, t1, labels=False)
    colorbar(fig, im, ax, "dTEC (TECu)")
    tag(ax, "b", x=-0.07)
    ax.set_title("Band-passed (25–200 min) and decimated, 0.25° × 2 min", loc="left")

    sub = GridSpecFromSubplotSpec(1, len(results), subplot_spec=gs[2], wspace=0.08)
    for i, r in enumerate(results):
        ax = fig.add_subplot(sub[i])
        im = vespa(ax, r, day)
        speed_axis(ax, labels=(i == 0), speeds=[300, 500, 1000])  # the panel is too short for more
        ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{r['lat_lo']:g}–{r['lat_hi']:g}°N", loc="left")
        if i == 0:
            ax.set_ylabel("Speed (m s$^{-1}$)")
            tag(ax, "c", x=-0.22, y=1.15)
    colorbar(fig, im, ax, "Semblance", ticks=[0, 0.5, 1])

    ax_s = fig.add_subplot(ge[0])
    ax = fig.add_subplot(ge[1], sharex=ax_s)
    train_strip(ax_s, tr, t0, t1)
    tag(ax_s, "d", x=-0.07, y=0.2)
    ax_s.set_title("Trains (bar colour: confidence level; label: speed in m s$^{-1}$)",
                   loc="left")
    tier_legend(ax_s, loc="lower right", bbox_to_anchor=(1.0, 0.95))
    k = (t >= 0) & (t < 86400)
    im = keogram(ax, X[:, k], lats, day + pd.to_timedelta(t[k], unit="s"))
    draw_fronts(ax, tr, de)
    ax.set_ylim(37.5, 57.5)
    time_axis(ax, t0, t1)
    colorbar(fig, im, ax, "dTEC (TECu)")
    ax.set_xlabel("UT (h), 12 August 2024")
    save(fig, 1)


# ── Figure 2: coverage ─────────────────────────────────────────────────────────

CONST = {"G": ("GPS", "#000000"), "R": ("GLONASS", "#56B4E9"),
         "E": ("Galileo", "#009E73"), "C": ("BeiDou", "#D55E00")}


def fig2():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cov = pd.read_csv(ROOT / "results/vespagram/daily_coverage.csv", parse_dates=["date"])
    con = pd.read_csv(DATA / "constellations.csv", parse_dates=["date"])
    ipp = np.load(DATA / "ipp_density.npz")

    fig = plt.figure(figsize=(WIDE, 120 * MM))
    gs = GridSpec(2, 2, height_ratios=[1, 1.35], hspace=0.38, wspace=0.3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(cov["date"], cov["n_stations"], color="k", lw=0.4)
    ax.set_ylabel("Stations per day")
    ax.set_ylim(0, 320)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    tag(ax, "a", x=-0.13)

    ax = fig.add_subplot(gs[0, 1])
    for c, (name, col) in CONST.items():
        if c in con:
            ax.plot(con["date"], con[c] / 1e6, color=col, lw=0.8, label=name)
            ax.annotate(name, (con["date"].iloc[-1], con[c].iloc[-1] / 1e6), xytext=(3, 0),
                        textcoords="offset points", fontsize=6.5, va="center", color="k")
    ax.set_ylabel("Valid observations per day (10$^6$)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=pd.Timestamp("2027-06-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handlelength=1.2, columnspacing=1.0)
    tag(ax, "b", x=-0.1)

    proj = ccrs.LambertConformal(central_longitude=12, central_latitude=47,
                                 standard_parallels=(40, 55))
    vmax = max(ipp[f"d{i}"].max() for i in range(len(ipp["days"])))
    for i, day in enumerate(ipp["days"]):
        ax = fig.add_subplot(gs[1, i], projection=proj)
        ax.set_extent([-8, 32, 33, 61], crs=ccrs.PlateCarree())
        H = np.where(ipp[f"d{i}"] > 0, ipp[f"d{i}"], np.nan)
        im = ax.pcolormesh(ipp["lon"], ipp["lat"], H, cmap="Blues",
                           norm=LogNorm(vmin=10, vmax=vmax), transform=ccrs.PlateCarree(),
                           rasterized=True)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="0.25")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.2, edgecolor="0.5")
        for lat in (37.5, 57.5):
            ax.plot(np.linspace(-8, 32, 50), np.full(50, lat), color="#D55E00", lw=0.9,
                    ls=(0, (4, 2)), transform=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, lw=0.2, color="0.6", x_inline=False, y_inline=False,
                          xlocs=range(-10, 41, 10), ylocs=range(30, 66, 10))
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {"size": 6, "rotation": 0}
        gl.ylabel_style = {"size": 6}
        n, ts = int(np.nansum(H)), pd.Timestamp(str(day))
        ax.set_title(f"{ts.day} {ts:%B %Y} ({n / 1e6:.1f} × 10$^6$ pierce points)", loc="left")
        tag(ax, "c" if i == 0 else "", x=-0.1, y=1.06)
    cax = fig.add_axes([0.92, 0.08, 0.01, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pierce points per 0.5° × 0.5° cell", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, 2)


# ── Figure 3: technical validation ────────────────────────────────────────────

LEVELS = [("high", "High", TIER["high"]), ("medium", "Medium", TIER["medium"]),
          ("low", "Low", TIER["low"]), ("high_medium", "High +\nmedium", HM_C)]


def match_bars(ax, d, key, chance_key, ylabel):
    for i, (k, label, color) in enumerate(LEVELS):
        val, ch = d[k][key], d[k][chance_key]
        ax.bar(i, val, width=0.62, color=color, lw=0)
        ax.plot([i - 0.38, i + 0.38], [ch, ch], color="k", lw=1.0)
        ax.text(i, max(val, ch) + 2, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(LEVELS)), [l for _, l, _ in LEVELS])
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.plot([], [], color="k", lw=1.0, label="Chance level")
    ax.legend(loc="upper left")


def fig3():
    N = json.load(open(HERE / "numbers.json"))
    ratio = pd.read_csv(ROOT / "results/climatology_compare/ratio.csv")
    hp = ratio[ratio["group"].str.startswith("Hp30")]

    fig, axes = plt.subplots(2, 2, figsize=(WIDE, 105 * MM))
    fig.subplots_adjust(hspace=0.5, wspace=0.28)

    ax = axes[0, 0]
    match_bars(ax, N["published_roi"]["train_matching"], "published_matched_pct",
               "published_chance_pct", "Published trains recovered (%)")
    ax.set_title(f"Published GNSS-TEC trains (n = {N['published_roi']['n_trains']})")
    tag(ax, "a", x=-0.15)

    ax = axes[0, 1]
    match_bars(ax, N["hfint"]["matching"], "south_matched_pct", "south_chance_pct",
               "HF-INT events recovered (%)")
    ax.set_title(f"HF-INT southward events (n = {N['hfint']['n_south']})")
    tag(ax, "b", x=-0.15)

    ax = axes[1, 0]
    xi = np.arange(len(hp))
    r = hp["ratio_vespagram_hm"].to_numpy()
    ax.plot(xi, r, "o", ms=3.5, color=HM_C)
    for x, v in zip(xi, r):
        ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.5)
    # hours with an LSTID in each catalogue: the counts the ratio is made of
    ax.text(-0.45, 7.0, "Active hours: ROI detector (upper row), vespagram (lower row)",
            ha="left", va="center", fontsize=6, color="0.3")
    for y, col in [(6.3, "occ_roi"), (5.6, "occ_vespagram_hm")]:
        for x, n in zip(xi, hp[col] * hp["hours"] / 100):
            ax.text(x, y, f"{n:,.0f}", ha="center", va="center", fontsize=6, color="0.3")
    ax.axhline(1, color="0.5", lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(5))
    ax.set_xticks(xi, hp["level"])
    ax.set_xlim(-0.5, len(xi) - 0.5)
    ax.set_xlabel("Hp30, 3-h mean")
    ax.set_ylabel("Occurrence ratio,\nvespagram / ROI detector")
    ax.set_ylim(0, 7.4)
    tag(ax, "c", x=-0.15)

    ax = axes[1, 1]
    occ = N["hfint"]["occurrence"]
    yrs = sorted(int(y) for y in occ["hm_per_year"])
    for key, label, color in [("hm_per_year", "High + medium trains", HM_C),
                              ("south_per_year", "HF-INT southward events", HF_C)]:
        n = np.array([occ[key][str(y)] for y in yrs], float)
        ax.plot(yrs, n / n.mean(), "o-", color=color, ms=3, lw=1.0, label=label)
    ax.set_ylabel("Yearly count / 2015–2025 mean")
    ax.set_xticks(yrs[::2])
    ax.set_ylim(0, 2.2)
    ax.legend(loc="upper left")
    ax.text(0.98, 0.05, f"r = {occ['corr_year']:.2f}", transform=ax.transAxes, ha="right")
    tag(ax, "d", x=-0.15)
    save(fig, 3)


# ── Figure 4: case events ─────────────────────────────────────────────────────

CASES = [("a", pd.Timestamp("2024-08-11 12:00"), pd.Timestamp("2024-08-13 00:00")),
         ("b", pd.Timestamp("2015-03-17 00:00"), pd.Timestamp("2015-03-18 03:00")),
         ("c", pd.Timestamp("2024-05-10 12:00"), pd.Timestamp("2024-05-11 12:00"))]


def fig4():
    tr, de = catalogue(), detections()
    drv = read_features(["time", "sym_h", "sme"]).to_pandas().set_index("time")
    drv.index = drv.index.tz_localize(None)

    fig = plt.figure(figsize=(WIDE, 225 * MM))
    outer = GridSpec(3, 1, hspace=0.32, figure=fig)
    for (letter, t0, t1), spec in zip(CASES, outer):
        gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=spec, hspace=0.12,
                                     height_ratios=[0.5, 2.2, 0.75, 0.75])
        trs = trains_in(tr, t0, t1)
        ax_s = fig.add_subplot(gs[0])
        train_strip(ax_s, trs, t0, t1)
        tag(ax_s, letter, x=-0.07, y=0.3)
        if letter == "a":
            tier_legend(ax_s, loc="lower right", bbox_to_anchor=(1.0, 0.95))

        ax = fig.add_subplot(gs[1], sharex=ax_s)
        X, lats, times = bandpassed(t0, t1)
        im = keogram(ax, X, lats, times)
        draw_fronts(ax, trs, de)
        ax.set_ylim(37.5, 57.5)
        time_axis(ax, t0, t1, labels=False)
        midnights(ax, t0, t1)
        colorbar(fig, im, ax, "dTEC (TECu)")

        d = drv.loc[t0:t1]
        for k, (col, label, color) in enumerate([("sym_h", "SYM-H\n(nT)", SYMH_C),
                                                 ("sme", "SME\n(nT)", SME_C)]):
            a = fig.add_subplot(gs[2 + k], sharex=ax_s)
            a.plot(mdates.date2num(d.index), d[col], color=color, lw=0.6)
            a.set_ylabel(label)
            if col == "sym_h":
                a.axhline(0, color="0.6", lw=0.4)
            time_axis(a, t0, t1, labels=(k == 1))
            midnights(a, t0, t1)
        a.set_xlabel("UT (h)", labelpad=12)
        day_labels(a, t0, t1)
    save(fig, 4)


# ── Figure 5: parameter distributions ─────────────────────────────────────────

def fig5():
    tr = catalogue()
    tr["slowness_s_km"] = 1e3 / tr["speed_m_s"]
    # bin in slowness, 3 grid steps per bin
    dp = 1e3 * CFG.slowness_step * 3
    p_bins = np.arange(1e3 / 1500 - dp / 2, 1e3 / 300 + dp, dp)
    panels = [("slowness_s_km", "Speed (m s$^{-1}$)", p_bins, False),
              ("period_min", "Period (min)", np.arange(30, 125, 5), False),
              ("wavelength_km", "Wavelength (km)", np.arange(500, 5600, 200), False),
              ("amplitude", "Amplitude (TECu)", np.geomspace(0.01, 0.5, 26), True),
              ("dur_h", "Duration (h)", np.arange(0.5, 24.5, 1.0), False)]
    hm = tr["confidence"].isin(["high", "medium"])
    groups = [("High + medium confidence", HM_C, hm), ("Low confidence", TIER["low"], ~hm)]
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 92 * MM))
    fig.subplots_adjust(hspace=0.45, wspace=0.28)
    for ax, (col, label, bins, logx), letter in zip(axes.ravel(), panels, "abcde"):
        for name, color, m in groups:
            x = tr.loc[m, col].to_numpy()
            w = np.full(len(x), 100 / len(x))
            ax.hist(np.clip(x, bins[0], bins[-1] - 1e-9), bins=bins, weights=w, histtype="step",
                    color=color, lw=1.2, label=f"{name} (n = {len(x)})")
            ax.axvline(np.median(x), color=color, lw=0.6, ls=(0, (2, 1.5)))
        if logx:
            ax.set_xscale("log")
        if col == "slowness_s_km":                 # speed labels, fast on the right
            sp = [300, 400, 500, 700, 1000]
            ax.set_xticks([1e3 / s for s in sp], [str(s) for s in sp])
            ax.invert_xaxis()
        ax.set_xlabel(label)
        ax.set_ylabel("Share of trains (%)")
        tag(ax, letter, x=-0.16)
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].axis("off")
    axes[1, 2].legend(h, l, loc="center", handlelength=1.5)
    save(fig, 5)


# ── Figure 6: catalogue overview ──────────────────────────────────────────────

def fig6():
    tr = catalogue()
    hm = tr[tr["confidence"].isin(["high", "medium"])]
    mid = hm["t_start"] + (hm["t_end"] - hm["t_start"]) / 2
    monthly = mid.dt.to_period("M").value_counts().sort_index()
    months = pd.period_range("2015-01", "2025-12", freq="M")
    monthly = monthly.reindex(months, fill_value=0)

    ssn = pd.read_csv(DATA / "SN_m_tot_V2.0.csv", sep=";", header=None,
                      names=["y", "m", "frac", "ssn", "sd", "n", "prov"])
    ssn["month"] = pd.PeriodIndex.from_fields(year=ssn["y"], month=ssn["m"], freq="M")
    ssn = ssn.set_index("month")["ssn"].reindex(months)

    hours = pickle.load(open(VESPA / "lstid_hours_highmed.pkl", "rb"))
    active = pd.DatetimeIndex(sorted({pd.Timestamp(r["ref_time"]) for r in hours}))
    grid = pd.date_range("2015-01-01", "2025-12-31 23:00", freq="h")
    y = grid.isin(active)
    H = pd.DataFrame({"m": grid.month, "h": grid.hour, "y": y}) \
          .pivot_table(index="m", columns="h", values="y", aggfunc="mean").to_numpy() * 100
    sza = pd.read_parquet(FEATURES, columns=["sza"])
    sza = sza[(sza.index.year >= 2015) & (sza.index.year <= 2025)]
    S = sza.groupby([sza.index.month, sza.index.hour])["sza"].mean().unstack().to_numpy()

    fig = plt.figure(figsize=(WIDE, 72 * MM))
    gs = GridSpec(2, 2, width_ratios=[1.35, 1], height_ratios=[0.8, 1], hspace=0.15,
                  wspace=0.28, figure=fig)
    x = months.to_timestamp() + pd.Timedelta(days=14)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, ssn.to_numpy(), width=24, color="0.55", lw=0)
    ax.set_ylabel("Sunspot\nnumber")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(labelbottom=False)
    tag(ax, "a", x=-0.09)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.bar(x, monthly.to_numpy(), width=24, color=HM_C, lw=0)
    ax2.set_ylabel("High + medium\ntrains per month")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(pd.Timestamp("2014-12-01"), pd.Timestamp("2026-01-31"))
    tag(ax2, "b", x=-0.09)

    ax = fig.add_subplot(gs[:, 1])
    im = ax.imshow(H, aspect="auto", origin="lower", cmap="Blues",
                   extent=[-0.5, 23.5, 0.5, 12.5], vmin=0)
    # terminator at the IPP height: the Sun sets there when its zenith angle at the ground
    # exceeds 90 deg by the dip of the horizon, arccos(R / (R + h))
    h_km, r_km = 300, 6371
    sza_300 = 90 + np.degrees(np.arccos(r_km / (r_km + h_km)))
    ax.contour(np.arange(24), np.arange(1, 13), S, levels=[sza_300], colors="#D55E00",
               linewidths=1.0, linestyles="--")
    ax.plot([], [], color="#D55E00", lw=1.0, ls="--",
            label=f"Solar terminator at {h_km} km (SZA {sza_300:.0f}°), 47.5°N, 10°E")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_xticks(range(0, 24, 3))
    ax.set_yticks(range(1, 13), [calendar.month_abbr[i] for i in range(1, 13)])
    ax.set_xlabel("UT (h)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    cb = colorbar(fig, im, ax, "Hours with a high- or medium-\nconfidence LSTID (%)")
    tag(ax, "c", x=-0.1)
    save(fig, 6)


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for n in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else FIGS):
        FIGS[n]()
