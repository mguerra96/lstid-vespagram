"""
compare_climatology.py

Side-by-side climatology and driver analysis of the ROI and vespagram LSTID
catalogues, on the record pickles written by catalogue_records.py.

Both methods are evaluated on ONE common hourly grid (2015-01-01 .. 2025-12-31,
every hour; every day has a keogram). Occurrence = share of hours with an
LSTID present ("occupancy"), which is independent of how each method splits
activity into events.

Figures (results/climatology_compare/):
  occurrence.png       occupancy vs UT hour, month, year (+ F10.7), both methods
  month_hour.png       month x UT-hour occupancy maps + solar terminator
  characteristics.png  per-train speed, period, amplitude, duration distributions
  drivers.png          adjusted driver effect, added skill over season/LT/F10.7,
                       occupancy vs SME, response lag
  tiers.png            vespagram confidence tiers (high / medium / low) vs drivers
  drivers.csv          regression table for every (catalogue, driver)

Driver model (as driver_regression.py): one logistic regression per driver,
  logit P(LSTID in hour) ~ z(driver, 3 h trailing mean) + season (annual +
  semiannual) + UT (diurnal + semidiurnal) + z(F10.7)
and dAUC = AUC(driver + controls) - AUC(controls only): the skill the driver
adds beyond the regular seasonal/diurnal/solar-cycle pattern.
"""

import sys
import pickle
import calendar
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vespagram
import climatology as clim
from driver_regression import (DRIVER_COLS, _load_driver_series, _rolling_on_grid,
                               _f107_series, _zscore, F107_COL)

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results/climatology_compare"

ROI_DIR   = ROOT / "results/roi"      # record pickles of catalogue_records.py
VESPA_DIR = ROOT / "results/vespagram"
# key -> (label, colour, records folder, pickle suffix)
METHODS = {"roi":          ("ROI", "#d9822b", ROI_DIR, ""),
           "vespagram":    ("Vespagram, all tiers", "#8fb3de", VESPA_DIR, ""),
           "vespagram_hm": ("Vespagram, high+medium", "#1f4e8c", VESPA_DIR, "_highmed")}
TIERS   = {"high": ("High", "#08519c"), "medium": ("Medium", "#4292c6"), "low": ("Low", "#9ecae1")}
REF, NEW = "roi", "vespagram_hm"      # pair whose shape correlation is reported
CONTROLS = ["sin_m", "cos_m", "sin_2m", "cos_2m", "sin_h", "cos_h", "sin_2h", "cos_2h", "f107"]
LAG_DRIVERS = ["SME", "Hp30", "Newell", "SymH"]
LAG_MAX_H   = 12
SME_BIN     = 100     # nT, occupancy-vs-SME bins
SME_CAP     = 1000
MIN_EXPOSURE = 50     # hours per SME bin before its occupancy is shown


# ── data ────────────────────────────────────────────────────────────────────────

def load(method, kind):
    _, _, folder, suffix = METHODS[method]
    return pickle.load(open(folder / f"lstid_{kind}{suffix}.pkl", "rb"))


def load_tier(tier):
    return pickle.load(open(VESPA_DIR / f"lstid_hours_{tier}.pkl", "rb"))


def common_grid():
    return pl.datetime_range(dt.datetime(2015, 1, 1), dt.datetime(2025, 12, 31, 23),
                             interval="1h", time_zone="UTC", eager=True)


def outcome(grid, hour_records):
    active = {pd.Timestamp(r["ref_time"]).tz_localize("UTC") for r in hour_records}
    return np.array([t in active for t in pd.to_datetime(grid.to_numpy(), utc=True)], int)


def base_design(grid):
    ts = pd.to_datetime(grid.to_numpy(), utc=True)
    m, h = ts.month.to_numpy(), ts.hour.to_numpy()
    df = pd.DataFrame({"time": ts})
    for k, (x, n) in {"m": (m, 12), "h": (h, 24)}.items():
        df[f"sin_{k}"]  = np.sin(2 * np.pi * x / n)
        df[f"cos_{k}"]  = np.cos(2 * np.pi * x / n)
        df[f"sin_2{k}"] = np.sin(4 * np.pi * x / n)
        df[f"cos_2{k}"] = np.cos(4 * np.pi * x / n)
    df["f107"] = _zscore(_rolling_on_grid(_f107_series(), F107_COL, grid, window_hours=1))
    return df


def driver_on_grid(name, grid, window_h=3, lag_h=0, z=True):
    q = grid if lag_h == 0 else grid.dt.offset_by(f"-{int(lag_h * 60)}m")
    v = _rolling_on_grid(_load_driver_series(name), name, q, window_hours=window_h)
    return _zscore(v) if z else v


# ── models ──────────────────────────────────────────────────────────────────────

def _fit(y, X):
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    return res, roc_auc_score(y, res.predict(sm.add_constant(X)))


def regress(base, drivers, y):
    rows = []
    ok_all = np.isfinite(y)
    _, auc0 = _fit(y, base[CONTROLS])
    for name, col in drivers.items():
        ok = np.isfinite(col)
        X  = base.loc[ok, CONTROLS].assign(**{name: col[ok]})
        res, auc = _fit(y[ok], X)
        _, auc_c = _fit(y[ok], base.loc[ok, CONTROLS]) if not ok.all() else (None, auc0)
        lo, hi = res.conf_int().loc[name]
        rows.append({"driver": name, "coef": res.params[name], "ci_lo": lo, "ci_hi": hi,
                     "pvalue": res.pvalues[name], "auc": auc, "auc_controls": auc_c,
                     "dauc": auc - auc_c, "n": int(ok.sum()), "n_pos": int(y[ok].sum())})
    return pd.DataFrame(rows)


def lag_sweep(base, grid, y, names=LAG_DRIVERS):
    rows = []
    for name in names:
        for lag in range(LAG_MAX_H + 1):
            col = driver_on_grid(name, grid, window_h=1, lag_h=lag)
            ok  = np.isfinite(col)
            _, auc   = _fit(y[ok], base.loc[ok, CONTROLS].assign(**{name: col[ok]}))
            _, auc_c = _fit(y[ok], base.loc[ok, CONTROLS])
            rows.append({"driver": name, "lag_h": lag, "dauc": auc - auc_c})
    return pd.DataFrame(rows)


# ── figures ─────────────────────────────────────────────────────────────────────

def _style(ax):
    ax.grid(linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_occurrence(grid, ys, path):
    ts = pd.to_datetime(grid.to_numpy(), utc=True)
    keys = {"UT hour": ts.hour.to_numpy(), "Month": ts.month.to_numpy(), "Year": ts.year.to_numpy()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (xl, k) in zip(axes, keys.items()):
        lv = np.unique(k)
        for m, y in ys.items():
            label, c = METHODS[m][:2]
            occ = [y[k == v].mean() * 100 for v in lv]
            ax.plot(lv, occ, color=c, marker="o", ms=4, lw=2, label=f"{label} ({y.mean()*100:.1f}%)")
        ax.set_xlabel(xl)
        ax.set_ylim(bottom=0)
        _style(ax)
        if xl == "Month":
            ax.set_xticks(range(1, 13), [calendar.month_abbr[i][0] for i in range(1, 13)])
        if xl == "UT hour":
            ax.set_xticks(range(0, 24, 3))
        if xl == "Year":
            ax.set_xticks(lv, [str(v)[2:] for v in lv])
            f107 = clim.load_f107_by_year(list(lv))
            ax2 = ax.twinx()
            ax2.plot(lv, [f107[v] for v in lv], color="#888888", ls="--", lw=1.2, label="F10.7")
            ax2.set_ylabel("F10.7 [sfu]", color="#888888")
            ax2.tick_params(axis="y", colors="#888888")
            ax2.spines["top"].set_visible(False)
            ax2.legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].set_ylabel("Hours with an LSTID [%]")
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    r = {xl: np.corrcoef(*[[ys[m][k == v].mean() for v in np.unique(k)] for m in (REF, NEW)])[0, 1]
         for xl, k in keys.items()}
    fig.suptitle("LSTID occupancy (share of hours with an LSTID present)   "
                 f"shape correlation {METHODS[REF][0]} vs {METHODS[NEW][0]}: " +
                 ", ".join(f"{xl.lower()} r={v:.2f}" for xl, v in r.items()), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return r


def _month_hour(grid, y):
    ts = pd.to_datetime(grid.to_numpy(), utc=True)
    df = pd.DataFrame({"m": ts.month, "h": ts.hour, "y": y})
    return df.pivot_table(index="m", columns="h", values="y", aggfunc="mean").to_numpy() * 100


def fig_month_hour(grid, ys, path, labels=METHODS):
    sza = clim.load_sza_month_hour(list(range(2015, 2026)))
    maps = {m: _month_hour(grid, y) for m, y in ys.items()}
    fig, axes = plt.subplots(1, len(maps), figsize=(6.5 * len(maps), 4.8))
    for ax, (m, H) in zip(np.atleast_1d(axes), maps.items()):
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis")
        ax.contour(np.arange(24), np.arange(12), sza, levels=[90], colors="orange",
                   linewidths=1.8, linestyles="--")
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(12), [calendar.month_abbr[i] for i in range(1, 13)])
        ax.set_xlabel("UT hour")
        ax.set_title(f"{labels[m][0]}", fontsize=11)
        fig.colorbar(im, ax=ax).set_label("Hours with an LSTID [%]")
    if REF in maps and NEW in maps:
        r = np.corrcoef(maps[REF].ravel(), maps[NEW].ravel())[0, 1]
        fig.suptitle(f"Month x UT occupancy (dashed: solar terminator)   pattern correlation "
                     f"{labels[REF][0]} vs {labels[NEW][0]}: r={r:.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_characteristics(trains, path):
    # speed bins uniform in SLOWNESS, 3 vespagram grid steps wide and centred on the
    # grid: the slowness grid is discrete, so equal-width speed bins show a comb
    vc = vespagram.VespaConfig()
    dp = 2 * vc.max_slowness / (vc.n_slowness - 1)
    p_edges = (np.arange(np.ceil(1 / 1500 / dp), np.floor(1 / 250 / dp) + 3, 3) - 0.5) * dp
    specs = [("speed_m_s", "Speed [m/s]", 1, np.sort(1 / p_edges)),
             ("period_s", "Period [min]", 1 / 60, np.arange(20, 210, 7.5)),
             ("peak_amplitude", "Amplitude [TECu]", 1, np.linspace(0, 0.3, 41)),
             ("duration_h", "Train duration [h]", 1, np.arange(0, 12.25, 0.5))]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (k, xl, sc, bins) in zip(axes, specs):
        for m, recs in trains.items():
            label, c = METHODS[m][:2]
            v = np.abs(np.array([r[k] for r in recs], float)) * sc
            v = v[np.isfinite(v)]
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                    lw=2, color=c, label=f"{label}  median {np.median(v):.3g}")
        ax.set_xlabel(xl)
        ax.set_yticks([])
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    axes[2].text(0.98, 0.55, "ROI: |peak| of the front\nVespagram: √2 × RMS of the beam",
                 transform=axes[2].transAxes, ha="right", fontsize=7, color="#666666")
    fig.suptitle("Per-train characteristics  (n = " +
                 ", ".join(f"{METHODS[m][0]} {len(r)}" for m, r in trains.items()) + ")",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _coef_panel(ax, tabs, colors, labels):
    order = tabs[list(tabs)[-1]].sort_values("coef")["driver"].tolist()
    y0 = np.arange(len(order))
    off = np.linspace(-0.18, 0.18, len(tabs))
    for o, (k, t) in zip(off, tabs.items()):
        t = t.set_index("driver").loc[order]
        ax.errorbar(t["coef"], y0 + o, xerr=[t["coef"] - t["ci_lo"], t["ci_hi"] - t["coef"]],
                    fmt="o", ms=5, capsize=3, color=colors[k], label=labels[k])
    ax.axvline(0, color="#999999", lw=1, ls="--")
    ax.set_yticks(y0, order)
    ax.set_xlabel("Adjusted effect (log-odds per 1 SD)")
    _style(ax)
    return order


def _dauc_panel(ax, tabs, order, colors):
    y0 = np.arange(len(order))
    h = 0.8 / len(tabs)
    for i, (k, t) in enumerate(tabs.items()):
        t = t.set_index("driver").loc[order]
        ax.barh(y0 - 0.4 + h * (i + 0.5), t["dauc"], height=h * 0.9, color=colors[k])
    ax.set_yticks(y0, [])
    ax.set_xlabel("Added skill ΔAUC over season/UT/F10.7")
    _style(ax)


def _sme_panel(ax, sme, ys, colors, labels):
    x = np.clip(sme, 0, SME_CAP)
    b = np.round(x / SME_BIN) * SME_BIN
    lv = np.unique(b[np.isfinite(b)])
    lv = [v for v in lv if (b == v).sum() >= MIN_EXPOSURE]
    for k, y in ys.items():
        ax.plot(lv, [y[b == v].mean() * 100 for v in lv], marker="o", ms=4, lw=2,
                color=colors[k], label=labels[k])
    ax.set_xlabel(f"SME, 3 h trailing mean [nT]  (last bin = {SME_CAP}+)")
    ax.set_ylabel("Hours with an LSTID [%]")
    ax.set_ylim(bottom=0)
    _style(ax)
    ax.legend(fontsize=9, frameon=False)
    exp = ax.twinx()
    exp.bar(lv, [(b == v).sum() for v in lv], width=SME_BIN * 0.8, color="#cccccc",
            alpha=0.5, zorder=0)
    exp.set_yscale("log")
    exp.set_ylabel("Hours in bin (grey)", color="#999999")
    exp.tick_params(axis="y", colors="#999999")
    ax.set_zorder(exp.get_zorder() + 1)
    ax.patch.set_visible(False)


def fig_drivers(tabs, sme, ys, lags, path):
    colors = {m: METHODS[m][1] for m in tabs}
    labels = {m: METHODS[m][0] for m in tabs}
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 0.8, 1.2])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    order = _coef_panel(ax1, tabs, colors, labels)
    ax1.legend(fontsize=9, frameon=False, loc="lower right")
    ax1.set_title("Driver effect on LSTID occurrence (95% CI)", fontsize=11)
    _dauc_panel(ax2, tabs, order, colors)
    ax2.set_title("Skill added by the driver", fontsize=11)

    ax3 = fig.add_subplot(gs[0, 2])
    _sme_panel(ax3, sme, ys, colors, labels)
    ax3.set_title("Occupancy vs auroral activity (raw)", fontsize=11)

    ax4 = fig.add_subplot(gs[1, :])
    ls = {"roi": "--", "vespagram": ":", "vespagram_hm": "-"}
    cmap = {"SME": "#2b6cb0", "Hp30": "#2f855a", "Newell": "#b7791f", "SymH": "#805ad5"}
    for m, L in lags.items():
        for d, g in L.groupby("driver"):
            g = g.sort_values("lag_h")
            ax4.plot(g["lag_h"], g["dauc"], ls=ls[m], color=cmap[d], marker="o", ms=3,
                     label=f"{d} ({METHODS[m][0]})")
            best = g.loc[g["dauc"].idxmax()]
            ax4.scatter([best["lag_h"]], [best["dauc"]], s=80, facecolors="none",
                        edgecolors=cmap[d], lw=1.5, zorder=5)
    ax4.set_xlabel("Lag [h]  (driver averaged over the 1 h window ending lag hours before the hour)")
    ax4.set_ylabel("ΔAUC over season/UT/F10.7")
    ax4.set_title("Response delay (circle = peak).  Solid: vespagram high+medium, "
                  "dotted: vespagram all, dashed: ROI", fontsize=11)
    ax4.set_xticks(range(LAG_MAX_H + 1))
    _style(ax4)
    ax4.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tiers(grid, yd, tabs, sme, path):
    """Independent check of the confidence tiers (no audit labels involved): real
    LSTIDs follow auroral activity, so a tier made mostly of false detections
    should show a flat occupancy-vs-SME curve and a small driver effect."""
    ts = pd.to_datetime(grid.to_numpy(), utc=True)
    colors = {k: TIERS[k][1] for k in yd}
    labels = {k: TIERS[k][0] for k in yd}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.2, 1.2]})
    for ax, (xl, k) in zip(axes[:2], {"UT hour": ts.hour.to_numpy(),
                                      "Month": ts.month.to_numpy()}.items()):
        lv = np.unique(k)
        for d, y in yd.items():
            occ = np.array([y[k == v].mean() for v in lv])
            ax.plot(lv, occ / occ.mean(), color=colors[d], marker="o", ms=4, lw=2,
                    label=f"{labels[d]} (mean {y.mean()*100:.1f}% of hours)")
        ax.axhline(1, color="#999999", lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("Occupancy / own mean")
        _style(ax)
    axes[0].set_xticks(range(0, 24, 3))
    axes[1].set_xticks(range(1, 13), [calendar.month_abbr[i][0] for i in range(1, 13)])
    axes[0].legend(fontsize=8, frameon=False)
    x = np.clip(sme, 0, SME_CAP)
    b = np.round(x / SME_BIN) * SME_BIN
    lv = [v for v in np.unique(b[np.isfinite(b)]) if (b == v).sum() >= MIN_EXPOSURE]
    for d, y in yd.items():
        occ = np.array([y[b == v].mean() for v in lv])
        axes[2].plot(lv, occ / y.mean(), marker="o", ms=4, lw=2, color=colors[d])
    axes[2].axhline(1, color="#999999", lw=1)
    axes[2].set_xlabel(f"SME, 3 h trailing mean [nT]  ({SME_CAP}+)")
    axes[2].set_ylabel("Occupancy / own mean")
    _style(axes[2])
    _coef_panel(axes[3], tabs, colors, labels)
    axes[3].set_title("Adjusted driver effect", fontsize=10)
    fig.suptitle("Vespagram LSTIDs by confidence tier (shapes normalised to each "
                 "tier's mean occupancy)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    grid = common_grid()
    base = base_design(grid)
    print(f"grid: {len(grid)} hours")

    trains = {m: load(m, "trains") for m in METHODS}
    ys     = {m: outcome(grid, load(m, "hours")) for m in METHODS}
    yd     = {d: outcome(grid, load_tier(d)) for d in TIERS}
    for m, y in ys.items():
        print(f"  {m:13s} trains={len(trains[m]):6d}  occupancy={y.mean()*100:.1f}%")
    for d, y in yd.items():
        print(f"  tier {d:8s} occupancy={y.mean()*100:.2f}%")
    for m in ("vespagram", "vespagram_hm"):
        both = (ys["roi"] & ys[m]).sum()
        print(f"  ROI hours also in {m}: {both / ys['roi'].sum()*100:.0f}%   "
              f"{m} hours also ROI: {both / ys[m].sum()*100:.0f}%")

    r = fig_occurrence(grid, ys, OUT / "occurrence.png")
    print("  occupancy shape correlation:", {k: round(v, 2) for k, v in r.items()})
    fig_month_hour(grid, ys, OUT / "month_hour.png")
    fig_characteristics(trains, OUT / "characteristics.png")

    print("drivers ...")
    drivers = {n: driver_on_grid(n, grid) for n in DRIVER_COLS}
    tabs = {m: regress(base, drivers, y) for m, y in ys.items()}
    tabd = {d: regress(base, drivers, y) for d, y in yd.items()}
    out = pd.concat([t.assign(catalogue=k) for k, t in {**tabs, **tabd}.items()])
    out.to_csv(OUT / "drivers.csv", index=False)
    print(out.pivot_table(index="driver", columns="catalogue", values=["coef", "dauc"])
             .round(3).to_string())

    sme = driver_on_grid("SME", grid, z=False)
    lags = {m: lag_sweep(base, grid, y) for m, y in ys.items()}
    pd.concat([L.assign(catalogue=m) for m, L in lags.items()]).to_csv(OUT / "lags.csv", index=False)
    fig_drivers(tabs, sme, ys, lags, OUT / "drivers.png")
    fig_tiers(grid, yd, tabd, sme, OUT / "tiers.png")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
