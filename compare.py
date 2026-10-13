"""
compare.py

Compare the ROI-based LSTID catalogue (detect_roi.py -> lstid_rois.pkl) with the
vespagram-based one (vespagram.py -> trains.parquet).

The ROI method reports one record per front, so its fronts are merged into WAVE
TRAINS (intervals closer than TRAIN_GAP). The vespagram catalogue already is one
record per train. Trains from the two methods are matched when they overlap in time.

Outputs (results/comparison/):
  trains.parquet          every train of both methods with its match status
  summary.png             yearly counts, match rates, speed/period agreement
  examples/<category>/    side-by-side figures for sampled days per category:
      agree           both methods see the same train(s), similar speed
      roi_only        a clear ROI train the vespagram does not class as LSTID
      vespa_only      a clear vespagram train with no ROI LSTID
      speed_disagree  matched train whose speeds differ by more than 1.5x
      poleward        a vespagram poleward train (the ROI method has no direction)
"""

import pickle
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import detect_roi
import vespagram

ROOT      = Path(__file__).parent
ROI_PKL   = ROOT / "results/roi/lstid_rois.pkl"
VESPA_PQ  = ROOT / "results/vespagram/trains.parquet"
OUT_DIR   = ROOT / "results/comparison"

TRAIN_GAP     = dt.timedelta(minutes=60)   # merge LSTID intervals closer than this
MATCH_TOL     = dt.timedelta(minutes=15)   # trains match if they overlap within this
SPEED_RATIO   = 1.5                        # matched but |log ratio| above -> disagree
N_PER_CATEGORY = 5


# ── Loading ─────────────────────────────────────────────────────────────────────

def load_roi():
    recs = pickle.load(open(ROI_PKL, "rb"))
    return pl.DataFrame([{k: r[k] for k in ("date", "start_time", "end_time", "ref_time",
                                             "start_lat", "end_lat", "speed_m_s",
                                             "period_s", "mean_amplitude")}
                         for r in recs])


def load_vespa():
    return pl.read_parquet(VESPA_PQ)


# ── Trains and matching ─────────────────────────────────────────────────────────

def build_trains(starts, ends, speeds, periods, method):
    """Merge intervals into trains (gap < TRAIN_GAP). Trains are kept on the
    global timeline so a train crossing midnight is one train; it is assigned
    to the day holding its midpoint."""
    order  = np.argsort(starts)
    trains = []
    for i in order:
        s, e = starts[i], ends[i]
        if trains and s <= trains[-1]["end"] + TRAIN_GAP:
            tr = trains[-1]
            tr["end"] = max(tr["end"], e)
            tr["speeds"].append(speeds[i]); tr["periods"].append(periods[i])
            tr["n"] += 1
        else:
            trains.append({"start": s, "end": e, "speeds": [speeds[i]],
                           "periods": [periods[i]], "n": 1})
    return [{"method":     method,
             "start":      tr["start"],
             "end":        tr["end"],
             "date":       (tr["start"] + (tr["end"] - tr["start"]) / 2).date(),
             "duration_h": (tr["end"] - tr["start"]).total_seconds() / 3600,
             "speed":      float(np.nanmedian(tr["speeds"])),
             "period_min": float(np.nanmedian(tr["periods"])),
             "n_members":  tr["n"]}
            for tr in trains]


def match(a, b):
    """For each train in a, the index of the most-overlapping train in b (or -1)."""
    b_start = np.array([t["start"] for t in b])
    b_end   = np.array([t["end"] for t in b])
    out = []
    for t in a:
        ov = np.minimum(b_end, t["end"] + MATCH_TOL) - np.maximum(b_start, t["start"] - MATCH_TOL)
        ov = np.array([o.total_seconds() for o in ov]) if len(ov) else np.array([])
        out.append(int(np.argmax(ov)) if ov.size and ov.max() > 0 else -1)
    return out


def trains_table(roi, vespa):
    roi_tr = build_trains(roi["start_time"].to_list(), roi["end_time"].to_list(),
                          roi["speed_m_s"].to_numpy(), (roi["period_s"] / 60).to_numpy(), "roi")
    for t in roi_tr:
        t["direction"] = None                 # the ROI speed is unsigned
        t["geometry"]  = None
        t["amplitude"] = None
    ves_tr = [{"method": "vespa", "start": r["t_start"], "end": r["t_end"], "date": r["date"],
               "duration_h": (r["t_end"] - r["t_start"]).total_seconds() / 3600,
               "speed": r["speed_m_s"], "period_min": r["period_min"],
               "n_members": r["n_bands"], "direction": r["direction"],
               "geometry": r["geometry"], "amplitude": r["amplitude"]}
              for r in vespa.iter_rows(named=True)]

    for tr, m, other in [(t, m, ves_tr) for t, m in zip(roi_tr, match(roi_tr, ves_tr))] +                         [(t, m, roi_tr) for t, m in zip(ves_tr, match(ves_tr, roi_tr))]:
        tr["matched"]      = m >= 0
        tr["other_speed"]  = other[m]["speed"] if m >= 0 else np.nan
        tr["other_period"] = other[m]["period_min"] if m >= 0 else np.nan
    return pl.DataFrame(roi_tr + ves_tr, infer_schema_length=None)


# ── Summary figure ──────────────────────────────────────────────────────────────

INK   = "#222222"
MUTED = "#777777"
C_ROI, C_VES = "#2a6fdb", "#d9822b"     # method identity (always direct-labelled)


def summary_figure(tr, path):
    tr = tr.with_columns(pl.col("date").dt.year().alias("year"))
    years = sorted(tr["year"].unique().to_list())

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax = axs[0, 0]
    count = lambda df: [dict(df.group_by("year").len().iter_rows()).get(y, 0) for y in years]
    roi_n = count(tr.filter(pl.col("method") == "roi"))
    eq_n  = count(tr.filter(pl.col("direction") == "equatorward"))
    po_n  = count(tr.filter(pl.col("direction") == "poleward"))
    x = np.array(years)
    ax.bar(x - 0.2, roi_n, width=0.38, color=C_ROI, label="ROI-based")
    ax.bar(x + 0.2, eq_n, width=0.38, color=C_VES, label="vespagram, equatorward")
    ax.bar(x + 0.2, po_n, width=0.38, bottom=eq_n, color="white", edgecolor=C_VES,
           hatch="///", label="vespagram, poleward")
    ax.set_title("LSTID wave trains per year", loc="left")
    ax.legend(frameon=False)

    ax = axs[0, 1]
    for m, c, lbl in [("roi", C_ROI, "ROI trains also found by vespagram"),
                      ("vespa", C_VES, "vespagram trains also found by ROI")]:
        g = (tr.filter(pl.col("method") == m).group_by("year")
               .agg(pl.col("matched").mean()).sort("year"))
        ax.plot(g["year"], g["matched"] * 100, "o-", color=c, lw=2, ms=6, label=lbl)
    ax.set_ylim(0, 100); ax.set_ylabel("%")
    ax.set_title("Agreement per year", loc="left")
    ax.legend(frameon=False)

    mt = tr.filter((pl.col("method") == "roi") & pl.col("matched"))
    for ax, col, ocol, name, lim in [(axs[1, 0], "speed", "other_speed", "speed [m/s]", (250, 1600)),
                                     (axs[1, 1], "period_min", "other_period", "period [min]", (20, 130))]:
        x, y = mt[col].to_numpy(), mt[ocol].to_numpy()
        ax.scatter(x, y, s=10, alpha=0.35, color=INK, lw=0)
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
        ok = np.isfinite(x) & np.isfinite(y)
        r  = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan
        med = np.nanmedian(y / x)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"ROI-based {name}"); ax.set_ylabel(f"vespagram {name}")
        ax.set_title(f"Matched trains: {name.split()[0]}  (n={ok.sum()}, r={r:.2f}, "
                     f"median vespa/ROI={med:.2f})", loc="left")

    for a in axs.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Example selection ───────────────────────────────────────────────────────────

def pick_examples(tr, n=N_PER_CATEGORY, seed=0):
    """Days per category, spread over different years."""
    rng = np.random.default_rng(seed)
    roi, ves = tr.filter(pl.col("method") == "roi"), tr.filter(pl.col("method") == "vespa")

    day_has = lambda df: set(df["date"].to_list())
    unmatched_roi = roi.filter(~pl.col("matched") & (pl.col("duration_h") >= 1) & (pl.col("n_members") >= 3))
    unmatched_ves = ves.filter(~pl.col("matched") & (pl.col("duration_h") >= 1))
    ratio         = (pl.col("other_speed") / pl.col("speed")).log().abs()
    disagree      = roi.filter(pl.col("matched") & (ratio > np.log(SPEED_RATIO)))
    any_unmatched = day_has(tr.filter(~pl.col("matched") & (pl.col("duration_h") >= 0.75)))
    agree         = roi.filter(pl.col("matched") & (ratio <= np.log(1.25))
                               & ~pl.col("date").is_in(list(any_unmatched | day_has(disagree))))

    poleward      = ves.filter(pl.col("direction") == "poleward")
    cats = {"agree": agree, "roi_only": unmatched_roi, "vespa_only": unmatched_ves,
            "speed_disagree": disagree, "poleward": poleward}
    picks = {}
    for name, df in cats.items():
        days  = sorted(day_has(df))
        if not days:
            picks[name] = []
            continue
        years = sorted({d.year for d in days})
        chosen_years = rng.choice(years, size=min(n, len(years)), replace=False)
        picks[name] = sorted(rng.choice([d for d in days if d.year == y])
                             for y in chosen_years)
    return picks


def pick_single_band(tr, seed=0):
    """
    Five days that each contain a single-band vespagram train, one per case,
    for judging whether single-band trains are real:
      confirmed   the ROI method found the same train
      strong      unmatched, amplitude in the top quartile of single-band trains
      typical     unmatched, amplitude near the single-band median
      solar_min   unmatched, typical amplitude, in 2019-2020
      poleward    a poleward single-band train
    Days are drawn from different years where possible.
    """
    rng = np.random.default_rng(seed)
    sb  = (tr.filter((pl.col("method") == "vespa") & (pl.col("geometry") == "single band"))
             .with_columns(pl.col("date").dt.year().alias("year")))
    q25, q50, q75 = (sb["amplitude"].quantile(q) for q in (0.25, 0.5, 0.75))
    typical = pl.col("amplitude").is_between(0.8 * q50, 1.25 * q50)
    eq_um   = (pl.col("direction") == "equatorward") & ~pl.col("matched")
    cases = {
        "confirmed": sb.filter((pl.col("direction") == "equatorward") & pl.col("matched")),
        "strong":    sb.filter(eq_um & (pl.col("amplitude") >= q75)),
        "typical":   sb.filter(eq_um & typical & ~pl.col("year").is_in([2019, 2020])),
        "solar_min": sb.filter(eq_um & typical & pl.col("year").is_in([2019, 2020])),
        "poleward":  sb.filter(pl.col("direction") == "poleward"),
    }
    picks, used_years = {}, set()
    for name, df in cases.items():
        pool = df.filter(~pl.col("year").is_in(list(used_years))) if name != "solar_min" else df
        pool = pool if pool.height else df
        row  = pool.row(int(rng.integers(pool.height)), named=True)
        picks[name] = row["date"]
        used_years.add(row["year"])
    print(f"single-band amplitude quartiles [TECu]: {q25:.3f} / {q50:.3f} / {q75:.3f}")
    return picks


# ── Per-day comparison figure ───────────────────────────────────────────────────

def _hours(ts, day):
    return (ts - dt.datetime.combine(day, dt.time())).total_seconds() / 3600


def comparison_figure(day, category, roi, tr, path):
    rcfg = detect_roi.Config(input_date=day)
    Z, lats, times_sec, _ = detect_roi.load_grid(rcfg)
    Z_ds = detect_roi.preprocess(Z, rcfg)
    ext  = [times_sec[0] / 3600, times_sec[-1] / 3600, lats.min(), lats.max()]

    lo = dt.datetime.combine(day, dt.time()) - dt.timedelta(hours=2)
    hi = lo + dt.timedelta(hours=28)
    r_day = roi.filter((pl.col("ref_time") >= lo) & (pl.col("ref_time") < hi))
    t_day = tr.filter((pl.col("end") >= lo) & (pl.col("start") < hi))

    fig = plt.figure(figsize=(16, 12.5))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.32], hspace=0.28)

    def keogram(ax, title):
        ax.imshow(Z_ds, cmap="RdBu_r", vmin=-0.12, vmax=0.12, aspect="auto",
                  extent=ext, interpolation="nearest")
        ax.set_ylabel("latitude [°N]")
        ax.set_title(title, loc="left", fontsize=11)

    def label(ax, x, y, text, muted=False):
        ax.text(x, y, text, fontsize=8, ha="center", va="top",
                color=MUTED if muted else INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # (a) ROI-based
    ax = fig.add_subplot(gs[0])
    keogram(ax, f"(a) ROI-based LSTIDs: {r_day.height} fronts  "
                "(RANSAC line per ROI; solid = +dTEC, dashed = −dTEC)")
    for r in r_day.iter_rows(named=True):
        ax.plot([_hours(r["start_time"], day), _hours(r["end_time"], day)],
                [r["start_lat"], r["end_lat"]], color="k", lw=1.3,
                ls="-" if r["mean_amplitude"] > 0 else "--")
    for t in t_day.filter(pl.col("method") == "roi").iter_rows(named=True):
        label(ax, (_hours(t["start"], day) + _hours(t["end"], day)) / 2, lats.max() - 0.3,
              f"{t['speed']:.0f} m/s · {t['period_min']:.0f} min")

    # (b) vespagram-based — rerun detection for this day to draw the band fronts
    vcfg = vespagram.VespaConfig(input_date=day)
    _, vlats, _, _, dets, vtrains, members = vespagram.detect(vcfg)
    ax = fig.add_subplot(gs[1], sharex=fig.axes[0])
    keogram(ax, f"(b) vespagram LSTID trains: {len(vtrains)}  "
                "(↓ equatorward, ↑ poleward; solid = crest, dashed = trough)")
    for vt, idx in zip(vtrains, members):
        vespagram._draw_train(ax, vt, [dets[i] for i in idx], vlats, vcfg)

    # (c) train timeline
    ax = fig.add_subplot(gs[2], sharex=fig.axes[0])
    for yi, m, name in [(1, "roi", "ROI"), (0, "vespa", "vespagram")]:
        for t in t_day.filter(pl.col("method") == m).iter_rows(named=True):
            x0, x1 = _hours(t["start"], day), _hours(t["end"], day)
            ax.barh(yi, x1 - x0, left=x0, height=0.6,
                    color=INK if t["matched"] else "white", edgecolor=INK,
                    hatch=None if t["matched"] else "///", lw=1)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["ROI", "vespagram"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("(c) wave trains — filled = matched by the other method, hatched = unmatched",
                 loc="left", fontsize=11)

    for a in fig.axes:
        a.set_xlim(0, 24); a.set_xticks(range(0, 25, 2))
    fig.axes[-1].set_xlabel("time [h UT]")
    fig.suptitle(f"{day}  —  category: {category}", x=0.125, ha="left", y=0.93, fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", nargs="*", help="also draw these YYYY-MM-DD days (category 'manual')")
    ap.add_argument("-n", type=int, default=N_PER_CATEGORY, help="examples per category")
    ap.add_argument("--categories", nargs="*",
                    help="only draw these example categories (e.g. single_band); default: all")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roi, vespa = load_roi(), load_vespa()
    tr = trains_table(roi, vespa)
    tr.write_parquet(OUT_DIR / "trains.parquet")
    summary_figure(tr, OUT_DIR / "summary.png")

    for m in ("roi", "vespa"):
        s = tr.filter(pl.col("method") == m)
        print(f"{m:6s} trains: {s.height:5d}   matched: {s['matched'].mean() * 100:5.1f}%   "
              f"median duration {s['duration_h'].median():.1f} h")
    ves = tr.filter(pl.col("method") == "vespa")
    print(f"vespagram trains by direction: "
          f"{dict(ves.group_by('direction').len().iter_rows())}")

    picks = pick_examples(tr, n=args.n)
    sb    = pick_single_band(tr)
    picks["single_band"] = list(sb.values())
    if args.days:
        picks["manual"] = [dt.date.fromisoformat(d) for d in args.days]
    if args.categories:
        picks = {k: v for k, v in picks.items() if k in args.categories}
    case_of = {d: c for c, d in sb.items()}
    for cat, days in picks.items():
        (OUT_DIR / "examples" / cat).mkdir(parents=True, exist_ok=True)
        for d in days:
            label = f"{cat} / {case_of[d]}" if cat == "single_band" else cat
            name  = f"{case_of[d]}_{d}" if cat == "single_band" else str(d)
            comparison_figure(d, label, roi, tr, OUT_DIR / "examples" / cat / f"{name}_compare.png")
            vespagram.run(vespagram.VespaConfig(input_date=d), do_plot=True)
        print(f"{cat:15s} " + ", ".join(f"{case_of[d]}={d}" if cat == "single_band" else str(d)
                                         for d in days))


if __name__ == "__main__":
    main()
