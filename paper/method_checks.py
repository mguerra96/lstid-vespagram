"""
method_checks.py

Numbers quoted in Methods and Usage Notes that describe the method rather than
the catalogue:

  passband    response of the processing chain to a sinusoid in time (arc
              detrending, 10-min mean, decimation, band-pass)
  resolution  width of the semblance peak of a synthetic plane wave on a 10 deg
              sub-array
  density     does the number of observations change the number of trains?
              Days that lose a constellation against their neighbours, and
              occurrence at the same Hp30 year by year
  empty       share of keogram cells without any observation (--empty-cells;
              reads the daily TEC files, about 0.9 TB; per-day values are kept
              in results/vespagram/empty_cells.csv and a rerun resumes from it)

Results are merged into paper/numbers.json under "method_checks".

  python paper/method_checks.py                  passband, resolution, density
  python paper/method_checks.py --empty-cells    also the empty cells, all days
  python paper/method_checks.py --tec-file paper/external/tec_2024-08-12.parquet
                                                 also the empty cells of the given days only
                                                 (the paper quotes 2024-08-12 and 2017-01-10,
                                                 the day with the fewest observations, from
                                                 TEC_DIR/tec_2017-01-10.parquet)
"""

import sys
import json
import argparse
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from scipy.signal import savgol_coeffs, freqz, butter, sosfreqz

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
import vespagram as v
from paths import FEATURES, TEC_DIR

NUMBERS  = Path(__file__).resolve().parent / "numbers.json"
COVERAGE = ROOT / "results/vespagram/daily_coverage.csv"
EMPTY_CSV = ROOT / "results/vespagram/empty_cells.csv"

# pykeo settings that are not in VespaConfig
TEC_STEP_S  = 30      # sampling of the TEC arcs and of the keogram [s]
SG_WINDOW   = 181     # Savitzky-Golay detrending window [samples]
SG_ORDER    = 3
MEAN_WINDOW = 20      # 10-min moving mean [samples]
KEO_LAT     = (37.5, 57.5, 0.05)


# ── passband ────────────────────────────────────────────────────────────────────

def passband(cfg):
    """Amplitude response of the whole chain to a sinusoid of each period."""
    per = np.arange(10, 241, 1.0)                             # minutes
    f   = 1 / (per * 60)
    w   = 2 * np.pi * f * TEC_STEP_S
    _, h_sg = freqz(savgol_coeffs(SG_WINDOW, SG_ORDER), worN=w)
    h_det = np.abs(1 - h_sg * np.exp(1j * w * (SG_WINDOW // 2)))   # centred filter
    _, h_mm = freqz(np.ones(MEAN_WINDOW) / MEAN_WINDOW, worN=w)
    n_dec = int(round(cfg.time_step_s / TEC_STEP_S))
    _, h_dec = freqz(np.ones(n_dec) / n_dec, worN=w)
    sos = butter(4, [1 / cfg.band_max_period_s, 1 / cfg.band_min_period_s],
                 btype="band", fs=1 / cfg.time_step_s, output="sos")
    _, h_bp = sosfreqz(sos, worN=2 * np.pi * f * cfg.time_step_s)
    h = h_det * np.abs(h_mm) * np.abs(h_dec) * np.abs(h_bp) ** 2   # band-pass runs twice
    half = per[h >= h.max() / np.sqrt(2)]
    return {"peak_period_min": float(per[np.argmax(h)]), "peak_gain": round(float(h.max()), 2),
            "half_power_band_min": [float(half.min()), float(half.max())],
            "gain_at_period_min": {int(p): round(float(h[per == p][0]), 2)
                                   for p in (30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180)}}


# ── slowness resolution ─────────────────────────────────────────────────────────

def resolution(cfg, period_min=55.0):
    """Width at half maximum of the semblance peak of a noise-free plane wave on a
    10-degree sub-array (the slowness range runs through infinite speed so that the
    peak is not cut)."""
    lats = np.arange(57.5, 37.5 - 1e-9, -cfg.lat_step_deg)
    t    = np.arange(0, 24 * 3600, cfg.time_step_s, dtype=float)
    tc, T = 12 * 3600.0, period_min * 60
    p_fine = np.linspace(-1 / 300, 1 / 120, 4000)
    lo, hi = 42.5, 52.5
    rows = (lats >= lo - 1e-6) & (lats <= hi + 1e-6)
    dist = v.meridian_distance(lats[rows], 0.5 * (lo + hi))
    out = {"period_min": period_min}
    for speed in (600, 1200):
        tau = t[None, :] + v.meridian_distance(lats, 47.5)[:, None] / speed
        Z = np.sin(2 * np.pi * (tau - tc) / T) * np.exp(-((tau - tc) / (1.5 * T)) ** 2)
        X = v.bandpass(Z, cfg.time_step_s, cfg)
        S, _ = v.slant_stack(X[rows], dist, p_fine, cfg.time_step_s, cfg)
        s = S[:, int(np.argmin(np.abs(t - tc)))]
        k = int(np.argmax(s))
        a = b = k
        while a > 0 and s[a - 1] >= 0.5 * s[k]:
            a -= 1
        while b < len(s) - 1 and s[b + 1] >= 0.5 * s[k]:
            b += 1
        out[f"{speed}_m_s_subarray_10deg"] = {
            "peak_speed_m_s": round(1 / p_fine[k]),
            "fwhm_s_per_km": round(float(p_fine[b] - p_fine[a]) * 1e3, 2),
            # None: the half maximum reaches infinite speed (p <= 0)
            "speed_range_m_s": [round(1 / p_fine[b]), None if p_fine[a] <= 0 else round(1 / p_fine[a])]}
    return out


# ── observation density ─────────────────────────────────────────────────────────

def density(n_boot=2000, seed=0):
    """
    Natural experiment: days on which a constellation is missing (observations
    below 88% of the +-10-day median with the stations intact) against their
    neighbours within +-3 days. And high+medium train hours per day at the same
    Hp30, year by year.
    """
    tr = pl.read_parquet(ROOT / "results/vespagram/trains.parquet").with_columns(
        (pl.col("t_start") + (pl.col("t_end") - pl.col("t_start")) / 2).dt.date().alias("d"),
        ((pl.col("t_end") - pl.col("t_start")).dt.total_minutes() / 60).alias("dur_h"))
    cov = pl.read_csv(COVERAGE, try_parse_dates=True).sort("date")
    days = cov["date"].to_list()
    idx = {d: i for i, d in enumerate(days)}
    n_all, n_hm, h_hm = np.zeros(len(days)), np.zeros(len(days)), np.zeros(len(days))
    for d, c, du in zip(tr["d"], tr["confidence"], tr["dur_h"]):
        if d in idx:
            n_all[idx[d]] += 1
            if c != "low":
                n_hm[idx[d]] += 1
                h_hm[idx[d]] += du
    ff = pl.read_parquet(FEATURES, columns=["time", "hp30", "f107"]).with_columns(pl.col("time").dt.date().alias("d"))
    hp = dict(ff.group_by("d").agg(pl.col("hp30").mean()).iter_rows())
    hp = np.array([hp.get(d, np.nan) for d in days])

    def ratio_to_neighbours(x, k=10):
        return np.array([x[i] / np.median(np.r_[x[max(0, i - k):i], x[i + 1:i + 1 + k]]) for i in range(len(x))])
    r_obs = ratio_to_neighbours(cov["n_observations"].to_numpy().astype(float))
    r_st  = ratio_to_neighbours(cov["n_stations"].to_numpy().astype(float))
    drop  = (r_obs < 0.88) & (r_st > 0.97)
    any_drop = (r_obs < 0.93) | (r_st < 0.95)
    rows = []
    for i in np.nonzero(drop)[0]:
        nb = [j for j in range(max(0, i - 3), min(len(days), i + 4)) if j != i and not any_drop[j]]
        if len(nb) >= 3:
            rows.append([n_all[i], n_all[nb].mean(), n_hm[i], n_hm[nb].mean(), hp[i] - np.nanmean(hp[nb]), r_obs[i]])
    a = np.array(rows)
    rng = np.random.default_rng(seed)
    out = {"dropout_days": len(a), "dropout_days_share_pct": round(100 * len(a) / len(days), 1),
           "observations_relative_to_neighbours": round(float(np.median(a[:, 5])), 2),
           "hp30_difference": round(float(np.nanmean(a[:, 4])), 2)}
    for nm, k in (("all_trains", 0), ("high_medium_trains", 2)):
        change = lambda b: (b[:, k].mean() / b[:, k + 1].mean() - 1) * 100
        bs = [change(a[rng.integers(0, len(a), len(a))]) for _ in range(n_boot)]
        out[f"change_{nm}_pct"] = round(float(change(a)), 1)
        out[f"change_{nm}_ci95_pct"] = [round(float(q), 1) for q in np.percentile(bs, [2.5, 97.5])]

    yrs = np.array([d.year for d in days])
    f107 = dict(ff.with_columns(pl.col("time").dt.year().alias("y")).group_by("y").agg(pl.col("f107").mean()).iter_rows())
    sel = lambda y: (yrs == y) & (hp >= 1) & (hp < 2)
    out["hm_hours_per_day_at_hp30_1_2"] = {int(y): round(float(h_hm[sel(y)].mean()), 2) for y in range(2015, 2026)}
    out["mean_f107"] = {int(y): round(float(f107[y])) for y in range(2015, 2026)}
    out["median_observations_per_day_1e6"] = {
        int(y): round(float(np.median(cov["n_observations"].to_numpy()[yrs == y])) / 1e6, 1) for y in range(2015, 2026)}
    return out


# ── empty keogram cells ─────────────────────────────────────────────────────────

def _empty_share(path):
    """Share of the cells of one day's keogram grid that received no observation."""
    lo, hi, step = KEO_LAT
    n_lat, n_t = int(round((hi - lo) / step)) + 1, v.SECONDS_PER_DAY // TEC_STEP_S
    try:
        df = (pl.scan_parquet(path)
                .filter((pl.col("ele") >= 20) & pl.col("lat_ipp").is_between(lo, hi) & pl.col("vtec_detrended").is_finite())
                .select(((pl.col("lat_ipp") - lo) / step).ceil().clip(0, n_lat - 1).cast(pl.Int64).alias("li"),
                        (pl.col("epoch").dt.hour().cast(pl.Int64) * 3600 + pl.col("epoch").dt.minute().cast(pl.Int64) * 60
                         + pl.col("epoch").dt.second().cast(pl.Int64)).alias("sod"))
                .select((pl.col("li") * n_t + pl.col("sod") // TEC_STEP_S).n_unique()).collect())
    except Exception as e:
        return path.stem[4:], None, f"{type(e).__name__}: {e}"
    return path.stem[4:], 1 - df.item() / (n_lat * n_t), None


def empty_detail(path, dec_lat=5, dec_t=4):
    """Empty cells of one day: overall, south of 40 N, longest gap along latitude, and
    cells of the detector's decimated grid (dec_lat x dec_t cells) without any observation."""
    lo, hi, step = KEO_LAT
    n_lat, n_t = int(round((hi - lo) / step)) + 1, v.SECONDS_PER_DAY // TEC_STEP_S
    df = (pl.scan_parquet(path)
            .filter((pl.col("ele") >= 20) & pl.col("lat_ipp").is_between(lo, hi) & pl.col("vtec_detrended").is_finite())
            .select(((pl.col("lat_ipp") - lo) / step).ceil().clip(0, n_lat - 1).cast(pl.Int64).alias("li"),
                    (pl.col("epoch").dt.hour().cast(pl.Int64) * 3600 + pl.col("epoch").dt.minute().cast(pl.Int64) * 60
                     + pl.col("epoch").dt.second().cast(pl.Int64)).alias("sod"))
            .unique().collect())
    empty = np.ones((n_lat, n_t), bool)
    empty[df["li"].to_numpy(), df["sod"].to_numpy() // TEC_STEP_S] = False
    lat = lo + step * np.arange(n_lat)
    gap = 0
    for col in empty.T:                                   # longest run of empty cells along latitude
        run = np.diff(np.flatnonzero(np.r_[True, ~col, True])) - 1
        gap = max(gap, int(run.max()))
    n = (n_lat - 1) // dec_lat * dec_lat
    dec = empty[:n].reshape(n // dec_lat, dec_lat, n_t // dec_t, dec_t).all(axis=(1, 3))
    return {"empty_pct": round(100 * float(empty.mean()), 2),
            "empty_pct_south_of_40N": round(100 * float(empty[lat < 40].mean()), 2),
            "longest_latitude_gap_deg": round(gap * step, 2),
            "decimated_cells_empty_pct": round(100 * float(dec.mean()), 2)}


def empty_cells(workers):
    """Per-day shares are appended to EMPTY_CSV as they arrive, so an interrupted run resumes."""
    if not TEC_DIR.exists():
        raise FileNotFoundError(f"TEC files not found: {TEC_DIR}")
    done = {}
    if EMPTY_CSV.exists():
        done = dict(pl.read_csv(EMPTY_CSV, schema_overrides={"date": pl.String}).iter_rows())
    else:
        EMPTY_CSV.write_text("date,empty_share\n", encoding="utf-8")
    files = [f for f in sorted(TEC_DIR.glob("tec_*.parquet")) if f.stem[4:] not in done]
    print(f"empty cells: {len(done)} days already done, {len(files)} to do")
    with ProcessPoolExecutor(max_workers=workers) as ex, open(EMPTY_CSV, "a", encoding="utf-8") as fh:
        for d, s, err in ex.map(_empty_share, files, chunksize=4):
            if err:
                print("  failed:", d, err)
                continue
            done[d] = s
            fh.write(f"{d},{s:.6f}\n")
            fh.flush()
    days = np.array([dt.date.fromisoformat(d) for d in sorted(done)])
    share = np.array([done[d] for d in sorted(done)]) * 100
    yrs = np.array([d.year for d in days])
    return {"days": len(share), "median_pct": round(float(np.median(share)), 1),
            "p5_p95_pct": [round(float(q), 1) for q in np.percentile(share, [5, 95])],
            "max_pct": round(float(share.max()), 1),
            "median_pct_by_year": {int(y): round(float(np.median(share[yrs == y])), 1) for y in sorted(set(yrs))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--empty-cells", action="store_true", help="also count the empty keogram cells (reads the TEC files)")
    ap.add_argument("--tec-file", type=Path, nargs="+", default=[],
                    help="count the empty keogram cells of these daily TEC files only")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = v.VespaConfig()
    N = json.loads(NUMBERS.read_text(encoding="utf-8")) if NUMBERS.exists() else {}
    out = N.get("method_checks", {})
    out.update(passband=passband(cfg), resolution=resolution(cfg), density=density())
    out.pop("overlap", None)                     # trains no longer overlap by construction
    if args.empty_cells:
        out["empty_cells"] = empty_cells(args.workers)
    out.pop("empty_cells_pct_by_day", None)
    for f in args.tec_file:
        out.setdefault("empty_cells_by_day", {})[f.stem[4:]] = empty_detail(f)
    N["method_checks"] = out
    NUMBERS.write_text(json.dumps(N, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
