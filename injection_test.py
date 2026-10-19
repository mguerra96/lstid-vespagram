"""
injection_test.py

Injection-recovery test of the vespagram detector: wave packets of known speed,
period, amplitude and latitudinal extent are added to real quiet-day keograms
(band-passed, as the detector sees them) and run through vespagram.analyse.
It answers what no reference catalogue can: how weak a wave can be and still be
found, how accurate the catalogued speed is, and whether the speed interval
(speed_lo_m_s .. speed_hi_m_s) contains the true speed as often as it should.

Quiet days: no catalogued train overlapping the day, daily-mean Hp30 < 1.5,
spread evenly over the years. One packet per run of the detector.

Limits: single plane-wave packets at constant speed; periods inside the passband;
quiet-day backgrounds only; the packet is added after the arc-level detrending,
so the detrending filter is not tested.

  python injection_test.py --days 132 --rep 12 --workers 12    run (about 15 min) and summarise
  python injection_test.py --summary                           summarise the saved results
Output: results/vespagram/injection/injection_results.parquet, summary.txt
"""

import io
import sys
import time
import argparse
import itertools
import datetime as dt
from pathlib import Path
from dataclasses import replace
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl

import vespagram as v
from paths import FEATURES

ROOT     = Path(__file__).parent
OUT      = ROOT / "results/vespagram/injection"

SPEEDS  = [350, 450, 600, 800, 1100, 1400]        # m/s, equatorward
PERIODS = [40, 55, 70]                            # min
AMPS    = [0.01, 0.015, 0.02, 0.03, 0.05, 0.1]    # TECu, peak, before the band-pass
EXTENTS = {"full": (37.5, 57.5), "south": (37.5, 50.0)}
WIDTH   = 1.5                                     # Gaussian envelope half-width [periods]
HP30_MAX = 1.5


def quiet_days(n, seed=0):
    """n days without any catalogued train and with a quiet daily Hp30, spread over the years."""
    tr = pl.read_parquet(ROOT / "results/vespagram/trains.parquet")
    busy = set()
    for s, e in zip(tr["t_start"], tr["t_end"]):
        d = s.date()
        while d <= e.date():
            busy.add(d)
            d += dt.timedelta(days=1)
    hp = (pl.read_parquet(FEATURES, columns=["time", "hp30"])
            .with_columns(pl.col("time").dt.date().alias("d"))
            .group_by("d").agg(pl.col("hp30").mean()).filter(pl.col("hp30") < HP30_MAX))
    ok = sorted(d for d in hp["d"] if d not in busy and dt.date(2015, 1, 2) <= d <= dt.date(2025, 12, 30))
    by_year = {}
    for d in ok:
        by_year.setdefault(d.year, []).append(d)
    rng, per = np.random.default_rng(seed), max(1, n // len(by_year))
    return sorted(d for ds in by_year.values() for d in rng.choice(ds, size=min(per, len(ds)), replace=False))


def run_day(job):
    """Inject each trial (speed, period, amp, extent, centre time) into one day's keogram."""
    day, trials = job
    cfg = replace(v.VespaConfig(), input_date=day)
    try:
        X, lats, t = v.load_and_prepare(cfg)
    except Exception as e:
        return [], f"{day}: {type(e).__name__}: {e}"
    dt_s = t[1] - t[0]
    dist = v.meridian_distance(lats, 47.5)
    _, _, base, _ = v.analyse(X, lats, t, cfg)
    n_base = sum(1 for b in base if b["t_end_s"] > 0 and b["t_start_s"] < v.SECONDS_PER_DAY)
    out = []
    for speed, period, amp, ext, tc in trials:
        T, w = period * 60.0, WIDTH * period * 60.0
        lo, hi = EXTENTS[ext]
        tau = t[None, :] + dist[:, None] / speed          # north (dist > 0) sees the front earlier
        Z = amp * np.sin(2 * np.pi * (tau - tc) / T) * np.exp(-((tau - tc) / w) ** 2)
        Z[(lats < lo - 1e-6) | (lats > hi + 1e-6)] = 0.0
        _, _, trains, _ = v.analyse(X + v.bandpass(Z, dt_s, cfg), lats, t, cfg)
        # the packet is above 1/e of its peak within tc +- w
        hits = [r for r in trains if r["t_end_s"] > tc - w and r["t_start_s"] < tc + w]
        rec = dict(day=day, speed=speed, period=period, amp=amp, extent=ext, tc=tc,
                   n_base=n_base, n_hits=len(hits), detected=bool(hits))
        if hits:
            ov = [min(r["t_end_s"], tc + w) - max(r["t_start_s"], tc - w) for r in hits]
            r = hits[int(np.argmax(ov))]
            rec.update(r_speed=r["speed_m_s"], r_lo=r["speed_lo_m_s"], r_hi=r["speed_hi_m_s"],
                       r_period=r["period_s"] / 60, r_amp=r["amplitude"], r_semb=r["semblance"],
                       r_cycles=r["n_cycles"], r_score=r["confidence_score"], r_conf=r["confidence"],
                       r_cover=r["coverage"], r_start=r["t_start_s"], r_end=r["t_end_s"])
        out.append(rec)
    return out, None


def run(n_days, n_rep, workers):
    days = quiet_days(n_days)
    rng = np.random.default_rng(1)
    cells = list(itertools.product(SPEEDS, PERIODS, AMPS, EXTENTS)) * n_rep
    rng.shuffle(cells)
    per_day = {d: [] for d in days}
    for k, (sp, T, a, ext) in enumerate(cells):
        per_day[days[k % len(days)]].append((sp, T, a, ext, float(rng.uniform(4 * 3600, 20 * 3600))))
    print(f"{len(cells)} injections into {len(days)} quiet days")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(run_day, j) for j in per_day.items()]):
            r, err = f.result()
            rows += r
            if err:
                print("  failed:", err)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(OUT / "injection_results.parquet")
    print(f"done in {time.time() - t0:.0f} s -> {OUT / 'injection_results.parquet'}")


# ── summary ─────────────────────────────────────────────────────────────────────

def _pivot(df, val, cols="speed"):
    rv, cv = sorted(df["amp"].unique()), sorted(df[cols].unique())
    print(f"{'amp / ' + cols:>14s} " + " ".join(f"{c:>6}" for c in cv))
    for r in rv:
        print(f"{r:>14} " + " ".join(f"{val(df.filter((pl.col('amp') == r) & (pl.col(cols) == c))):6.2f}" for c in cv))


def _ratio(g):
    r = (g["r_speed"] / g["speed"]).to_numpy()
    return (f"n={g.height:4d}  median {np.median(r):.3f}  16-84% {np.percentile(r, 16):.3f}..{np.percentile(r, 84):.3f}"
            f"  |error| > 20%: {np.mean(np.abs(r - 1) > 0.2):.2f}")


def summarise():
    d = pl.read_parquet(OUT / "injection_results.parquet")
    det = d.filter(pl.col("detected"))
    full, south = d.filter(pl.col("extent") == "full"), d.filter(pl.col("extent") == "south")
    fd = full.filter(pl.col("detected"))
    hm = lambda g: (g["detected"] & g["r_conf"].is_in(["high", "medium"])).fill_null(False).mean()
    n_rep = d.height // (len(SPEEDS) * len(PERIODS) * len(AMPS) * len(EXTENTS))
    print("INJECTION-RECOVERY TEST OF THE VESPAGRAM DETECTOR\n"
          "\nWhat was done"
          "\n  A wave of known speed, period and amplitude is added to the keogram of a real quiet day,"
          "\n  and the detector is run exactly as for the catalogue. Comparing what the detector reports"
          "\n  with what was put in shows how weak a wave can be and still be found, and how accurate"
          "\n  the catalogued speed, period and amplitude are."
          f"\n  Wave: an equatorward sine packet under a Gaussian envelope ({WIDTH} periods half-width), one per run,"
          "\n    centred at a random time between 04 and 20 UT."
          f"\n  Speeds {SPEEDS} m/s; periods {PERIODS} min; peak amplitudes {AMPS} TECu;"
          f"\n    over the full array (37.5-57.5 N) or confined to the south (37.5-50 N); {n_rep} waves per combination."
          f"\n  Quiet days: no catalogued train and daily-mean Hp30 < {HP30_MAX}, spread evenly over 2015-2025."
          "\n  A wave counts as detected when a catalogued train overlaps it in time."
          "\n\nHow to read the numbers"
          "\n  Section 1: fraction of the injected waves that were detected (1.00 = all of them)."
          "\n  'Recovered / true': catalogued value divided by injected value (1.000 = exact, 0.95 = 5% too low)."
          "\n  '16-84%': the range that holds the central 68% of the cases, i.e. the typical scatter."
          "\n  '|error| > 20%': fraction of waves whose catalogued speed is off by more than 20%."
          "\n\nLimits"
          "\n  Single plane waves at constant speed, on quiet days only. The wave is added after the"
          "\n  arc-level detrending, so the loss of long periods in that filter is not tested here.\n")
    print(f"{d.height} injections into {d['day'].n_unique()} quiet days; trains on those days without "
          f"any injection: {d.group_by('day').agg(pl.col('n_base').first())['n_base'].sum()}")

    print("\n1. Share of injected waves detected (any confidence), full latitudinal extent: amplitude [TECu] x speed [m/s]")
    _pivot(full, lambda g: g["detected"].mean())
    print("\n   detected with high or medium confidence")
    _pivot(full, hm)
    print("\n   detected (any confidence): amplitude x period [min]")
    _pivot(full, lambda g: g["detected"].mean(), cols="period")
    print("\n   detected (any confidence), waves confined to 37.5-50 N")
    _pivot(south, lambda g: g["detected"].mean())

    print("\n2. Recovered / true speed of the detected full-extent waves")
    for c in ("high", "medium", "low"):
        print(f"   {c:20s} {_ratio(fd.filter(pl.col('r_conf') == c))}")
    for c, nm in ((True, "coverage >= 2/3"), (False, "coverage < 2/3")):
        print(f"   {nm:20s} {_ratio(fd.filter((pl.col('r_cover') >= 2 / 3) == c))}")
    for sp in SPEEDS:
        print(f"   {sp:5d} m/s, amp>=0.03 {_ratio(fd.filter((pl.col('speed') == sp) & (pl.col('amp') >= 0.03)))}")

    print("\n3. Share of detected waves whose true speed lies inside [speed_lo_m_s, speed_hi_m_s] (a 16-84% range: expect 0.68)")
    for nm, g in (("all", det), ("high", det.filter(pl.col("r_conf") == "high")),
                  ("medium", det.filter(pl.col("r_conf") == "medium")), ("low", det.filter(pl.col("r_conf") == "low")),
                  ("coverage >= 2/3", det.filter(pl.col("r_cover") >= 2 / 3)),
                  ("coverage < 2/3", det.filter(pl.col("r_cover") < 2 / 3))):
        inside = ((g["speed"] >= g["r_lo"]) & (g["speed"] <= g["r_hi"])).mean()
        half = ((g["r_hi"] - g["r_lo"]) / g["r_speed"] / 2).median()
        err = (g["r_speed"] / g["speed"] - 1).abs().median()
        print(f"   {nm:20s} n={g.height:4d}  inside {inside:.2f}   median half-width {half * 100:4.1f}%   median |error| {err * 100:4.1f}%")

    print("\n4. Period and amplitude of the detected full-extent waves (recovered / injected, median)")
    for T in PERIODS:
        g = fd.filter(pl.col("period") == T)
        gs = g.filter(pl.col("amp") >= 0.05)
        print(f"   T = {T} min: period {(g['r_period'] / g['period']).median():.3f}   "
              f"amplitude, waves >= 0.05 TECu: {(gs['r_amp'] / gs['amp']).median():.2f}")

    print("\n5. Confidence of the detected full-extent waves by injected amplitude")
    for a in AMPS:
        g = fd.filter(pl.col("amp") == a)
        print(f"   {a:5.3f} TECu: n={g.height:4d}  " + "  ".join(f"{c} {float((g['r_conf'] == c).mean()):.2f}" for c in ("high", "medium", "low"))
              + f"   median score {g['r_score'].median():.2f}")

    s = det.filter((pl.col("extent") == "south") & (pl.col("amp") >= 0.03))
    g = fd.filter(pl.col("amp") >= 0.03)
    print(f"\n6. Array coverage, amp >= 0.03 TECu: full-extent waves median {g['r_cover'].median():.2f}; "
          f"waves confined to 37.5-50 N: n={s.height}, median {s['r_cover'].median():.2f}, "
          f"high confidence {float((s['r_conf'] == 'high').mean()):.2f}, speed {_ratio(s)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=132, help="quiet days to use")
    ap.add_argument("--rep", type=int, default=12, help="repetitions of the parameter grid")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--summary", action="store_true", help="only summarise the saved results")
    args = ap.parse_args()
    if not args.summary:
        run(args.days, args.rep, args.workers)
    buf = io.StringIO()
    with redirect_stdout(buf):
        summarise()
    print(buf.getvalue())
    (OUT / "summary.txt").write_text(buf.getvalue(), encoding="utf-8")
