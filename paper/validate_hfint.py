"""
paper/validate_hfint.py

Compare the vespagram catalogue with the HF-INT ionosonde LSTID catalogue
(Segarra et al. 2024, CORA doi:10.34810/data1383; expert-revised HF-INT events,
10 European Digisondes), unzipped to paper/external/hfint/. 2015-2025 overlap.

An HF-INT event is [start, start + duration]. Vespagram trains are matched by time
overlap within TOL (as for the ROI comparison). HF-INT onsets lag the wave (the
event must be seen on several sensors first), so a second matching allows the
train to start up to ONSET_LAG before the HF-INT onset.

Chance level: the same matching with the vespagram trains shifted by whole days
(SHIFTS), which keeps their UT and (roughly) seasonal distribution but breaks the
day-to-day link with the drivers.

Results are merged into paper/numbers.json under "hfint".

  python paper/validate_hfint.py
"""

import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from paper_numbers import overlap_match

HFINT = Path(__file__).resolve().parent / "external/hfint"
NUMBERS = Path(__file__).resolve().parent / "numbers.json"
YEARS = (2015, 2025)
SOUTH = (150, 220)                  # azimuth range of auroral (southward) events, as Segarra et al. 2025
TOL = dt.timedelta(minutes=15)
ONSET_LAG = dt.timedelta(minutes=60)
SHIFTS = [d for k in range(3, 33, 3) for d in (k, -k)]   # +-3..30 days, 20 shifts
TIERS = {"high": ["high"], "medium": ["medium"], "low": ["low"],
         "high_medium": ["high", "medium"], "all": ["high", "medium", "low"]}


def load_hfint():
    rows = []
    for f in sorted(HFINT.glob("Catalogue_LSTID_*.txt")):
        for line in f.read_text(encoding="latin-1").splitlines()[2:]:
            p = line.split()
            if len(p) < 11:
                continue
            hh, mm = p[3].replace(";", ":").split(":")    # one entry has a typo '17;10'
            start = dt.datetime(int(p[0]), int(p[1]), int(p[2]), int(hh), int(mm))
            rows.append({"start": start, "end": start + dt.timedelta(hours=float(p[4])),
                         "duration_h": float(p[4]), "period_min": float(p[5]), "amp_mhz": float(p[6]),
                         "sec_pct": float(p[7]), "vel": float(p[8]), "azi": float(p[9]), "qi": float(p[10])})
    df = pl.DataFrame(rows).filter(pl.col("start").dt.year().is_between(*YEARS))
    return df.with_columns(pl.col("azi").is_between(*SOUTH).alias("south"),
                           # apparent meridional speed of a plane wave travelling at vel along azi
                           (pl.col("vel") / pl.col("azi").radians().cos().abs()).alias("vel_merid"))


def match(a_s, a_e, b_s, b_e, lag=dt.timedelta(0)):
    """For each interval of a: does an interval of b overlap [s - lag, e] within TOL?"""
    return overlap_match([s - lag for s in a_s], a_e, b_s, b_e)


def chance(a_s, a_e, b_s, b_e, lag=dt.timedelta(0)):
    return float(np.mean([match(a_s, a_e, [s + dt.timedelta(days=d) for s in b_s],
                                [e + dt.timedelta(days=d) for e in b_e], lag).mean() for d in SHIFTS]))


def main():
    hf = load_hfint()
    ves = pl.read_parquet(ROOT / "results/vespagram/trains.parquet")
    hs, he = hf["start"].to_list(), hf["end"].to_list()
    south = hf["south"].to_numpy()

    out = {"source": "Segarra et al. 2024, CORA doi:10.34810/data1383 (HF-INT, expert revised)",
           "years": list(YEARS), "n_events": hf.height, "n_south": int(south.sum()),
           "south_azimuth": list(SOUTH),
           "median_period_min": round(float(hf["period_min"].median()), 1),
           "median_vel": round(float(hf["vel"].median())),
           "median_duration_h": round(float(hf["duration_h"].median()), 1)}

    rec = {}
    for name, levels in TIERS.items():
        v = ves.filter(pl.col("confidence").is_in(levels))
        vs, ve = v["t_start"].to_list(), v["t_end"].to_list()
        r = {"n_trains": v.height}
        for lag_name, lag in (("", dt.timedelta(0)), ("_lag60", ONSET_LAG)):
            hit = match(hs, he, vs, ve, lag)
            r[f"south_matched_pct{lag_name}"] = round(float(hit[south].mean() * 100), 1)
            r[f"other_matched_pct{lag_name}"] = round(float(hit[~south].mean() * 100), 1)
            for sub, keep in (("south", south), ("other", ~south)):
                ss = [s for s, k in zip(hs, keep) if k]; se = [e for e, k in zip(he, keep) if k]
                r[f"{sub}_chance_pct{lag_name}"] = round(chance(ss, se, vs, ve, lag) * 100, 1)
        # reverse: share of trains that coincide with a southward HF-INT event (2015-2025)
        r["trains_matched_by_south_pct"] = round(float(overlap_match(
            vs, ve, [s - ONSET_LAG for s, k in zip(hs, south) if k],
            [e for e, k in zip(he, south) if k]).mean() * 100), 1)
        rec[name] = r
    out["matching"] = rec

    # speeds of matched southward events vs the overlapping high+medium train
    hm = ves.filter(pl.col("confidence").is_in(["high", "medium"]))
    pairs = []
    for row in hf.filter(pl.col("south")).iter_rows(named=True):
        o = hm.filter((pl.col("t_start") < row["end"] + TOL) & (pl.col("t_end") > row["start"] - ONSET_LAG - TOL))
        if o.height:
            ov = [min(e, row["end"]) - max(s, row["start"] - ONSET_LAG) for s, e in zip(o["t_start"], o["t_end"])]
            best = o.row(int(np.argmax(ov)), named=True)
            pairs.append((row["vel_merid"], best["speed_m_s"], row["period_min"], best["period_min"]))
    p = np.array(pairs)
    out["matched_speeds"] = {
        "n_pairs": len(p),
        "hfint_merid_median": round(float(np.median(p[:, 0]))),
        "vespagram_median": round(float(np.median(p[:, 1]))),
        "spearman_r": round(float(pl.DataFrame({"a": p[:, 0], "b": p[:, 1]}).select(
            pl.corr("a", "b", method="spearman")).item()), 2),
        "hfint_period_median": round(float(np.median(p[:, 2]))),
        "vespagram_period_median": round(float(np.median(p[:, 3])))}

    # occurrence: yearly counts and monthly profile (southward HF-INT vs high+medium trains)
    hmid = hm["t_start"] + (hm["t_end"] - hm["t_start"]) / 2
    hfs = hf.filter(pl.col("south"))
    yr = list(range(YEARS[0], YEARS[1] + 1))
    a = [hfs.filter(pl.col("start").dt.year() == y).height for y in yr]
    b = [int((hmid.dt.year() == y).sum()) for y in yr]
    am = [hfs.filter(pl.col("start").dt.month() == m).height for m in range(1, 13)]
    bm = [int((hmid.dt.month() == m).sum()) for m in range(1, 13)]
    out["occurrence"] = {"south_per_year": dict(zip(yr, a)), "hm_per_year": dict(zip(yr, b)),
                         "corr_year": round(float(np.corrcoef(a, b)[0, 1]), 2),
                         "corr_month": round(float(np.corrcoef(am, bm)[0, 1]), 2)}

    N = json.loads(NUMBERS.read_text(encoding="utf-8"))
    N["hfint"] = out
    NUMBERS.write_text(json.dumps(N, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
