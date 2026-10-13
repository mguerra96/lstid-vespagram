"""
paper/validate_published.py

Compare the vespagram catalogue with the PUBLISHED ROI catalogue of Guerra et al.
(2025), i.e. LSTID_ROI.mat from Zenodo (doi:10.5281/zenodo.14988318), exported to
paper/external/LSTID_ROI_published.csv with MATLAB (writetable). The file spans 11 Jan 2021 - 16 Oct 2024.

Fronts are merged into trains with the same rule as the rebuilt ROI catalogue
(catalogue_records.merge_fronts), then matched to vespagram trains by time overlap
(catalogue_records.overlap_match).
Results are merged into paper/numbers.json under "published_roi".

  python paper/validate_published.py
"""

import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "analysis")]
from catalogue_records import active_hours, merge_fronts, overlap_match
from validate_hfint import chance   # same day-shift chance level as for HF-INT

PUB = Path(__file__).resolve().parent / "external/LSTID_ROI_published.csv"
NUMBERS = Path(__file__).resolve().parent / "numbers.json"
# span actually covered by the published file (it ends on 16 Oct 2024, not 31 Dec)
SPAN = (dt.datetime(2021, 1, 1), dt.datetime(2024, 10, 17))


def published_trains():
    df = (pl.read_csv(PUB, infer_schema_length=0)
          .with_columns(pl.col("start_time", "end_time").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                        pl.col("speed", "period").cast(pl.Float64))
          .sort("start_time"))
    return df, [{"start_time": s, "end_time": e, "n": len(m)}
                for s, e, m in merge_fronts(df.iter_rows(named=True))]


def main():
    fronts, pub = published_trains()
    ves = pl.read_parquet(ROOT / "results/vespagram/trains.parquet")
    mid = ves["t_start"] + (ves["t_end"] - ves["t_start"]) / 2
    ves = ves.filter((mid >= SPAN[0]) & (mid < SPAN[1]))

    ps = [t["start_time"] for t in pub]; pe = [t["end_time"] for t in pub]
    out = {"source": "Guerra et al. 2025, Zenodo doi:10.5281/zenodo.14988318, LSTID_ROI.mat",
           "n_fronts": fronts.height, "n_trains": len(pub),
           "speed_mean_std": [round(float(fronts["speed"].mean())), round(float(fronts["speed"].std()))],
           "period_min_mean_std": [round(float(fronts["period"].mean() / 60), 1),
                                   round(float(fronts["period"].std() / 60), 1)]}

    tiers = {"high": ["high"], "medium": ["medium"], "low": ["low"],
             "high_medium": ["high", "medium"], "all": ["high", "medium", "low"]}
    rec = {}
    for name, levels in tiers.items():
        v = ves.filter(pl.col("confidence").is_in(levels))
        vs, ve = v["t_start"].to_list(), v["t_end"].to_list()
        rec[name] = {"n_trains": v.height,
                     "published_matched_pct": round(float(overlap_match(ps, pe, vs, ve).mean() * 100), 1),
                     "published_chance_pct": round(chance(ps, pe, vs, ve) * 100, 1),
                     "matched_by_published_pct": round(float(overlap_match(vs, ve, ps, pe).mean() * 100), 1)}
    out["train_matching"] = rec

    # hourly occupancy, same slot rule as catalogue_records.active_hours
    n_hours = int((SPAN[1] - SPAN[0]).total_seconds() // 3600)
    h_pub = {r["ref_time"] for r in active_hours(pub)}
    hm = ves.filter(pl.col("confidence").is_in(["high", "medium"]))
    h_hm = {r["ref_time"] for r in active_hours(
        [{"start_time": a, "end_time": b} for a, b in zip(hm["t_start"], hm["t_end"])])}
    h_pub = {h for h in h_pub if SPAN[0] <= h.replace(tzinfo=None) < SPAN[1]}
    h_hm = {h for h in h_hm if SPAN[0] <= h.replace(tzinfo=None) < SPAN[1]}
    out["hours"] = {"published_pct": round(len(h_pub) / n_hours * 100, 1),
                    "hm_pct": round(len(h_hm) / n_hours * 100, 1),
                    "published_hours_in_hm_pct": round(len(h_pub & h_hm) / len(h_pub) * 100, 1),
                    "hm_hours_in_published_pct": round(len(h_pub & h_hm) / len(h_hm) * 100, 1)}

    N = json.loads(NUMBERS.read_text(encoding="utf-8"))
    N["published_roi"] = out
    NUMBERS.write_text(json.dumps(N, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
