"""
paper/confidence_tiers.py

Where the confidence tiers come from: the agreement of the trains with two independent
catalogues (published ROI 2021-2024, southward HF-INT 2015-2025), by confidence-score
bin, against chance.

  lift = share of trains that coincide with a reference event
         / the same share with the reference shifted by +-3..30 days

Chance is computed per train, so long trains, which coincide more often by chance, are
fair. Results are merged into paper/numbers.json under "confidence_tiers".

  python paper/confidence_tiers.py
"""

import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "analysis")]
from catalogue_records import overlap_match
from validate_hfint import load_hfint, SHIFTS, ONSET_LAG, NUMBERS
from validate_published import published_trains, SPAN

EDGES = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.01]
TIERS = {"low": (0, 0.20), "medium": (0.20, 0.35), "high": (0.35, 1.01)}


def flags(ts, te, rs, re_):
    """Per train: does it coincide with a reference event, and its chance of doing so."""
    obs = overlap_match(ts, te, rs, re_).astype(float)
    ch = np.mean([overlap_match(ts, te, [s + dt.timedelta(days=d) for s in rs],
                                [e + dt.timedelta(days=d) for e in re_]) for d in SHIFTS], axis=0)
    return obs, ch


def lifts(score, obs, ch, bins):
    out = {}
    for name, (a, b) in bins.items():
        m = (score >= a) & (score < b)
        out[name] = {"n": int(m.sum()), "lift": round(float(obs[m].mean() / ch[m].mean()), 2) if m.sum() else None}
    return out


def main():
    tr = pl.read_parquet(ROOT / "results/vespagram/trains.parquet").with_columns(
        (pl.col("t_start") + (pl.col("t_end") - pl.col("t_start")) / 2).alias("mid"))
    ts, te, score = tr["t_start"].to_list(), tr["t_end"].to_list(), tr["confidence_score"].to_numpy()

    hf = load_hfint().filter(pl.col("south"))
    obs_hf, ch_hf = flags(ts, te, [s - ONSET_LAG for s in hf["start"]], hf["end"].to_list())

    _, pub = published_trains()
    in_pub = ((tr["mid"] >= SPAN[0]) & (tr["mid"] < SPAN[1])).to_numpy()
    sub = tr.filter(pl.lit(in_pub))
    obs_p, ch_p = flags(sub["t_start"].to_list(), sub["t_end"].to_list(),
                        [t["start_time"] for t in pub], [t["end_time"] for t in pub])

    bins = {f"{a:.2f}-{b:.2f}": (a, b) for a, b in zip(EDGES[:-1], EDGES[1:])}
    out = {"published_roi": {"tiers": lifts(score[in_pub], obs_p, ch_p, TIERS),
                             "bins": lifts(score[in_pub], obs_p, ch_p, bins)},
           "hfint_south": {"tiers": lifts(score, obs_hf, ch_hf, TIERS),
                           "bins": lifts(score, obs_hf, ch_hf, bins)}}

    N = json.loads(NUMBERS.read_text(encoding="utf-8"))
    N["confidence_tiers"] = out
    NUMBERS.write_text(json.dumps(N, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
