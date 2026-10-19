"""
catalogue_records.py

Turn the ROI and vespagram LSTID catalogues into record pickles with the same
fields as lstid_rois.pkl, so every analysis/ script runs on either unchanged.
Run it after every vespagram.py catalogue run. The analysis scripts default to
the vespagram high+medium records it writes.

Two units per catalogue, so the two methods are compared like for like:
  lstid_trains.pkl   one record per wave TRAIN (ref_time = train midpoint).
                     ROI fronts are merged into trains by merge_fronts
                     (intervals closer than TRAIN_GAP). Used for counts and wave
                     characteristics (climatology.py, characteristics_temporal.py,
                     driver_characteristics.py).
  lstid_hours.pkl    one record per LSTID-ACTIVE hour slot (ref_time = the slot):
                     a slot is active when any train overlaps [h-30 min, h+30 min).
                     Used for occurrence vs drivers (driver_activity.py,
                     driver_regression.py, driver_lag.py), whose outcome is "was an
                     LSTID present in this hour" -- one point per train would weigh
                     a 6 h train like a 1 h one, and ROI fronts are not trains.
The vespagram also gets hour pickles per confidence tier (high / medium / low) and
for the high+medium subset (lstid_hours_highmed.pkl).

Field mapping for the vespagram:
  speed_m_s       train speed (full-array or band-median), unsigned
  period_s        period_min * 60
  peak_amplitude  the catalogue amplitude as is: already sqrt(2) * RMS of the beam
                  over the detection (peak of an equivalent sinusoid; the beam is
                  the delay-and-sum stack, so incoherent parts are averaged down)
For ROI trains: medians over member fronts (peak_amplitude is |peak| per front).

  python analysis/catalogue_records.py [--vespa results/vespagram/trains.parquet]
"""

import pickle
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent

# The two rules every comparison between catalogues uses, defined here only.
TRAIN_GAP = dt.timedelta(minutes=60)   # fronts closer than this are one wave train
MATCH_TOL = dt.timedelta(minutes=15)   # two trains match if they overlap within this

ROI_PKL   = ROOT / "results/roi/lstid_rois.pkl"
VESPA_PQ  = ROOT / "results/vespagram/trains.parquet"
# records live beside their catalogue, so each analysis script's default output
# (<pickle folder>/climatology) lands next to the catalogue it describes
OUT_ROI   = ROOT / "results/roi"
OUT_VESPA = ROOT / "results/vespagram"

SLOT = dt.timedelta(hours=1)


def merge_fronts(fronts):
    """Group fronts (dicts with start_time and end_time) into wave trains: a front
    that starts within TRAIN_GAP of the end of the running train joins it. Trains
    stay on the global timeline, so one that crosses midnight is one train.
    Returns a list of [start, end, member fronts]."""
    trains = []
    for r in sorted(fronts, key=lambda r: r["start_time"]):
        if trains and r["start_time"] <= trains[-1][1] + TRAIN_GAP:
            trains[-1][1] = max(trains[-1][1], r["end_time"])
            trains[-1][2].append(r)
        else:
            trains.append([r["start_time"], r["end_time"], [r]])
    return trains


def overlap_match(a_start, a_end, b_start, b_end):
    """For each interval of a: does any interval of b overlap it (within MATCH_TOL)?"""
    bs = np.array(b_start, dtype="datetime64[us]")
    be = np.array(b_end, dtype="datetime64[us]")
    tol = np.timedelta64(int(MATCH_TOL.total_seconds()), "s")
    return np.array([np.any((bs < np.datetime64(e) + tol) & (be > np.datetime64(s) - tol))
                     for s, e in zip(a_start, a_end)])


def roi_trains(pkl=ROI_PKL):
    med = lambda m, k, f=float: float(np.nanmedian([f(x[k]) for x in m]))
    return [{"date":           (start + (end - start) / 2).date(),   # day holding the midpoint
             "start_time":     start,
             "end_time":       end,
             "ref_time":       start + (end - start) / 2,
             "speed_m_s":      med(members, "speed_m_s"),
             "period_s":       med(members, "period_s"),
             "peak_amplitude": med(members, "peak_amplitude", lambda v: abs(float(v))),
             "duration_h":     (end - start).total_seconds() / 3600,
             "n_fronts":       len(members)}
            for start, end, members in merge_fronts(pickle.load(open(pkl, "rb")))]


def vespa_trains(path=VESPA_PQ):
    df = pl.read_parquet(path)
    return [{"date":           r["date"],
             "start_time":     r["t_start"],
             "end_time":       r["t_end"],
             "ref_time":       r["t_start"] + (r["t_end"] - r["t_start"]) / 2,
             "speed_m_s":      float(r["speed_m_s"]),
             "period_s":       float(r["period_min"]) * 60,
             "peak_amplitude": float(r["amplitude"]),
             "duration_h":     (r["t_end"] - r["t_start"]).total_seconds() / 3600,
             "confidence":     r["confidence"]}
            for r in df.iter_rows(named=True)]


def active_hours(trains):
    """Distinct hour slots overlapped by any train, as records (ref_time = slot)."""
    slots = set()
    for t in trains:
        h = (t["start_time"] + SLOT / 2).replace(minute=0, second=0, microsecond=0)
        while h - SLOT / 2 < t["end_time"]:
            slots.add(h)
            h += SLOT
    return [{"date": h.date(), "ref_time": h} for h in sorted(slots)]


def _dump(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(records, open(path, "wb"))
    print(f"  {len(records):6d} records -> {path.relative_to(ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vespa", type=Path, default=VESPA_PQ)
    args = ap.parse_args(argv)

    roi = roi_trains()
    _dump(roi, OUT_ROI / "lstid_trains.pkl")
    _dump(active_hours(roi), OUT_ROI / "lstid_hours.pkl")

    ves = vespa_trains(args.vespa)
    _dump(ves, OUT_VESPA / "lstid_trains.pkl")
    _dump(active_hours(ves), OUT_VESPA / "lstid_hours.pkl")
    for c in ("high", "medium", "low"):
        _dump(active_hours([t for t in ves if t["confidence"] == c]),
              OUT_VESPA / f"lstid_hours_{c}.pkl")
    hm = [t for t in ves if t["confidence"] in ("high", "medium")]
    _dump(hm, OUT_VESPA / "lstid_trains_highmed.pkl")
    _dump(active_hours(hm), OUT_VESPA / "lstid_hours_highmed.pkl")


if __name__ == "__main__":
    main()
