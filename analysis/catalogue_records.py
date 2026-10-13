"""
catalogue_records.py

Turn the ROI and vespagram LSTID catalogues into record pickles with the same
fields as lstid_rois.pkl, so every analysis/ script runs on either unchanged.
Run it after every vespagram.py catalogue run. The analysis scripts default to
the vespagram high+medium records it writes.

Two units per catalogue, so the two methods are compared like for like:
  lstid_trains.pkl   one record per wave TRAIN (ref_time = train midpoint).
                     ROI fronts are merged into trains exactly as in compare.py
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

import sys
import pickle
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from compare import TRAIN_GAP   # same merge rule as the ROI/vespagram comparison

ROI_PKL   = ROOT / "results/roi/lstid_rois.pkl"
VESPA_PQ  = ROOT / "results/vespagram/trains.parquet"
# records live beside their catalogue, so each analysis script's default output
# (<pickle folder>/climatology) lands next to the catalogue it describes
OUT_ROI   = ROOT / "results/roi"
OUT_VESPA = ROOT / "results/vespagram"

SLOT = dt.timedelta(hours=1)


def roi_trains(pkl=ROI_PKL):
    fronts = sorted(pickle.load(open(pkl, "rb")), key=lambda r: r["start_time"])
    trains = []
    for r in fronts:
        if trains and r["start_time"] <= trains[-1]["end"] + TRAIN_GAP:
            tr = trains[-1]
            tr["end"] = max(tr["end"], r["end_time"])
            tr["members"].append(r)
        else:
            trains.append({"start": r["start_time"], "end": r["end_time"], "members": [r]})
    med = lambda m, k, f=float: float(np.nanmedian([f(x[k]) for x in m]))
    return [{"date":           (t["start"] + (t["end"] - t["start"]) / 2).date(),
             "start_time":     t["start"],
             "end_time":       t["end"],
             "ref_time":       t["start"] + (t["end"] - t["start"]) / 2,
             "speed_m_s":      med(t["members"], "speed_m_s"),
             "period_s":       med(t["members"], "period_s"),
             "peak_amplitude": med(t["members"], "peak_amplitude", lambda v: abs(float(v))),
             "duration_h":     (t["end"] - t["start"]).total_seconds() / 3600,
             "n_fronts":       len(t["members"]),
             "direction":      None}
            for t in trains]


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
             "direction":      r["direction"],
             "confidence":     r["confidence"],
             "geometry":       r["geometry"],
             "n_bands":        r["n_bands"],
             "lat_lo":         r["lat_lo"],
             "lat_hi":         r["lat_hi"]}
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
