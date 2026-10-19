"""
test_vespagram.py

Synthetic checks of detection and train building: known wave fields are built on
the same decimated grid the pipeline uses, band-passed, and run through
vespagram.analyse. Then checks of the rule that settles trains crossing midnight,
on hand-made train records. Run: python test_vespagram.py (exit code = failures)
"""

import datetime as dt

import numpy as np

import vespagram as v

CFG  = v.VespaConfig()
LATS = np.arange(57.5, 37.49, -CFG.lat_step_deg)          # row 0 = max latitude
T    = np.arange(-7200.0, 86400.0 + 7200.0, CFG.time_step_s)
RNG  = np.random.default_rng(1)


def packet(t_center, period_s, amp, arrival, lat_mask=None, width_s=5400):
    """A Gaussian-enveloped wave packet whose fronts reach each row at
    t + arrival(lat) (arrival < 0 = earlier). Returns an (n_lat, n_t) field."""
    Z = np.zeros((len(LATS), len(T)))
    for i, la in enumerate(LATS):
        if lat_mask is not None and not lat_mask(la):
            continue
        tt = T - arrival(la)
        Z[i] = amp * np.sin(2 * np.pi * (tt - t_center) / period_s) * \
               np.exp(-((tt - t_center) / width_s) ** 2)
    return Z


def plane(speed, lat_ref=47.5):
    """Arrival offset of an equatorward plane wave (north sees it first)."""
    return lambda la: -(la - lat_ref) * v.M_PER_DEG / speed


def curved(v_north, v_south, lat_ref=47.5):
    """Speed changing linearly with latitude; arrival = integral of 1/v."""
    def arrival(la):
        lat = np.linspace(lat_ref, la, 200)
        vel = v_south + (v_north - v_south) * (lat - 37.5) / 20
        return -np.trapezoid(1 / vel, lat * v.M_PER_DEG)
    return arrival


def run(Z, noise=0.005, **overrides):
    cfg = v.replace(CFG, **overrides)
    Z = Z + noise * RNG.standard_normal(Z.shape)
    X = v.bandpass(Z, cfg.time_step_s, cfg)
    _, _, trains, _ = v.analyse(X, LATS, T, cfg)
    return trains


def describe(trains):
    return "; ".join(f"{t['t_start_s'] / 3600:.1f}-{t['t_end_s'] / 3600:.1f}h "
                     f"{t['speed_m_s']:.0f} m/s [{t['speed_lo_m_s']:.0f}-{t['speed_hi_m_s']:.0f}] "
                     f"{t['period_s'] / 60:.0f} min, cover {t['coverage']:.2f}, "
                     f"{t['n_crests']}/{t['n_troughs']} crests/troughs, "
                     f"score {t['confidence_score']:.2f} {t['confidence']}" for t in trains) or "none"


CASES = {
    # name: (field, check on the resulting trains)
    "plane 600 m/s": (
        packet(12 * 3600, 3600, 0.05, plane(600)),
        lambda tr: len(tr) == 1 and abs(tr[0]["speed_m_s"] - 600) < 60
                   and tr[0]["coverage"] > 0.9 and tr[0]["confidence"] == "high"),
    # one cycle only: a clear wave crossing the whole array scores high however
    # short it is (the score has no duration term)
    "single cycle 600 m/s": (
        packet(12 * 3600, 3600, 0.05, plane(600), width_s=1800),
        lambda tr: len(tr) == 1 and tr[0]["n_crests"] >= 1 and tr[0]["n_troughs"] >= 1
                   and tr[0]["confidence"] == "high"),
    "curved 400 N -> 1000 S: one train between the two speeds": (
        packet(12 * 3600, 3600, 0.05, curved(400, 1000)),
        lambda tr: len(tr) == 1 and 400 < tr[0]["speed_m_s"] < 1000),
    "poleward 500 m/s: not catalogued": (
        packet(12 * 3600, 3600, 0.05, plane(-500)),
        lambda tr: len(tr) == 0),
    # outside the LSTID window: the flank of the ridge must not be picked at the
    # window edge
    "slow 200 m/s: not catalogued": (
        packet(12 * 3600, 3600, 0.05, plane(200)),
        lambda tr: len(tr) == 0),
    "near-vertical 4000 m/s: not catalogued": (
        packet(12 * 3600, 3600, 0.05, plane(4000)),
        lambda tr: len(tr) == 0),
    "north-only 450 m/s (47.5-57.5): one train, partial coverage": (
        packet(12 * 3600, 3600, 0.05, plane(450), lat_mask=lambda la: la >= 47.5),
        lambda tr: len(tr) == 1 and tr[0]["coverage"] < 0.8),
    # no two simultaneous LSTID trains over this array: one train by definition
    "two simultaneous, 400 and 1200 m/s: one train": (
        packet(12 * 3600, 3600, 0.05, plane(400)) +
        packet(12 * 3600 + 900, 3600, 0.05, plane(1200)),
        lambda tr: len(tr) == 1),
    "500 S-half + 1000 N-half: one train, interval reaching both": (
        packet(12 * 3600, 3600, 0.05, plane(500), lat_mask=lambda la: la < 47.5) +
        packet(12 * 3600 + 1200, 3600, 0.05, plane(1000), lat_mask=lambda la: la >= 47.5),
        lambda tr: len(tr) == 1 and tr[0]["speed_lo_m_s"] <= 600 and tr[0]["speed_hi_m_s"] >= 900),
    # packets far apart are two trains
    "consecutive 500 then 1200 m/s, 7 h apart: two trains": (
        packet(8 * 3600, 3600, 0.05, plane(500)) +
        packet(15 * 3600, 3600, 0.05, plane(1200)),
        lambda tr: len(tr) == 2),
    "noise only": (
        np.zeros((len(LATS), len(T))),
        lambda tr: len(tr) == 0),
}


# ── trains crossing midnight (vespagram.resolve_day_boundaries) ────────────────

D0, D1 = dt.date(2024, 1, 1), dt.date(2024, 1, 2)


def rec(day, train_id, start_h, end_h):
    """A train record as one day's run reports it; hours count from 00 UT of D0."""
    t0 = dt.datetime.combine(D0, dt.time())
    return {"date": day, "train_id": train_id,
            "t_start": t0 + dt.timedelta(hours=start_h), "t_end": t0 + dt.timedelta(hours=end_h)}


BOUNDARY_CASES = {
    # name: (records of the two days, the (date, train_id) that must stay)
    "one train claimed by both days in two versions: the longer stays": (
        [rec(D0, 1, 21, 24.5), rec(D1, 1, 23, 28)], {(D1, 1)}),
    "trains of adjacent days that do not overlap: both stay": (
        [rec(D0, 1, 20, 23.5), rec(D1, 1, 24.5, 27)], {(D0, 1), (D1, 1)}),
    "trains that do not cross midnight are left alone": (
        [rec(D0, 1, 3, 6), rec(D0, 2, 10, 14), rec(D1, 1, 30, 33)],
        {(D0, 1), (D0, 2), (D1, 1)}),
}


def boundary_result(records):
    """(date, train_id) of the trains and of the detections that are kept."""
    dets = [{"date": r["date"], "train_id": r["train_id"]} for r in records for _ in range(2)]
    trains, dets = v.resolve_day_boundaries(records, dets)
    ids = lambda rows: sorted({(r["date"], r["train_id"]) for r in rows})
    return ids(trains), ids(dets), len(trains), len(dets)


if __name__ == "__main__":
    n_fail = 0
    for name, (Z, check, *overrides) in CASES.items():
        tr = run(Z, noise=0.02 if name == "noise only" else 0.005, **(overrides[0] if overrides else {}))
        ok = check(tr)
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL':5s}  {name:48s} -> {describe(tr)}")
    for name, (records, expected) in BOUNDARY_CASES.items():
        kept, kept_dets, n_tr, n_det = boundary_result(records)
        # one row per train, and exactly the detections of the kept trains
        ok = kept == kept_dets == sorted(expected) and n_tr == len(expected) and n_det == 2 * n_tr
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL':5s}  midnight: {name} -> kept {[(str(d), k) for d, k in kept]}")
    print(f"\n{n_fail} unexpected failure(s)")
    raise SystemExit(n_fail)
