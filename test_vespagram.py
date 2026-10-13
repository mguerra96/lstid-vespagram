"""
test_vespagram.py

Synthetic checks of detection and train linking: known wave fields are built on
the same decimated grid the pipeline uses, band-passed, and run through
vespagram.analyse. Run: python test_vespagram.py
"""

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
                     f"{t['direction'][:3]} {t['speed_m_s']:.0f} m/s {t['period_s'] / 60:.0f} min "
                     f"lat {t['lat_lo']}-{t['lat_hi']} {t['geometry']} {t['confidence']} "
                     f"(bands {t['n_bands_picked']}/{t['n_bands']})" for t in trains) or "none"


CASES = {
    # name: (field, check on the resulting trains)
    "plane 600 m/s": (
        packet(12 * 3600, 3600, 0.05, plane(600)),
        lambda tr: len(tr) == 1 and abs(tr[0]["speed_m_s"] - 600) < 60
                   and tr[0]["geometry"] == "planar" and tr[0]["confidence"] == "high"),
    "curved 400 N -> 1000 S": (
        packet(12 * 3600, 3600, 0.05, curved(400, 1000)),
        lambda tr: len(tr) == 1 and tr[0]["geometry"] == "curved"),
    "poleward 500 m/s (dropped: equatorward-only catalogue)": (
        packet(12 * 3600, 3600, 0.05, plane(-500)),
        lambda tr: len(tr) == 0),
    "poleward 500 m/s, poleward kept": (
        packet(12 * 3600, 3600, 0.05, plane(-500)),
        lambda tr: len(tr) == 1 and tr[0]["direction"] == "poleward",
        {"directions": ("equatorward", "poleward")}),
    "north-only 450 m/s (47.5-57.5)": (
        packet(12 * 3600, 3600, 0.05, plane(450), lat_mask=lambda la: la >= 47.5),
        lambda tr: len(tr) == 1 and tr[0]["lat_lo"] >= 42.5),
    # simultaneous LSTID trains are one train by definition (merge_simultaneous)
    "two simultaneous, 400 and 1200 m/s, same period: one train": (
        packet(12 * 3600, 3600, 0.05, plane(400)) +
        packet(12 * 3600 + 900, 3600, 0.05, plane(1200)),
        lambda tr: len(tr) == 1),
    "two simultaneous, kept apart without merge_simultaneous": (
        packet(12 * 3600, 3600, 0.05, plane(400)) +
        packet(12 * 3600 + 900, 3600, 0.05, plane(1200)),
        lambda tr: len(tr) == 2,
        {"merge_simultaneous": False}),
    # two packets at different speeds one after the other, not overlapping in
    # time: two trains (packets closer than this overlap and are merged)
    "consecutive 500 then 1200 m/s, 5 h apart": (
        packet(9 * 3600, 3600, 0.05, plane(500)) +
        packet(14 * 3600, 3600, 0.05, plane(1200)),
        lambda tr: len(tr) == 2),
    # a slow wave in the south and a fast one in the north at the same time: one
    # 'curved' train (formerly a known limitation; one train by definition now)
    "500 S-half + 1000 N-half: one curved train": (
        packet(12 * 3600, 3600, 0.05, plane(500), lat_mask=lambda la: la < 47.5) +
        packet(12 * 3600 + 1200, 3600, 0.05, plane(1000), lat_mask=lambda la: la >= 47.5),
        lambda tr: len(tr) == 1 and tr[0]["geometry"] == "curved"),
    # near-vertical front (2500 m/s) tilted poleward in the south half and
    # equatorward in the north half: never one train mixing both directions
    "mixed direction, 2500 m/s S-half poleward + N-half equatorward": (
        packet(12 * 3600, 3600, 0.05, plane(-2500), lat_mask=lambda la: la < 47.5) +
        packet(12 * 3600, 3600, 0.05, plane(2500), lat_mask=lambda la: la >= 47.5),
        lambda tr: all(len({np.sign(s) for s in t["profile_speed"]}) <= 1 for t in tr)),
    "noise only": (
        np.zeros((len(LATS), len(T))),
        lambda tr: len(tr) == 0),
}


if __name__ == "__main__":
    n_fail = 0
    for name, (Z, check, *overrides) in CASES.items():
        tr = run(Z, noise=0.02 if name == "noise only" else 0.005, **(overrides[0] if overrides else {}))
        ok = check(tr)
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL':5s}  {name:48s} -> {describe(tr)}")
    print(f"\n{n_fail} unexpected failure(s)")
