"""
vespagram.py

LSTID detection and characterisation by slant-stacking the keogram, treating each
latitude row as one sensor of a north-south array (the seismological vespagram /
semblance method).

Pipeline:
  1. Load keogram grid (reuses detect_roi.load_grid, padded with neighbour days)
  2. Decimate in latitude and time, band-pass each row to the LSTID period band
  3. For the full aperture and for overlapping sliding latitude bands, and for each
     trial slowness p: delay-and-sum the rows -> semblance S(t, p) (coherence,
     0..1, amplitude-independent) and beam RMS A(t, p)
  4. Pick ridges of S along p; each connected ridge in one array is a DETECTION
     (speed, span, beam crests/troughs, period, amplitude)
  5. Link detections across arrays into TRAINS when their crests line up where the
     arrays overlap (phase continuity), so one wave seen by several arrays is one
     train and a curved front becomes one train with a speed-vs-latitude profile
  6. Keep LSTID trains only (300-1500 m/s, period >= 30 min) and class them by
     direction: equatorward or poleward. Linking happens BEFORE this filter, so a
     curved front whose slow northern band alone would fail still stays whole.
     Only the directions in cfg.directions are kept (default: equatorward only).
     Each train gets a confidence score in 0..1 (confidence_score) and, from it,
     a tier (high / medium / low, see train_confidence).
  7. Plot and save detections + trains

Sign convention: p > 0 is EQUATORWARD propagation (the north of the array sees
the front first). Speed = 1 / |p|.
"""

import time
import logging
import argparse
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, binary_dilation
from scipy.signal import butter, sosfiltfilt, find_peaks
from skimage.measure import label
from geopy.distance import geodesic

import paths
from detect_roi import Config as RoiConfig, load_grid, discover_dates, SECONDS_PER_DAY


# ── Configuration ───────────────────────────────────────────────────────────────

@dataclass
class VespaConfig:
    # ── Paths & date ──
    keo_folder: Path = paths.KEO_DIR
    out_dir:    Path = paths.ROOT / "results/vespagram"
    input_date: dt.date = dt.date(2015, 1, 7)
    longitude:  float = 17.5
    border_pad_hours: float = 12    # hours of the neighbouring days read on each side: long
                                    # enough for the day that owns a midnight-crossing train
                                    # (midpoint inside it) to see the whole train

    # ── Pre-processing ──
    lat_step_deg: float = 0.25    # latitude spacing of the decimated array [deg]
    time_step_s:  int   = 120     # time cadence after decimation [s]
    band_min_period_s: float = 25 * 60    # band-pass short-period corner [s]
    band_max_period_s: float = 200 * 60   # band-pass long-period corner [s]

    # ── Arrays: the full aperture plus sliding latitude bands ──
    band_width_deg: float = 10     # sliding sub-array aperture [deg]
    band_step_deg:  float = 2.5    # sliding step [deg]

    # ── Slant stack ──
    max_slowness: float = 1 / 120   # |p| grid limit [s/m]  (speeds down to 120 m/s)
    n_slowness:   int   = 241
    semblance_window_s: float = 30 * 60   # boxcar window for semblance / beam RMS [s]
    min_valid_rows_frac: float = 0.7      # need this fraction of rows inside the data

    # ── Detection ──
    semblance_min: float = 0.6     # ridge must be at least this coherent [-]
    beam_rms_min:  float = 0.01    # and at least this strong [TECu]
    min_duration_s: float = 45 * 60
    lstid_min_speed: float = 300   # [m/s]
    lstid_max_speed: float = 1500  # [m/s]
    lstid_min_period: float = 30 * 60
    # ridges are only picked inside this (wider) speed range, either direction, so
    # near-vertical structures (e.g. simultaneous TEC changes) never become
    # detections and cannot chain onto an LSTID train; the margin lets a curved
    # front whose band speeds straddle 300/1500 m/s still link up
    ridge_min_speed: float = 250   # [m/s]
    ridge_max_speed: float = 2000  # [m/s]

    # ── Linking detections into trains ──
    link_time_tol_s:   float = 15 * 60  # detections must overlap in time within this
    link_period_ratio: float = 1.3      # and have periods within this ratio
    link_min_corr:     float = 0.8      # waveform correlation needed at the shared latitude [-]
    extend_semblance_min: float = 0.5   # a neighbouring band joins a train if this coherent
                                        # at the train's speed (below semblance_min: the band
                                        # need not have its own ridge, only line up) [-]
    default_period_s:  float = 60 * 60  # used when a detection has no measurable period
    curved_speed_ratio: float = 1.5     # band speeds differing by more -> 'curved' [-]
    keep_single_band:  bool  = False    # keep trains seen by only one band (excluded by
                                        # default: mostly not LSTIDs in the blind audits)
    merge_simultaneous: bool = True     # LSTID trains of one direction that overlap in
                                        # time are one train: over this array there are
                                        # no two simultaneous LSTID trains
    directions: tuple = ("equatorward",)  # directions kept in the catalogue; poleward
                                          # trains showed no auroral-driver dependence
                                          # and were not confirmed in the audit
    confidence_cycles: float = 2.0      # e-folding number of tracked cycles in the
                                        # confidence score [-]
    confidence_medium: float = 0.30     # score thresholds of the 'medium' and 'high'
    confidence_high:   float = 0.40     # tiers [-]
    confidence_min_speed: float = 350   # trains slower than this drop one tier: near the
                                        # 300 m/s limit they blend with medium-scale TIDs [m/s]

    @property
    def grid_path(self) -> Path:
        return self.keo_folder / f"keogram_grid_{self.input_date}.parquet"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1-2 — Load, decimate, band-pass
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(cfg):
    """
    Return (X, lats, t) where X is the band-passed keogram on the decimated grid
    (row 0 = max latitude), lats the latitude of each row [deg] and t the time of
    each column [s since day-D midnight, negative/>=86400 in the padding].
    """
    roi_cfg = RoiConfig(keo_folder=cfg.keo_folder, input_date=cfg.input_date,
                        border_pad_hours=cfg.border_pad_hours)
    Z, lats_asc, t, _ = load_grid(roi_cfg)
    lats = lats_asc[::-1]                      # load_grid flips Z: row 0 = max lat

    # time: block-average to the target cadence
    step = int(t[1] - t[0])
    f    = max(1, int(round(cfg.time_step_s / step)))
    n    = (Z.shape[1] // f) * f
    Z    = Z[:, :n].reshape(Z.shape[0], -1, f).mean(axis=2)
    t    = t[:n].reshape(-1, f).mean(axis=1)

    # latitude: anti-alias smooth, then subsample rows
    dlat = abs(lats[1] - lats[0])
    r    = max(1, int(round(cfg.lat_step_deg / dlat)))
    Z    = gaussian_filter1d(Z, sigma=r / 2, axis=0)[::r]
    lats = lats[::r]

    return bandpass(Z, t[1] - t[0], cfg), lats, t


def bandpass(Z, dt_s, cfg):
    """Zero-phase band-pass of each row to the LSTID period band."""
    sos = butter(4, [1 / cfg.band_max_period_s, 1 / cfg.band_min_period_s],
                 btype="band", fs=1 / dt_s, output="sos")
    return sosfiltfilt(sos, Z, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Slant stack
# ══════════════════════════════════════════════════════════════════════════════

def meridian_distance(lats, lat_ref, lon):
    """Signed along-meridian distance of each latitude from lat_ref [m], north > 0."""
    return np.array([np.sign(la - lat_ref) * geodesic((lat_ref, lon), (la, lon)).meters
                     for la in lats])


def shift_rows(X, shift):
    """
    Sample row i at fractional index (k - shift[i]) by linear interpolation.
    Returns (Y, valid) where valid marks samples that fell inside the record.
    """
    n_row, n_t = X.shape
    tt    = np.arange(n_t)[None, :] - shift[:, None]
    i0    = np.floor(tt).astype(int)
    w     = tt - i0
    valid = (i0 >= 0) & (i0 + 1 < n_t)
    i0    = np.clip(i0, 0, n_t - 2)
    rows  = np.arange(n_row)[:, None]
    Y     = (1 - w) * X[rows, i0] + w * X[rows, i0 + 1]
    return np.where(valid, Y, 0.0), valid


def slant_stack(X, dist, p_grid, dt_s, cfg):
    """
    Delay-and-sum over trial slownesses. For an equatorward front with slowness p
    the row at distance d (north > 0) sees it d*p seconds EARLIER than the
    reference latitude, so it is sampled at t - d*p to align it.

    Returns S (semblance) and A (beam RMS), both shaped (n_p, n_t):
      S = sum_win (sum_i x_i)^2 / (N * sum_win sum_i x_i^2)   in [0, 1]
      A = sqrt(mean_win beam^2),  beam = mean_i x_i
    """
    win   = max(1, int(round(cfg.semblance_window_s / dt_s)))
    n_row = X.shape[0]
    S = np.full((len(p_grid), X.shape[1]), np.nan)
    A = np.full_like(S, np.nan)
    for j, p in enumerate(p_grid):
        Y, valid = shift_rows(X, dist * p / dt_s)
        n    = valid.sum(axis=0)
        s    = Y.sum(axis=0)
        num  = uniform_filter1d(s ** 2, win)
        den  = uniform_filter1d(n * (Y ** 2).sum(axis=0), win)
        beam = s / np.maximum(n, 1)
        ok   = n >= cfg.min_valid_rows_frac * n_row
        S[j, ok] = num[ok] / np.maximum(den[ok], 1e-12)
        A[j, ok] = np.sqrt(np.maximum(uniform_filter1d(beam ** 2, win), 0))[ok]
    return S, A


def beam_at(X, dist, p, dt_s):
    """Beam (mean of aligned rows) at a single slowness."""
    Y, valid = shift_rows(X, dist * p / dt_s)
    return Y.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Ridge picking: one detection per connected ridge per array
# ══════════════════════════════════════════════════════════════════════════════

M_PER_DEG = 111_195.0   # along-meridian metres per degree (for linking/drawing)


def array_list(lats, cfg):
    """(lat_lo, lat_hi, is_full) for the full aperture and each sliding band."""
    lo, hi = float(lats.min()), float(lats.max())
    starts = np.arange(lo, hi - cfg.band_width_deg + 1e-6, cfg.band_step_deg)
    return [(lo, hi, True)] + [(float(s), float(s + cfg.band_width_deg), False) for s in starts]


def refine_slowness(S, jj, kk, p_grid):
    """
    Sub-grid slowness of each ridge pixel (jj, kk): the vertex of the parabola
    through the semblance at the pixel and at its two neighbours along p. Without
    it every speed sits on the slowness grid (1 / (n * dp)), which leaves only a
    handful of possible values above 1000 m/s.
    """
    j0  = np.clip(jj, 1, len(p_grid) - 2)
    a, b, c = S[j0 - 1, kk], S[j0, kk], S[j0 + 1, kk]
    den = a - 2 * b + c                                   # < 0 at a maximum
    off = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, 1.0), 0.0)
    return p_grid[j0] + np.clip(off, -0.5, 0.5) * (p_grid[1] - p_grid[0])


def pick_detections(S, A, X, dist, p_grid, t, lat_lo, lat_hi, is_full, cfg):
    """
    Ridges = pixels that are a local maximum of S along p and pass the semblance
    and beam-amplitude thresholds. Ridges are joined across time (allowing one
    slowness bin of jitter) and each connected ridge becomes one detection, at
    the median of its sub-grid ridge slownesses (refine_slowness).
    Detections in the day padding are kept: they may link to a day-D train.
    """
    dt_s = t[1] - t[0]
    Sf   = np.nan_to_num(S, nan=0.0)
    up   = np.vstack([Sf[1:], np.full((1, Sf.shape[1]), -np.inf)])
    down = np.vstack([np.full((1, Sf.shape[1]), -np.inf), Sf[:-1]])
    in_range = ((np.abs(p_grid) >= 1 / cfg.ridge_max_speed)
                & (np.abs(p_grid) <= 1 / cfg.ridge_min_speed))[:, None]
    ridge = ((Sf >= up) & (Sf >= down) & in_range
             & (Sf >= cfg.semblance_min)
             & (np.nan_to_num(A) >= cfg.beam_rms_min))

    labels = label(binary_dilation(ridge, structure=np.ones((3, 1))), connectivity=2)
    dets = []
    for lbl in range(1, labels.max() + 1):
        jj, kk = np.nonzero(ridge & (labels == lbl))
        if kk.size == 0:
            continue
        t0, t1 = t[kk.min()], t[kk.max()]
        if t1 - t0 < cfg.min_duration_s:
            continue

        p_ref = refine_slowness(Sf, jj, kk, p_grid)
        dets.append(make_detection(X, dist, float(np.median(p_ref)), kk.min(), kk.max() + 1,
                                   t, lat_lo, lat_hi, is_full, float(np.mean(Sf[jj, kk])), cfg,
                                   ridge_t=t[kk], ridge_p=p_ref, ridge_s=Sf[jj, kk]))
    return dets


def make_detection(X, dist, p, k0, k1, t, lat_lo, lat_hi, is_full, semblance, cfg,
                   ridge_t=None, ridge_p=None, ridge_s=None, extended=False):
    """
    Describe one detection: the array's beam at slowness p over samples [k0, k1),
    its crests and troughs, period and amplitude. The full-length beam is kept
    (as the wave seen at the array centre) for the waveform-correlation link test.
    extended=True marks a band added by the extension step rather than picked
    from its own vespagram ridge.
    """
    dt_s = t[1] - t[0]
    beam = beam_at(X, dist, p, dt_s)
    b    = beam[k0:k1]

    # crests and troughs of the beam: alternate by construction of a band-
    # limited signal; half-period = spacing between consecutive extrema
    min_sep = max(1, int(cfg.band_min_period_s / 2 / dt_s))
    pk, _ = find_peaks(b,  distance=min_sep, height=cfg.beam_rms_min)
    tr, _ = find_peaks(-b, distance=min_sep, height=cfg.beam_rms_min)
    ext   = np.sort(np.concatenate([pk, tr]))

    return {
        "array":      f"{lat_lo:g}-{lat_hi:g}",
        "is_full":    is_full,
        "lat_lo":     lat_lo,
        "lat_hi":     lat_hi,
        "lat_c":      0.5 * (lat_lo + lat_hi),
        "t_start_s":  float(t[k0]),
        "t_end_s":    float(t[k1 - 1]),
        "slowness":   p,
        "speed_m_s":  1 / abs(p) if p != 0 else np.inf,
        "period_s":   2 * np.median(np.diff(ext)) * dt_s if len(ext) >= 2 else np.nan,
        "amplitude":  float(np.sqrt(2) * np.sqrt(np.mean(b ** 2))),
        "semblance":  semblance,
        "n_extrema":  int(len(ext)),
        "crests_s":   t[k0 + pk].tolist(),
        "troughs_s":  t[k0 + tr].tolist(),
        "beam":       beam.astype(np.float32),
        "extended":   extended,
        "ridge_t":    np.array([]) if ridge_t is None else ridge_t,
        "ridge_p":    np.array([]) if ridge_p is None else ridge_p,
        "ridge_s":    np.array([]) if ridge_s is None else ridge_s,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Link detections across arrays into trains (waveform continuity)
# ══════════════════════════════════════════════════════════════════════════════

def wave_at(det, lat, t):
    """The detection's beam (the wave at its array centre) moved to latitude
    `lat` along its own moveout: a front reaching the centre at time t0 reaches
    `lat` at t0 - (lat - lat_c) * p, so the wave there is beam(t - tau)."""
    tau = -(lat - det["lat_c"]) * M_PER_DEG * det["slowness"]
    return np.interp(t - tau, t, det["beam"], left=0.0, right=0.0)


def link_score(a, b, t, cfg):
    """
    None if detections a and b (from different arrays) cannot be the same wave;
    otherwise a score in (0, 1], higher = better.

    Same wave means: overlapping in latitude and time, similar period, and — the
    decisive test — the wave each detection predicts at the middle of the shared
    latitude band has the same shape and timing. Each beam is moved there along
    its OWN measured speed and the two are correlated over their common time
    span (±half a period). A curved front passes (each array's local speed moves
    its wave correctly over the short distance to the shared band); two unrelated
    simultaneous trains do not, because their waveforms differ in phase and
    envelope. Speed alone cannot decide this: a 10° band cannot tell 500 from
    1000 m/s apart.
    """
    lo, hi = max(a["lat_lo"], b["lat_lo"]), min(a["lat_hi"], b["lat_hi"])
    if hi <= lo:
        return None
    tol = cfg.link_time_tol_s
    if a["t_start_s"] > b["t_end_s"] + tol or b["t_start_s"] > a["t_end_s"] + tol:
        return None
    Ta, Tb = a["period_s"], b["period_s"]
    if np.isfinite(Ta) and np.isfinite(Tb) and max(Ta, Tb) / min(Ta, Tb) > cfg.link_period_ratio:
        return None
    T = np.nanmean([Ta, Tb]) if np.isfinite(Ta) or np.isfinite(Tb) else cfg.default_period_s

    w0 = max(a["t_start_s"], b["t_start_s"]) - T / 2
    w1 = min(a["t_end_s"], b["t_end_s"]) + T / 2
    if w1 - w0 < T / 2:                     # too little common time to compare shapes
        return None
    m   = (t >= w0) & (t <= w1)
    lat = 0.5 * (lo + hi)
    sa, sb = wave_at(a, lat, t)[m], wave_at(b, lat, t)[m]
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    r   = float(np.sum(sa * sb) / den) if den > 0 else 0.0
    return r if r >= cfg.link_min_corr else None


def sequel_score(a, b, t, cfg):
    """
    None unless a and b are CONSECUTIVE pieces of one train in the same array:
    one ends where the other starts (gap and overlap both within link_time_tol_s)
    and the wave is continuous across the junction. A ridge whose speed drifts
    (e.g. fronts steepening over time) is otherwise split where it wanders. Both
    beams are the wave at the same array centre, so they are compared directly
    over one period either side of the junction.
    """
    first, second = (a, b) if a["t_start_s"] <= b["t_start_s"] else (b, a)
    if abs(second["t_start_s"] - first["t_end_s"]) > cfg.link_time_tol_s:
        return None
    Ta, Tb = a["period_s"], b["period_s"]
    if np.isfinite(Ta) and np.isfinite(Tb) and max(Ta, Tb) / min(Ta, Tb) > cfg.link_period_ratio:
        return None
    T = np.nanmean([Ta, Tb]) if np.isfinite(Ta) or np.isfinite(Tb) else cfg.default_period_s
    junction = 0.5 * (first["t_end_s"] + second["t_start_s"])
    m   = (t >= junction - T) & (t <= junction + T)
    sa, sb = a["beam"][m], b["beam"][m]
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    r   = float(np.sum(sa * sb) / den) if den > 0 else 0.0
    return r if r >= cfg.link_min_corr else None


def distinct_waves(a, b, t, cfg):
    """
    True if two detections of the SAME array are different waves: they overlap
    in time, or — in the full array only — their speeds differ by more than the
    array can resolve; and they are not consecutive pieces of one drifting ridge
    (sequel_score). Such a pair must never end up in one train: this lets the
    best-resolving array veto a merge that a coarser array (which blends the two
    waves) would propose. A 10° sub-array resolves speed too coarsely for a speed
    change between its consecutive pieces to prove two waves: with that veto a
    front fast in the north and slow in the south, slowing down over time, was
    split into two overlapping trains (12 August 2024, 04:36-10:34 UT). So
    sub-arrays veto only simultaneous detections.
    """
    if a["array"] != b["array"]:
        return False
    if sequel_score(a, b, t, cfg) is not None:
        return False
    if _overlap(a, b) > cfg.link_time_tol_s:
        return True
    if not a["is_full"]:
        return False
    T = np.nanmean([a["period_s"], b["period_s"]])
    T = T if np.isfinite(T) else cfg.default_period_s
    D = (a["lat_hi"] - a["lat_lo"]) * M_PER_DEG
    return abs(a["slowness"] - b["slowness"]) > 0.44 * T / D


def link_trains(dets, t, cfg):
    """
    Group detections into trains. Candidate links (link_score across arrays,
    sequel_score within one) are accepted best-first, but two groups are only
    merged if that would not put a pair of distinct_waves in one train. So a band
    ridge that blends two waves the full aperture resolves cannot chain them,
    whichever order the links arrive in. Detections propagating in opposite
    directions are never linked (ridges are picked at |v| <= ridge_max_speed, so
    the slowness sign is always defined): a near-vertical structure whose apparent
    direction flips between bands is not one travelling wave. Since every merge
    goes through a linked pair, each train keeps a single direction. Returns a
    list of trains, each a list of detection indices.
    """
    group = {i: {i} for i in range(len(dets))}          # detection -> its group (shared set)

    cands = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if np.sign(dets[i]["slowness"]) != np.sign(dets[j]["slowness"]):
                continue
            if dets[i]["array"] == dets[j]["array"]:
                sc = sequel_score(dets[i], dets[j], t, cfg)
            else:
                sc = link_score(dets[i], dets[j], t, cfg)
            if sc is not None:
                cands.append((sc, -abs(dets[i]["slowness"] - dets[j]["slowness"]), i, j))
    cands.sort(reverse=True)

    for _, _, i, j in cands:
        gi, gj = group[i], group[j]
        if gi is gj:
            continue
        if any(distinct_waves(dets[x], dets[y], t, cfg) for x in gi for y in gj):
            continue
        gi |= gj
        for k in gj:
            group[k] = gi

    seen, out = set(), []
    for g in group.values():
        if id(g) not in seen:
            seen.add(id(g))
            out.append(sorted(g))
    return out


def _overlap(a, b):
    """Time overlap of two detections [s] (negative = gap)."""
    return min(a["t_end_s"], b["t_end_s"]) - max(a["t_start_s"], b["t_start_s"])


def extend_trains(groups, dets, arrays, t, cfg):
    """
    Grow each train into neighbouring bands whose own ridge was too weak to be
    picked (or was cut short), when that band's data line up with the train.

    For a band member M of a train and an overlapping band B with no detection
    of its own at that time (in this train or any other), stack B at M's
    slowness over M's time span. B joins the train if that stack is coherent
    enough (mean semblance >= extend_semblance_min, mean beam RMS >= beam_rms_min)
    and its wave, moved to the shared latitudes, correlates with M's (the same
    test as link_score). Repeats until nothing is added, so a train can grow
    band by band. Added detections are marked extended=True.
    """
    bands = {name: a for name, a in arrays.items() if not a["is_full"]}
    owner = {}                                   # detection index -> group index
    for g, idx in enumerate(groups):
        for i in idx:
            owner[i] = g

    def band_busy(name, d):
        """Does band `name` already have a detection overlapping d in time?"""
        return any(dets[k]["array"] == name and _overlap(dets[k], d) > -cfg.link_time_tol_s
                   for k in owner)

    grown = True
    while grown:
        grown = False
        for g, idx in enumerate(groups):
            for i in list(idx):
                M = dets[i]
                if M["is_full"]:
                    continue
                for name, B in bands.items():
                    if name == M["array"] or B["lat_hi"] <= M["lat_lo"] or B["lat_lo"] >= M["lat_hi"]:
                        continue
                    if band_busy(name, M):
                        continue
                    k0 = int(np.searchsorted(t, M["t_start_s"]))
                    k1 = int(np.searchsorted(t, M["t_end_s"])) + 1
                    j  = int(np.argmin(np.abs(B["p"] - M["slowness"])))
                    sem = float(np.nanmean(B["S"][j, k0:k1]))
                    amp = float(np.nanmean(B["A"][j, k0:k1]))
                    if not (sem >= cfg.extend_semblance_min and amp >= cfg.beam_rms_min):
                        continue
                    cand = make_detection(B["X"], B["dist"], M["slowness"], k0, k1, t,
                                          B["lat_lo"], B["lat_hi"], False, sem, cfg, extended=True)
                    if link_score(M, cand, t, cfg) is None:
                        continue
                    dets.append(cand)
                    owner[len(dets) - 1] = g
                    idx.append(len(dets) - 1)
                    grown = True
    return groups


def _weighted_median(values, weights):
    order = np.argsort(values)
    cw    = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order][np.searchsorted(cw, 0.5 * cw[-1])])


def _weight(d):
    return d["semblance"] * max(d["t_end_s"] - d["t_start_s"], 1.0)


def _train_slowness(pool):
    """
    Slowness of a train and its spread: the semblance-weighted median of the ridge
    slownesses of the detections in `pool` (one sample per time step, so longer
    detections weigh more), with the weighted 16th and 84th percentiles. The
    spread mixes how well the ridge is located with any drift of the speed during
    the train. Returns (p, p_lo, p_hi) with p_lo <= p <= p_hi.
    """
    rp = np.concatenate([d["ridge_p"] for d in pool])
    rs = np.concatenate([d["ridge_s"] for d in pool])
    if rp.size == 0:                 # extended detections only: no ridge of their own
        p = _weighted_median([d["slowness"] for d in pool], [_weight(d) for d in pool])
        return p, p, p
    order = np.argsort(rp)
    cw    = np.cumsum(rs[order])
    lo, p, hi = (float(rp[order][min(np.searchsorted(cw, q * cw[-1]), rp.size - 1)])
                 for q in (0.16, 0.5, 0.84))
    return p, lo, hi


def classify(p, period, cfg):
    """'equatorward' / 'poleward' for an LSTID, None for anything else."""
    v = 1 / abs(p) if p != 0 else np.inf
    if not cfg.lstid_min_speed <= v <= cfg.lstid_max_speed:
        return None
    if not period >= cfg.lstid_min_period:          # NaN period -> not an LSTID
        return None
    return "equatorward" if p > 0 else "poleward"


CONFIDENCE_TIERS = ("low", "medium", "high")


def confidence_score(semblance, n_cycles, cfg):
    """
    Confidence of a train in 0..1: how coherent it is times how long it was tracked,

        score = (S - semblance_min) / (1 - semblance_min) * (1 - exp(-n_cycles / confidence_cycles))

    with S the train's semblance and n_cycles its duration in wave periods. Neither
    factor depends on the wave amplitude or on how much of the array the train
    covers, so a strong storm-time train confined to part of the aperture (source
    region inside the array) is not penalised. Against the published ROI catalogue,
    the HF-INT catalogue and Hp30 this ordering does better than the earlier rule
    (full-array detection = high), and latitudinal extent adds nothing to it.
    """
    coherence = np.clip((semblance - cfg.semblance_min) / (1 - cfg.semblance_min), 0, 1)
    return float(coherence * (1 - np.exp(-n_cycles / cfg.confidence_cycles)))


def train_confidence(score, speed, cfg):
    """
    Confidence tier of a train from its confidence_score:
      high    score >= confidence_high
      medium  score >= confidence_medium
      low     below
    then one tier down (high -> medium, medium -> low) if slower than
    confidence_min_speed: just above the 300 m/s LSTID limit, trains often sit in
    fields of slower medium-scale TIDs and were the doubtful cases by eye.
    The thresholds are where agreement with the published ROI catalogue (2021-2024)
    changes: below confidence_medium trains match it no more often than by chance,
    above confidence_high nearly three times as often. They are a convenience cut of
    the score, not a calibrated probability: that needs a blind audit by eye.
    """
    k = 2 if score >= cfg.confidence_high else 1 if score >= cfg.confidence_medium else 0
    if speed < cfg.confidence_min_speed:
        k = max(k - 1, 0)
    return CONFIDENCE_TIERS[k]


def summarize_train(members, cfg):
    """
    One record per train. Speed comes from the full aperture when it saw the
    train (best resolution), otherwise from the bands; speed_lo/speed_hi are the
    16th-84th percentile range of the ridge speed over the train (_train_slowness). The bands give a speed-vs-
    latitude profile, summarised by speed_ratio = fastest / slowest band, and a
    descriptive geometry flag:
      planar      band speeds within curved_speed_ratio of each other
      curved      band speeds differ by more (fronts bend with latitude)
      single band only one band saw it (latitude-confined)
    The threshold is descriptive, not a resolution limit: a 10° band cannot SEPARATE
    two waves 2x apart in speed, but it locates the speed of one wave far better.

    Bands added by extension (extended=True) count towards the latitude extent and
    n_bands, but not towards the speed or the speed profile: they were stacked at
    a neighbour's speed, so they carry no independent speed measurement.
    """
    full  = [d for d in members if d["is_full"]]
    bands = [d for d in members if not d["is_full"]]
    picked = [d for d in bands if not d["extended"]]
    pool  = full or picked or bands
    p, p_lo, p_hi = _train_slowness(pool)
    periods = [d["period_s"] for d in members if np.isfinite(d["period_s"])]
    period  = float(np.median(periods)) if periods else np.nan

    all_lat = {d["lat_c"] for d in bands}
    by_lat = {}
    for d in picked:
        by_lat.setdefault(d["lat_c"], []).append(d)
    prof_lat = sorted(by_lat)
    prof_p   = [_weighted_median([d["slowness"] for d in by_lat[la]],
                                 [_weight(d) for d in by_lat[la]]) for la in prof_lat]
    abs_p    = [abs(q) for q in prof_p]
    ratio    = max(abs_p) / min(abs_p) if len(abs_p) >= 2 and min(abs_p) > 0 else 1.0
    # the reliability flag counts only bands with their OWN ridge: extension shows
    # how far a train reaches, but a band stacked at a neighbour's speed is weaker
    # evidence that the train exists, so it must not turn 'single band' into planar
    geometry = ("single band" if len(prof_lat) <= 1 and not full
                else "curved" if ratio > cfg.curved_speed_ratio else "planar")

    ext = bands or full
    semblance = max(d["semblance"] for d in members)
    t_start   = min(d["t_start_s"] for d in members)
    t_end     = max(d["t_end_s"] for d in members)
    n_cycles  = (t_end - t_start) / period if period > 0 else 0.0   # NaN period -> 0
    score     = confidence_score(semblance, n_cycles, cfg)
    return {
        "t_start_s":       t_start,
        "t_end_s":         t_end,
        "slowness":        p,
        "speed_m_s":       1 / abs(p) if p != 0 else np.inf,
        "speed_lo_m_s":    1 / max(abs(p_lo), abs(p_hi)) if p != 0 else np.inf,
        "speed_hi_m_s":    1 / min(abs(p_lo), abs(p_hi)) if p != 0 else np.inf,
        "period_s":        period,
        "wavelength_km":   (1 / abs(p)) * period / 1000 if p != 0 else np.inf,
        "amplitude":       max(d["amplitude"] for d in members),
        "semblance":       semblance,
        "n_cycles":        n_cycles,
        "confidence_score": score,
        "confidence":      train_confidence(score, 1 / abs(p) if p != 0 else np.inf, cfg),
        "lat_lo":          min(d["lat_lo"] for d in ext),
        "lat_hi":          max(d["lat_hi"] for d in ext),
        "full_array":      bool(full),
        "n_bands":         len(all_lat),
        "n_bands_picked":  len(prof_lat),
        "geometry":        geometry,
        "speed_ratio":     ratio,
        "profile_lat":     prof_lat,
        "profile_speed":   [float(np.sign(q) / abs(q)) if q != 0 else np.inf for q in prof_p],
        "direction":       classify(p, period, cfg),
    }


def _is_single_band(members):
    """No full-aperture detection and own ridges in at most one band."""
    if any(d["is_full"] for d in members):
        return False
    return len({d["array"] for d in members if not d["extended"]}) <= 1


def merge_simultaneous(kept, dets, cfg):
    """
    Merge LSTID trains of the same direction that overlap in time into one train:
    over this array two simultaneous LSTID trains are taken to be one disturbance
    (e.g. a front fast in the north and slow in the south, whose parts the linking
    can keep apart). Overlaps chain, so A-B and B-C give one train. `kept` holds
    (summary, detection indices); a merged train is summarised again from all its
    detections and keeps the direction of its parts even if its speed falls just
    outside the LSTID range.
    """
    out = []
    for tr, idx in sorted(kept, key=lambda x: x[0]["t_start_s"]):
        k = next((j for j in range(len(out) - 1, -1, -1)
                  if out[j][0]["direction"] == tr["direction"]), None)
        if k is not None and tr["t_start_s"] < out[k][0]["t_end_s"]:
            merged = out[k][1] + idx
            s = summarize_train([dets[i] for i in merged], cfg)
            s["direction"] = tr["direction"]
            out[k] = (s, merged)
        else:
            out.append((tr, idx))
    return out


def build_trains(dets, arrays, t, cfg):
    """Link, extend into neighbouring bands, summarise, keep LSTID trains, merge
    those that overlap in time (merge_simultaneous), and keep the trains that
    overlap day D. A train whose midpoint is in D is 'owned' by D; one
    reaching into the padding with its midpoint outside D is not, and is kept
    only for resolve_day_boundaries to settle against the neighbouring day (the
    two days see different, padding-truncated parts of a train that crosses
    midnight, so each may link it differently). Extension appends to `dets`.
    Returns (trains, member lists) with trains numbered 1.. in time order."""
    groups = link_trains(dets, t, cfg)
    if not cfg.keep_single_band:
        # Drop single-band groups BEFORE extension: two blind audits found most
        # of them are not LSTIDs by eye (~36-43% confirmed vs ~70% for multi-band
        # trains). Dropping them here also frees their band, so a genuine
        # neighbouring train can extend into it.
        groups = [g for g in groups if not _is_single_band([dets[i] for i in g])]
    groups = extend_trains(groups, dets, arrays, t, cfg)
    kept = []
    for idx in groups:
        tr = summarize_train([dets[i] for i in idx], cfg)
        if tr["direction"] in cfg.directions:          # None = not an LSTID
            kept.append((tr, idx))
    if cfg.merge_simultaneous:
        kept = merge_simultaneous(kept, dets, cfg)
    out = []
    for tr, idx in kept:
        if tr["t_end_s"] <= 0 or tr["t_start_s"] >= SECONDS_PER_DAY:
            continue                                   # entirely in the padding
        tr["owned"] = bool(0 <= 0.5 * (tr["t_start_s"] + tr["t_end_s"]) < SECONDS_PER_DAY)
        out.append((tr, idx))
    out.sort(key=lambda x: x[0]["t_start_s"])
    for k, (tr, idx) in enumerate(out, 1):
        tr["train_id"] = k
        for i in idx:
            dets[i]["train_id"] = k
    return [tr for tr, _ in out], [idx for _, idx in out]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Plot
# ══════════════════════════════════════════════════════════════════════════════

SPEED_TICKS = [150, 200, 300, 500, 1000, 2000]
ARROW = {"equatorward": "↓", "poleward": "↑"}


def _slowness_ticks(cfg):
    ticks = [0.0] + [s / v for v in SPEED_TICKS for s in (1, -1) if 1 / v <= cfg.max_slowness]
    ticks = sorted(ticks)
    return ([p * 1e3 for p in ticks],
            ["∞" if p == 0 else f"{'+' if p > 0 else '−'}{1 / abs(p):.0f}" for p in ticks])


def _speed_axis(ax, cfg, axis="y"):
    """Label a slowness axis in m/s speeds; + = equatorward, - = poleward."""
    pos, lbl = _slowness_ticks(cfg)
    if axis == "y":
        ax.set_yticks(pos); ax.set_yticklabels(lbl, fontsize=8)
        ax.set_ylabel("speed [m/s]\n+ equatorward", fontsize=9)
        line = ax.axhline
    else:
        ax.set_xticks(pos); ax.set_xticklabels(lbl, fontsize=8)
        ax.set_xlabel("speed [m/s]   (+ equatorward, − poleward)", fontsize=9)
        line = ax.axvline
    for v in (cfg.lstid_min_speed, cfg.lstid_max_speed):
        for s in (1, -1):
            line(s * 1e3 / v, color="0.35", lw=0.8, ls="--")


def _draw_keogram(ax, X, lats, t, title):
    ax.imshow(X, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto",
              extent=[t[0] / 3600, t[-1] / 3600, lats[-1], lats[0]], origin="upper",
              interpolation="nearest")
    ax.set_ylabel("latitude [°N]", fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")


def _owned_slice(d, lat_min, lat_max, cfg):
    """Latitude slice a band detection is drawn over: its centre ± half a step,
    stretched to the array edge for the outermost bands. Adjacent bands thus
    tile the latitude axis and a curved front is drawn piecewise without overlap."""
    if d["is_full"]:
        return d["lat_lo"], d["lat_hi"]
    h  = cfg.band_step_deg / 2
    lo = lat_min if d["lat_lo"] <= lat_min + 1e-6 else d["lat_c"] - h
    hi = lat_max if d["lat_hi"] >= lat_max - 1e-6 else d["lat_c"] + h
    return lo, hi


def _draw_train(ax, tr, members, lats, cfg):
    """Draw a train's crest (solid) and trough (dashed) lines. A multi-band train
    is drawn piecewise over each band's owned slice; a single-band train over its
    whole band, boxed with a dotted outline so it stands out for inspection."""
    bands  = [d for d in members if not d["is_full"]] or members
    single = tr["geometry"] == "single band"
    for d in bands:
        lo, hi = (d["lat_lo"], d["lat_hi"]) if single else _owned_slice(d, lats.min(), lats.max(), cfg)
        la = np.linspace(lo, hi, 8)
        dd = (la - d["lat_c"]) * M_PER_DEG
        for tc, ls in [(c, "-") for c in d["crests_s"]] + [(c, "--") for c in d["troughs_s"]]:
            ax.plot((tc - dd * d["slowness"]) / 3600, la, color="k", lw=1.2, ls=ls)
    if single:
        ax.add_patch(Rectangle((tr["t_start_s"] / 3600, tr["lat_lo"]),
                               (tr["t_end_s"] - tr["t_start_s"]) / 3600,
                               tr["lat_hi"] - tr["lat_lo"],
                               fill=False, ec="k", lw=1, ls=":"))
    tm = 0.5 * (tr["t_start_s"] + tr["t_end_s"]) / 3600
    ax.text(tm, tr["lat_hi"] - 0.3,
            f"#{tr['train_id']} {ARROW[tr['direction']]} {tr['speed_m_s']:.0f} m/s · "
            f"{tr['period_s'] / 60:.0f} min · {tr['confidence']}" + (" · 1 band" if single else ""),
            fontsize=8, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def _draw_vespa(ax, r, t, cfg, title, ylabels=True):
    shown = np.where(np.nan_to_num(r["A"]) >= cfg.beam_rms_min, r["S"], np.nan)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower",
                   extent=[t[0] / 3600, t[-1] / 3600, r["p"][0] * 1e3, r["p"][-1] * 1e3],
                   interpolation="nearest")
    ax.set_facecolor("0.93")
    for d in r["dets"]:
        kept = "train_id" in d
        ax.scatter(d["ridge_t"] / 3600, d["ridge_p"] * 1e3, lw=0,
                   s=3 if kept else 2, c="k" if kept else "0.6")
    _speed_axis(ax, cfg)
    if not ylabels:
        ax.set_yticklabels([]); ax.set_ylabel("")
    ax.set_title(title, fontsize=10, loc="left")
    return im


def plot(X, lats, t, results, dets, trains, members, cfg):
    full, bands = results[0], results[1:]
    nb  = len(bands)
    fig = plt.figure(figsize=(18, 21))
    gs  = gridspec.GridSpec(4, nb, height_ratios=[1.1, 0.8, 0.9, 0.9], hspace=0.38, wspace=0.08)

    ax_k = fig.add_subplot(gs[0, :])
    _draw_keogram(ax_k, X, lats, t, f"(a) band-passed keogram and LSTID trains — {cfg.input_date}   "
                                    "(↓ equatorward, ↑ poleward; solid = crest, dashed = trough)")
    for tr, idx in zip(trains, members):
        _draw_train(ax_k, tr, [dets[i] for i in idx], lats, cfg)

    ax = fig.add_subplot(gs[1, :], sharex=ax_k)
    im = _draw_vespa(ax, full, t, cfg, f"(b) vespagram, full array {full['lat_lo']:g}–{full['lat_hi']:g}°N"
                     f"   (black = ridges in an LSTID train, grey = rejected; "
                     f"background hidden where beam RMS < {cfg.beam_rms_min} TECu)")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02).set_label("semblance", fontsize=9)

    for i, r in enumerate(bands):
        ax = fig.add_subplot(gs[2, i], sharex=ax_k)
        _draw_vespa(ax, r, t, cfg, f"({chr(ord('c') + i)}) {r['lat_lo']:g}–{r['lat_hi']:g}°N",
                    ylabels=(i == 0))

    # speed-vs-latitude profile per train (signed speed; band values outside the
    # plotted range, e.g. near-vertical fronts, are pinned to the edge)
    ax  = fig.add_subplot(gs[3, :2])
    lim = 2000
    for tr in trains:
        if not tr["profile_lat"]:
            continue
        x = np.clip(tr["profile_speed"], -lim, lim)
        ax.plot(x, tr["profile_lat"], "o-", color="k", lw=1.2, ms=4)
        ax.annotate(f"#{tr['train_id']}", (x[-1], tr["profile_lat"][-1]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    for v in (cfg.lstid_min_speed, cfg.lstid_max_speed):
        for sgn in (1, -1):
            ax.axvline(sgn * v, color="0.35", lw=0.8, ls="--")
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel("speed [m/s]   (+ equatorward, − poleward)", fontsize=9)
    ax.set_ylabel("band centre [°N]", fontsize=9)
    ax.set_title(f"({chr(ord('c') + nb)}) speed vs latitude (one point per band)", fontsize=10, loc="left")

    # train table
    ax = fig.add_subplot(gs[3, 2:]); ax.axis("off")
    hdr = f"{'#':>2}  {'time UT':11s} {'dir':3s} {'speed':>6s} {'period':>6s} {'λ km':>6s} {'amp':>6s} {'lat °N':11s} {'conf':6s} geometry"
    rows = [hdr, "─" * len(hdr)]
    for tr in trains:
        h0, h1 = tr["t_start_s"] / 3600, tr["t_end_s"] / 3600
        rows.append(f"{tr['train_id']:>2}  {int(h0):02d}:{int(h0 % 1 * 60):02d}-{int(h1) % 24:02d}:{int(h1 % 1 * 60):02d} "
                    f"{ARROW[tr['direction']]:>3s} {tr['speed_m_s']:6.0f} {tr['period_s'] / 60:6.0f} "
                    f"{tr['wavelength_km']:6.0f} {tr['amplitude']:6.3f} "
                    f"{tr['lat_lo']:4.1f}-{tr['lat_hi']:4.1f}  {tr['confidence']:6s} {tr['geometry']}")
    ax.text(0, 1, "\n".join(rows) if trains else "no LSTID trains", family="monospace",
            fontsize=9, va="top")
    ax.set_title(f"({chr(ord('d') + nb)}) LSTID trains", fontsize=10, loc="left")

    for a in fig.axes:
        if a.get_label() == "<colorbar>" or not a.axison:
            continue
        if a.get_xlabel().startswith("speed"):
            continue
        a.set_xlim(0, 24)
        a.set_xticks(range(0, 25, 4 if a.get_subplotspec().colspan.stop -
                                     a.get_subplotspec().colspan.start == 1 else 2))
        a.set_xlabel("time [h UT]", fontsize=9)

    out = cfg.out_dir / f"{cfg.input_date}_vespa.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved -> %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_COLS = ["train_id", "t_start", "t_end", "direction", "speed_m_s", "speed_lo_m_s", "speed_hi_m_s", "period_min",
              "wavelength_km", "amplitude", "semblance", "n_cycles", "confidence_score", "confidence",
              "lat_lo", "lat_hi", "geometry", "speed_ratio", "full_array", "n_bands"]


def detect(cfg):
    """
    Run steps 1-6 for one day. Returns (X, lats, t, results, dets, trains,
    members): results holds one dict per array (its vespagram and detections),
    dets all detections, trains the LSTID trains and members their detection
    indices.
    """
    X, lats, t = load_and_prepare(cfg)
    return (X, lats, t, *analyse(X, lats, t, cfg))


def analyse(X, lats, t, cfg):
    """
    Steps 3-6 on a prepared (band-passed, decimated) keogram — separated from
    loading so synthetic keograms can be tested. Returns (results, dets, trains,
    members).
    """
    dt_s   = t[1] - t[0]
    p_grid = np.linspace(-cfg.max_slowness, cfg.max_slowness, cfg.n_slowness)

    results, dets, arrays = [], [], {}
    for lat_lo, lat_hi, is_full in array_list(lats, cfg):
        rows  = (lats >= lat_lo - 1e-6) & (lats <= lat_hi + 1e-6)
        Xa    = X[rows]
        dist  = meridian_distance(lats[rows], 0.5 * (lat_lo + lat_hi), cfg.longitude)
        S, A  = slant_stack(Xa, dist, p_grid, dt_s, cfg)
        ds    = pick_detections(S, A, Xa, dist, p_grid, t, lat_lo, lat_hi, is_full, cfg)
        r     = {"lat_lo": lat_lo, "lat_hi": lat_hi, "is_full": is_full, "S": S, "A": A,
                 "p": p_grid, "X": Xa, "dist": dist, "dets": ds}
        results.append(r)
        arrays[f"{lat_lo:g}-{lat_hi:g}"] = r
        dets += ds

    trains, members = build_trains(dets, arrays, t, cfg)
    return results, dets, trains, members


def _abs_time(date, s):
    return dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=s)


def train_records(trains, date):
    return [{"date": date, **{k: v for k, v in tr.items()
                              if k not in ("t_start_s", "t_end_s", "period_s")},
             "t_start":    _abs_time(date, tr["t_start_s"]),
             "t_end":      _abs_time(date, tr["t_end_s"]),
             "period_min": tr["period_s"] / 60}
            for tr in trains]


def detection_records(dets, date):
    """Detections that belong to an LSTID train (ridge pixel arrays dropped)."""
    return [{"date": date, "train_id": d["train_id"],
             **{k: d[k] for k in ("array", "is_full", "lat_lo", "lat_hi", "slowness",
                                   "speed_m_s", "amplitude", "semblance", "n_extrema",
                                   "extended", "crests_s", "troughs_s")},
             "t_start":    _abs_time(date, d["t_start_s"]),
             "t_end":      _abs_time(date, d["t_end_s"]),
             "period_min": d["period_s"] / 60}
            for d in dets if "train_id" in d]


def resolve_day_boundaries(trains, dets, tol=dt.timedelta(minutes=15)):
    """
    Settle trains that cross midnight, so each wave is catalogued exactly once.
    With the long padding the day that owns a train (midpoint inside it) sees it
    whole, so the two days normally agree and the owned view is the record.
    Trains owned by their day are kept. A non-owned train is dropped if it
    overlaps in time an owned train of the neighbouring day (that day saw the
    wave and kept it); otherwise it is the only record of the wave and is kept,
    and among overlapping non-owned trains of the two days only the longest
    survives.

    The two days can still link a wave differently (linking depends on what else
    each view contains), each owning its own version. Two owned trains of
    adjacent days that overlap in time are such a pair when neither has a twin
    (same start and end within tol) in the other day's view; distinct waves
    that merely overlap are seen by both days and do have twins. Of such a pair
    only the longer version is kept. Returns (trains, dets) with the detections
    of dropped trains removed.
    """
    span = lambda r: (r["t_start"], r["t_end"])
    over = lambda a, b: span(a)[0] < span(b)[1] and span(b)[0] < span(a)[1]
    by_day = {}
    for r in trains:
        by_day.setdefault(r["date"], []).append(r)
    near = lambda r: [o for dd in (-1, 1)
                      for o in by_day.get(r["date"] + dt.timedelta(days=dd), [])]

    # pass 1: non-owned trains overlapping an owned train of the neighbouring day go
    orphans = [r for r in trains if not r["owned"]
               and not any(o["owned"] and over(r, o) for o in near(r))]
    # pass 2: among the surviving non-owned trains, overlapping ones keep the longest
    dur  = lambda x: (x["t_end"] - x["t_start"], x["date"])       # date breaks ties
    keep = {id(r) for r in trains if r["owned"]}
    for r in orphans:
        if all(dur(r) > dur(o) for o in orphans if o is not r and over(r, o)
               and o["date"] != r["date"]):
            keep.add(id(r))

    # pass 3: one wave owned by both days in two different versions -> keep the longer
    twin = lambda r, day: any(abs(o["t_start"] - r["t_start"]) <= tol
                              and abs(o["t_end"] - r["t_end"]) <= tol for o in by_day.get(day, []))
    for r in trains:
        if not r["owned"]:
            continue
        for o in by_day.get(r["date"] + dt.timedelta(days=1), []):
            if (o["owned"] and over(r, o) and id(r) in keep and id(o) in keep
                    and not twin(r, o["date"]) and not twin(o, r["date"])):
                keep.discard(id(min(r, o, key=dur)))

    kept = [r for r in trains if id(r) in keep]
    ids  = {(r["date"], r["train_id"]) for r in kept}
    return kept, [d for d in dets if (d["date"], d["train_id"]) in ids]


def run(cfg, do_plot=True):
    """Detect one day; with do_plot also write the figure and a per-day train CSV.
    Returns (train records, detection records)."""
    t_start = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    X, lats, t, results, dets, trains, members = detect(cfg)
    t_detect = time.time() - t_start

    tr_rec = train_records(trains, cfg.input_date)
    if do_plot:
        if tr_rec:
            pl.DataFrame(tr_rec).select(TRAIN_COLS).write_csv(
                cfg.out_dir / f"{cfg.input_date}_trains.csv")
        plot(X, lats, t, results, dets, trains, members, cfg)
    logging.info("%s: %d detections -> %d LSTID trains, detect %.2fs, total %.2fs",
                 cfg.input_date, len(dets), len(trains), t_detect, time.time() - t_start)
    return tr_rec, detection_records(dets, cfg.input_date)


def _process_date(job):
    """Picklable worker: never raises, so one bad grid cannot stop the batch."""
    cfg, do_plot = job
    try:
        return cfg.input_date, *run(cfg, do_plot), None
    except Exception as e:
        return cfg.input_date, [], [], f"{type(e).__name__}: {e}"


def _write_table(rows, folder, name):
    """Write records as <name>.parquet and, for readers without a parquet library,
    as <name>.csv with each list column as values separated by spaces."""
    df = pl.DataFrame(rows)
    df.write_parquet(folder / f"{name}.parquet")
    lists = [c for c, t in df.schema.items() if isinstance(t, pl.List)]
    df.with_columns(pl.col(c).cast(pl.List(pl.String)).list.join(" ") for c in lists) \
      .write_csv(folder / f"{name}.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Vespagram (slant-stack) LSTID detection.")
    parser.add_argument("dates", nargs="*", default=[str(VespaConfig.input_date)],
                        help="one or more selectors: YYYY, YYYY-MM or YYYY-MM-DD "
                             "(use 'all' for every grid)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip figures and per-day CSVs; only write the catalogue")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.workers == 1 else logging.ERROR,
                        format="%(message)s")
    base  = VespaConfig()
    dates = sorted({d for sel in args.dates
                    for d in discover_dates(base.keo_folder, None if sel == "all" else sel)})
    if not dates:
        print(f"No keogram grids match {args.dates} in {base.keo_folder}")
        return 1
    base.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(replace(base, input_date=d), not args.no_plot) for d in dates]
    trains, dets, failures = [], [], []
    t0 = time.time()
    if args.workers == 1:
        outs = map(_process_date, jobs)
    else:
        ex   = ProcessPoolExecutor(max_workers=args.workers)
        outs = (f.result() for f in as_completed([ex.submit(_process_date, j) for j in jobs]))
    for date, tr, de, err in tqdm(outs, total=len(jobs), unit="day", disable=args.workers == 1):
        trains.extend(tr)
        dets.extend(de)
        if err:
            failures.append((date, err))

    n_raw, n_owned = len(trains), sum(r["owned"] for r in trains)
    trains, dets = resolve_day_boundaries(trains, dets)
    print(f"Day boundaries: {n_raw - len(trains)} duplicate views of midnight-crossing "
          f"trains removed ({n_owned - sum(r['owned'] for r in trains)} of them a second "
          f"version owned by the neighbouring day), {sum(not r['owned'] for r in trains)} "
          f"kept from the neighbouring day")

    if len(dates) > 1 or args.no_plot:
        # only a full run without failed days writes the main catalogue; partial
        # runs get their own file so they can never overwrite it
        tag = "" if args.dates == ["all"] else "_" + "_".join(args.dates)
        tag += "_partial" if failures else ""
        # days finish in any order: sort, so that two builds give the same files
        trains.sort(key=lambda r: (r["t_start"], r["date"]))
        dets.sort(key=lambda d: (d["date"], d["train_id"], d["t_start"], d["array"]))
        _write_table(trains, base.out_dir, f"trains{tag}")
        _write_table(dets, base.out_dir, f"detections{tag}")
        print(f"Saved {len(trains)} LSTID trains ({len(dets)} detections) from "
              f"{len(dates) - len(failures)} days -> {base.out_dir}")
    print(f"Done in {time.time() - t0:.1f}s, {len(failures)} failed")
    for d, e in failures:
        print(f"  failed: {d}  ({e})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
