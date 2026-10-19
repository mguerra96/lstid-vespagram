"""
vespagram.py

LSTID detection and characterisation by slant-stacking the keogram, treating each
latitude row as one sensor of a north-south array (the seismological vespagram /
semblance method).

Pipeline:
  1. Load keogram grid (reuses detect_roi.load_grid, padded with neighbour days)
  2. Decimate in latitude and time, band-pass each row to the LSTID period band
  3. For three 10-degree sub-arrays (37.5-47.5, 42.5-52.5, 47.5-57.5 N) and for
     each trial slowness p: delay-and-sum the rows -> semblance S(t, p) (coherence,
     0..1, amplitude-independent) and beam RMS A(t, p). Only equatorward
     slownesses of the LSTID window are searched, plus a margin of three grid
     steps on each side so that a ridge at the edge of the window can be told
     from the flank of a wave outside it
  4. Pick ridges of S along p inside the LSTID window (300-1500 m/s); each
     connected ridge in one sub-array is a DETECTION (speed, span, beam
     crests/troughs, period, amplitude)
  5. Move every detection's times to the central latitude (47.5 N) along its own
     slowness. An LSTID TRAIN (an event, a period of LSTID activity) is a run of
     detections of any sub-array less than event_gap_s apart, with at least one
     crest and one trough. Its speed is the semblance-weighted mean slowness of all
     its ridge pixels; its confidence score is coherence x array coverage.
  6. Plot and save detections + trains

Sign convention: p > 0 is EQUATORWARD propagation (the north of the array sees
the front first). Speed = 1 / |p|. Only equatorward trains are catalogued.
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
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, binary_dilation
from scipy.signal import butter, sosfiltfilt, find_peaks
from skimage.measure import label

import paths
from detect_roi import Config as RoiConfig, load_grid, discover_dates, SECONDS_PER_DAY, M_PER_DEG


# ── Configuration ───────────────────────────────────────────────────────────────

@dataclass
class VespaConfig:
    # ── Paths & date ──
    keo_folder: Path = paths.KEO_DIR
    out_dir:    Path = paths.ROOT / "results/vespagram"
    input_date: dt.date = dt.date(2015, 1, 7)
    longitude:  float = 17.5        # central longitude of the keogram (not used by the detector)
    border_pad_hours: float = 12    # hours of the neighbouring days read on each side: long
                                    # enough for the day that owns a midnight-crossing train
                                    # (midpoint inside it) to see the whole train

    # ── Pre-processing ──
    lat_step_deg: float = 0.25    # latitude spacing of the decimated array [deg]
    time_step_s:  int   = 120     # time cadence after decimation [s]
    band_min_period_s: float = 25 * 60    # band-pass short-period corner [s]
    band_max_period_s: float = 200 * 60   # band-pass long-period corner [s]

    # ── Sub-arrays ──
    band_width_deg: float = 10     # sub-array aperture [deg]
    band_step_deg:  float = 5      # step between sub-arrays [deg]: three, half overlapping

    # ── Slant stack ──
    # equatorward LSTID window 1/1500..1/300 s/m plus a few steps on each side: a
    # ridge pixel must beat both neighbours along p, so with a step beyond the edge
    # the rising flank of a slower wave or of a simultaneous structure is never
    # taken for a ridge at 300 or 1500 m/s
    slowness_step:   float = 1 / 14400   # [s/m] (0.07 s/km; refined by a parabola)
    slowness_margin_steps: int = 3
    semblance_window_s: float = 30 * 60   # boxcar window for semblance / beam RMS [s]
    min_valid_rows_frac: float = 0.7      # need this fraction of rows inside the data

    # ── Detection ──
    semblance_min: float = 0.6     # ridge must be at least this coherent [-]
    beam_rms_min:  float = 0.01    # and at least this strong [TECu]
    min_duration_s: float = 45 * 60
    lstid_min_speed: float = 300   # [m/s]
    lstid_max_speed: float = 1500  # [m/s]

    # ── Trains (events) ──
    event_gap_s:      float = 60 * 60  # detections closer than this (at 47.5 N) are one train
    default_period_s: float = 60 * 60  # used when no detection has a measurable period
    # score thresholds of the 'medium' and 'high' tiers [-]: below 0.20 trains coincide
    # with the published ROI catalogue (2021-2024) no more often than by chance; from
    # 0.35 the agreement stops growing (about 2.5x chance; southward HF-INT events
    # 2015-2025: about 1.7x)
    confidence_medium: float = 0.20
    confidence_high:   float = 0.35

    @property
    def p_grid(self) -> np.ndarray:
        """Trial slownesses [s/m]: the equatorward LSTID window plus the margins."""
        margin = self.slowness_margin_steps * self.slowness_step
        lo = 1 / self.lstid_max_speed - margin
        hi = 1 / self.lstid_min_speed + margin
        return np.arange(lo, hi + self.slowness_step / 2, self.slowness_step)

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

def meridian_distance(lats, lat_ref):
    """Signed along-meridian distance of each latitude from lat_ref [m], north > 0."""
    return (np.asarray(lats, float) - lat_ref) * M_PER_DEG


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

def array_list(lats, cfg):
    """(lat_lo, lat_hi) of each sub-array."""
    lo, hi = float(lats.min()), float(lats.max())
    starts = np.arange(lo, hi - cfg.band_width_deg + 1e-6, cfg.band_step_deg)
    return [(float(s), float(s + cfg.band_width_deg)) for s in starts]


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


def pick_detections(S, A, X, dist, p_grid, t, lat_lo, lat_hi, cfg):
    """
    Ridges = pixels inside the LSTID window that are a local maximum of S along p
    and pass the semblance and beam-amplitude thresholds. Ridges are joined
    across time (allowing one slowness bin of jitter) and each connected ridge
    becomes one detection, at the median of its sub-grid ridge slownesses
    (refine_slowness). Detections in the day padding are kept: they may belong
    to a day-D train.
    """
    Sf   = np.nan_to_num(S, nan=0.0)
    up   = np.vstack([Sf[1:], np.full((1, Sf.shape[1]), -np.inf)])
    down = np.vstack([np.full((1, Sf.shape[1]), -np.inf), Sf[:-1]])
    in_range = ((p_grid >= 1 / cfg.lstid_max_speed - 1e-12)
                & (p_grid <= 1 / cfg.lstid_min_speed + 1e-12))[:, None]
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
                                   t, lat_lo, lat_hi, float(np.mean(Sf[jj, kk])), cfg,
                                   ridge_t=t[kk], ridge_p=p_ref, ridge_s=Sf[jj, kk]))
    return dets


def make_detection(X, dist, p, k0, k1, t, lat_lo, lat_hi, semblance, cfg,
                   ridge_t, ridge_p, ridge_s):
    """
    Describe one detection: the array's beam at slowness p over samples [k0, k1),
    its crests and troughs, period and amplitude. Times are those at the
    sub-array centre.
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
        "ridge_t":    ridge_t,
        "ridge_p":    ridge_p,
        "ridge_s":    ridge_s,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Trains: runs of detections of any sub-array
# ══════════════════════════════════════════════════════════════════════════════

def to_ref(det, s, ref_lat):
    """Time s at the detection's sub-array centre moved to ref_lat along the
    detection's slowness (an equatorward front reaches the north first)."""
    return s + (det["lat_c"] - ref_lat) * M_PER_DEG * det["slowness"]


def group_detections(dets, cfg):
    """Detections sorted by start (at the reference latitude); one starting less
    than event_gap_s after the end of the current group joins it. Returns lists
    of detection indices."""
    groups, end = [], -np.inf
    for i in sorted(range(len(dets)), key=lambda i: dets[i]["t0_ref"]):
        if groups and dets[i]["t0_ref"] - end < cfg.event_gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
            end = -np.inf
        end = max(end, dets[i]["t1_ref"])
    return groups


def count_distinct(times, tol):
    """Number of distinct fronts among `times`: those closer than tol (the same
    front seen by overlapping sub-arrays) count once."""
    times = np.sort(np.asarray(times, float))
    return 0 if times.size == 0 else int(1 + np.sum(np.diff(times) > tol))


def coverage(members, t0, t1, n_arrays, dt_s):
    """Mean over [t0, t1] of the share of sub-arrays with a detection running."""
    grid   = np.arange(t0, t1 + dt_s / 2, dt_s)
    active = np.zeros(grid.size)
    for name in {d["array"] for d in members}:
        on = np.zeros(grid.size, bool)
        for d in members:
            if d["array"] == name:
                on |= (grid >= d["t0_ref"]) & (grid <= d["t1_ref"])
        active += on
    return float(active.mean() / n_arrays)


CONFIDENCE_TIERS = ("low", "medium", "high")


def confidence_score(semblance, cover, cfg):
    """
    Confidence of a train in 0..1: how coherent its ridges are times how much of
    the array saw it,

        score = (S - semblance_min) / (1 - semblance_min) * coverage

    with S the mean semblance of the train's ridge pixels and coverage the mean
    share of sub-arrays with a running detection. Neither factor depends on the
    wave amplitude (which follows the solar cycle) or on the train's duration (a
    single clear wave crossing the whole array scores as high as a long train).
    """
    coherence = np.clip((semblance - cfg.semblance_min) / (1 - cfg.semblance_min), 0, 1)
    return float(coherence * cover)


def train_confidence(score, cfg):
    """Tier of a confidence score: high >= confidence_high, medium >=
    confidence_medium, low below. A convenience cut, not a probability."""
    return CONFIDENCE_TIERS[2 if score >= cfg.confidence_high
                            else 1 if score >= cfg.confidence_medium else 0]


def summarize_train(members, n_arrays, dt_s, cfg):
    """
    One record per train, from all its detections (times at the reference
    latitude). Speed = 1 / mean slowness of all ridge pixels weighted by their
    semblance: a longer ridge, or one seen by more sub-arrays, has more pixels and
    weighs more. speed_lo / speed_hi: the weighted 16th-84th percentile range of
    the same pixels. Fronts seen by several sub-arrays are counted once.
    """
    rp = np.concatenate([d["ridge_p"] for d in members])
    rs = np.concatenate([d["ridge_s"] for d in members])
    p  = float(np.average(rp, weights=rs))
    order = np.argsort(rp)
    cw = np.cumsum(rs[order])
    p_lo, p_hi = (float(rp[order][min(np.searchsorted(cw, q * cw[-1]), rp.size - 1)])
                  for q in (0.16, 0.84))
    periods = [d["period_s"] for d in members if np.isfinite(d["period_s"])]
    period  = float(np.median(periods)) if periods else np.nan
    t0 = min(d["t0_ref"] for d in members)
    t1 = max(d["t1_ref"] for d in members)
    tol = (period if np.isfinite(period) else cfg.default_period_s) / 4
    crests  = [c for d in members for c in d["crests_ref"]]
    troughs = [c for d in members for c in d["troughs_ref"]]
    semblance = float(rs.mean())
    cover = coverage(members, t0, t1, n_arrays, dt_s)
    score = confidence_score(semblance, cover, cfg)
    return {
        "t_start_s":        t0,
        "t_end_s":          t1,
        "slowness":         p,
        "speed_m_s":        1 / p,
        "speed_lo_m_s":     1 / p_hi,
        "speed_hi_m_s":     1 / p_lo,
        "period_s":         period,
        "wavelength_km":    period / p / 1000,
        "amplitude":        max(d["amplitude"] for d in members),
        "semblance":        semblance,
        "coverage":         cover,
        "n_detections":     len(members),
        "n_crests":         count_distinct(crests, tol),
        "n_troughs":        count_distinct(troughs, tol),
        "n_cycles":         (t1 - t0) / period if period > 0 else 0.0,   # NaN period -> 0
        "confidence_score": score,
        "confidence":       train_confidence(score, cfg),
    }


def build_trains(dets, n_arrays, ref_lat, t, cfg):
    """Move the detections to ref_lat, group them (group_detections), summarise
    each group, keep those with at least one crest and one trough and whose
    midpoint is in day D (a train crossing midnight belongs to one day only).
    Returns (trains, member lists) with trains numbered 1.. in time order."""
    for d in dets:
        d["t0_ref"]      = to_ref(d, d["t_start_s"], ref_lat)
        d["t1_ref"]      = to_ref(d, d["t_end_s"], ref_lat)
        d["crests_ref"]  = [to_ref(d, c, ref_lat) for c in d["crests_s"]]
        d["troughs_ref"] = [to_ref(d, c, ref_lat) for c in d["troughs_s"]]
    out = []
    for idx in group_detections(dets, cfg):
        tr = summarize_train([dets[i] for i in idx], n_arrays, t[1] - t[0], cfg)
        if tr["n_crests"] < 1 or tr["n_troughs"] < 1:
            continue
        if not 0 <= 0.5 * (tr["t_start_s"] + tr["t_end_s"]) < SECONDS_PER_DAY:
            continue
        out.append((tr, idx))
    out.sort(key=lambda x: x[0]["t_start_s"])
    for k, (tr, idx) in enumerate(out, 1):
        tr["train_id"] = k
        for i in idx:
            dets[i]["train_id"] = k
    return [tr for tr, _ in out], [idx for _, idx in out]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Plot
# ══════════════════════════════════════════════════════════════════════════════

SPEED_TICKS = [300, 400, 500, 700, 1000, 1500]


def _speed_axis(ax, cfg):
    """Label the slowness axis [s/km] in m/s speeds; dashed = LSTID window."""
    ax.set_yticks([1e3 / v for v in SPEED_TICKS])
    ax.set_yticklabels([str(v) for v in SPEED_TICKS], fontsize=8)
    ax.set_ylabel("speed [m/s]", fontsize=9)
    for v in (cfg.lstid_min_speed, cfg.lstid_max_speed):
        ax.axhline(1e3 / v, color="0.35", lw=0.8, ls="--")


def _draw_keogram(ax, X, lats, t, title):
    ax.imshow(X, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto",
              extent=[t[0] / 3600, t[-1] / 3600, lats[-1], lats[0]], origin="upper",
              interpolation="nearest")
    ax.set_ylabel("latitude [°N]", fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")


def _owned_slice(d, lat_min, lat_max, cfg):
    """Latitude slice a sub-array detection is drawn over: its centre ± half a
    step, stretched to the array edge for the outermost sub-arrays, so the
    sub-arrays tile the latitude axis without overlap."""
    h  = cfg.band_step_deg / 2
    lo = lat_min if d["lat_lo"] <= lat_min + 1e-6 else d["lat_c"] - h
    hi = lat_max if d["lat_hi"] >= lat_max - 1e-6 else d["lat_c"] + h
    return lo, hi


def _draw_train(ax, tr, members, lats, cfg):
    """Draw a train's crest (solid) and trough (dashed) lines, each detection over
    its sub-array's slice, and label the train."""
    for d in members:
        lo, hi = _owned_slice(d, lats.min(), lats.max(), cfg)
        la = np.linspace(lo, hi, 8)
        dd = (la - d["lat_c"]) * M_PER_DEG
        for tc, ls in [(c, "-") for c in d["crests_s"]] + [(c, "--") for c in d["troughs_s"]]:
            ax.plot((tc - dd * d["slowness"]) / 3600, la, color="k", lw=1.2, ls=ls)
    tm = 0.5 * (tr["t_start_s"] + tr["t_end_s"]) / 3600
    ax.text(tm, lats.max() - 0.3,
            f"#{tr['train_id']} {tr['speed_m_s']:.0f} m/s · {tr['period_s'] / 60:.0f} min · "
            f"{tr['confidence']}", fontsize=8, ha="center", va="top",
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
    nb  = len(results)
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, nb, height_ratios=[1.2, 0.9, 0.6], hspace=0.35, wspace=0.08)

    ax_k = fig.add_subplot(gs[0, :])
    _draw_keogram(ax_k, X, lats, t, f"(a) band-passed keogram and LSTID trains — {cfg.input_date}   "
                                    "(solid = crest, dashed = trough)")
    for tr, idx in zip(trains, members):
        ax_k.axvspan(tr["t_start_s"] / 3600, tr["t_end_s"] / 3600, color="k", alpha=0.06, lw=0)
        _draw_train(ax_k, tr, [dets[i] for i in idx], lats, cfg)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i], sharex=ax_k)
        im = _draw_vespa(ax, r, t, cfg, f"({chr(ord('b') + i)}) {r['lat_lo']:g}–{r['lat_hi']:g}°N",
                         ylabels=(i == 0))
    fig.colorbar(im, ax=fig.axes[-nb:], pad=0.01, fraction=0.02).set_label("semblance", fontsize=9)

    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    hdr = (f"{'#':>2}  {'time UT (47.5°N)':16s} {'speed':>6s} {'interval':>11s} {'period':>6s} "
           f"{'λ km':>6s} {'amp':>6s} {'sembl':>5s} {'cover':>5s} {'det':>3s} {'crest/trough':>12s} {'score':>5s} conf")
    rows = [hdr, "─" * len(hdr)]
    for tr in trains:
        h0, h1 = tr["t_start_s"] / 3600, tr["t_end_s"] / 3600
        rows.append(f"{tr['train_id']:>2}  {int(h0) % 24:02d}:{int(h0 % 1 * 60):02d}-{int(h1) % 24:02d}:{int(h1 % 1 * 60):02d}      "
                    f"{tr['speed_m_s']:6.0f} {tr['speed_lo_m_s']:5.0f}-{tr['speed_hi_m_s']:<5.0f} "
                    f"{tr['period_s'] / 60:6.0f} {tr['wavelength_km']:6.0f} {tr['amplitude']:6.3f} "
                    f"{tr['semblance']:5.2f} {tr['coverage']:5.2f} {tr['n_detections']:3d} "
                    f"{tr['n_crests']:>6d}/{tr['n_troughs']:<5d} {tr['confidence_score']:5.2f} {tr['confidence']}")
    ax.text(0, 1, "\n".join(rows) if trains else "no LSTID trains", family="monospace",
            fontsize=9, va="top")
    ax.set_title(f"({chr(ord('b') + nb)}) LSTID trains (black ridges in the vespagrams; "
                 f"grey = detections in no train)", fontsize=10, loc="left")

    for a in fig.axes:
        if a.get_label() == "<colorbar>" or not a.axison:
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

TRAIN_COLS = ["train_id", "t_start", "t_end", "speed_m_s", "speed_lo_m_s", "speed_hi_m_s", "period_min",
              "wavelength_km", "amplitude", "semblance", "coverage", "n_detections", "n_crests",
              "n_troughs", "n_cycles", "confidence_score", "confidence"]
# columns of the published dataset; the full tables stay in out_dir for the analyses
RELEASE_COLS = {
    "trains":     ["date", "train_id", "t_start", "t_end", "speed_m_s", "speed_lo_m_s",
                   "speed_hi_m_s", "period_min", "wavelength_km", "amplitude",
                   "confidence_score", "confidence"],
    "detections": ["date", "train_id", "array", "lat_lo", "lat_hi", "t_start", "t_end",
                   "speed_m_s", "period_min", "amplitude", "semblance", "n_extrema",
                   "crests_s", "troughs_s"],
}


def detect(cfg):
    """
    Run steps 1-5 for one day. Returns (X, lats, t, results, dets, trains,
    members): results holds one dict per sub-array (its vespagram and
    detections), dets all detections, trains the LSTID trains and members their
    detection indices.
    """
    X, lats, t = load_and_prepare(cfg)
    return (X, lats, t, *analyse(X, lats, t, cfg))


def analyse(X, lats, t, cfg):
    """
    Steps 3-5 on a prepared (band-passed, decimated) keogram — separated from
    loading so synthetic keograms can be tested. Returns (results, dets, trains,
    members).
    """
    dt_s   = t[1] - t[0]
    p_grid = cfg.p_grid

    results, dets = [], []
    arrays = array_list(lats, cfg)
    for lat_lo, lat_hi in arrays:
        rows  = (lats >= lat_lo - 1e-6) & (lats <= lat_hi + 1e-6)
        Xa    = X[rows]
        dist  = meridian_distance(lats[rows], 0.5 * (lat_lo + lat_hi))
        S, A  = slant_stack(Xa, dist, p_grid, dt_s, cfg)
        ds    = pick_detections(S, A, Xa, dist, p_grid, t, lat_lo, lat_hi, cfg)
        results.append({"lat_lo": lat_lo, "lat_hi": lat_hi, "S": S, "A": A, "p": p_grid, "dets": ds})
        dets += ds

    ref_lat = 0.5 * (float(lats.min()) + float(lats.max()))
    trains, members = build_trains(dets, len(arrays), ref_lat, t, cfg)
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
    """Detections that belong to an LSTID train (ridge pixel arrays dropped);
    times, crests and troughs at the detection's sub-array centre."""
    return [{"date": date, "train_id": d["train_id"],
             **{k: d[k] for k in ("array", "lat_lo", "lat_hi", "slowness",
                                   "speed_m_s", "amplitude", "semblance", "n_extrema",
                                   "crests_s", "troughs_s")},
             "t_start":    _abs_time(date, d["t_start_s"]),
             "t_end":      _abs_time(date, d["t_end_s"]),
             "period_min": d["period_s"] / 60}
            for d in dets if "train_id" in d]


def resolve_day_boundaries(trains, dets):
    """
    Each day keeps only the trains whose midpoint falls in it (build_trains), and
    with the 12-h padding both days see a train crossing midnight whole, so it is
    normally catalogued once. If the two views still differ (a run of detections
    reaching the end of one day's padding), both days can claim the same train:
    two trains of adjacent days that overlap in time are one, and the longer
    stays. Returns (trains, dets) with the detections of dropped trains removed.
    """
    over = lambda a, b: a["t_start"] < b["t_end"] and b["t_start"] < a["t_end"]
    dur  = lambda x: (x["t_end"] - x["t_start"], x["date"])       # date breaks ties
    by_day = {}
    for r in trains:
        by_day.setdefault(r["date"], []).append(r)
    keep = {id(r) for r in trains}
    for r in trains:
        for o in by_day.get(r["date"] + dt.timedelta(days=1), []):
            if over(r, o):
                keep.discard(id(min(r, o, key=dur)))

    kept = [r for r in trains if id(r) in keep]
    ids  = {(r["date"], r["train_id"]) for r in kept}
    return kept, [d for d in dets if (d["date"], d["train_id"]) in ids]


def write_day(cfg, X, lats, t, results, dets, trains, members):
    """The figure and the train CSV of one day (the output of detect), in cfg.out_dir."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tr_rec = train_records(trains, cfg.input_date)
    if tr_rec:
        pl.DataFrame(tr_rec).select(TRAIN_COLS).write_csv(
            cfg.out_dir / f"{cfg.input_date}_trains.csv")
    plot(X, lats, t, results, dets, trains, members, cfg)


def run(cfg, do_plot=True):
    """Detect one day; with do_plot also write the figure and a per-day train CSV.
    Returns (train records, detection records)."""
    t_start = time.time()
    X, lats, t, results, dets, trains, members = detect(cfg)
    t_detect = time.time() - t_start

    if do_plot:
        write_day(cfg, X, lats, t, results, dets, trains, members)
    logging.info("%s: %d detections -> %d LSTID trains, detect %.2fs, total %.2fs",
                 cfg.input_date, len(dets), len(trains), t_detect, time.time() - t_start)
    return train_records(trains, cfg.input_date), detection_records(dets, cfg.input_date)


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


def write_release(folder):
    """Copy the catalogue in `folder` to folder/release with the published columns."""
    (folder / "release").mkdir(exist_ok=True)
    for name, cols in RELEASE_COLS.items():
        _write_table(pl.read_parquet(folder / f"{name}.parquet").select(cols),
                     folder / "release", name)


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

    n_raw = len(trains)
    trains, dets = resolve_day_boundaries(trains, dets)
    print(f"Day boundaries: {n_raw - len(trains)} trains claimed by both days removed")

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
        if not tag:
            write_release(base.out_dir)
        print(f"Saved {len(trains)} LSTID trains ({len(dets)} detections) from "
              f"{len(dates) - len(failures)} days -> {base.out_dir}")
    print(f"Done in {time.time() - t0:.1f}s, {len(failures)} failed")
    for d, e in failures:
        print(f"  failed: {d}  ({e})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
