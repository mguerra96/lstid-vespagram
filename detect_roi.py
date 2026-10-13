"""
detect_roi.py

Pipeline:
  1. Load keogram grid (parquet)
  2. Downscale + Gaussian smooth
  3. Detect ROIs via connected components, splitting forked blobs with GMM
  4. Filter ROIs by height
  5. Fit RANSAC line per ROI → estimate propagation speed
  6. Match opposite-sign ROI pairs to identify LSTIDs
  7. Plot and save results
"""

import time
import json
import pickle
import logging
import argparse
import warnings
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, fields, replace
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe in worker processes
import matplotlib.pyplot as plt
# the gridspec colorbar trips tight_layout's compatibility check; the warning is
# cosmetic but would otherwise smear the progress bar, so silence just that one.
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
import matplotlib.gridspec as gridspec
import cv2
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential, join_segmentations, find_boundaries
from skimage.color import label2rgb
from skimage.transform import downscale_local_mean
from sklearn.preprocessing import minmax_scale
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RANSACRegressor
from geopy.distance import geodesic

import paths


# ── Configuration ───────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    All tunable parameters for one keogram run. Defaults reproduce the original
    hard-coded settings. Several thresholds are expressed in downscaled pixels or
    raw TECu and are therefore resolution-dependent; the physical equivalent for
    the current default grid is noted in each field's comment for reviewer context.
    """
    # ── Paths & date ──
    keo_folder: Path = paths.KEO_DIR
    out_dir:    Path = paths.ROOT / "results"
    input_date: dt.date = dt.date(2015, 12, 27)
    longitude:    float = 17.5   # centre longitude of the keogram [deg]
    ref_latitude: float = 47.5   # reference latitude used to compute ROI ref_time [deg]

    # ── Pre-processing ──
    border_pad_hours:        float = 2     # hours of the prev/next day to pad on each
                                           # side (avoids border effects on waves that
                                           # cross midnight); 0 disables padding
    time_downscaling_factor: int   = 4     # downsample along time axis before detection
    gaussian_sigma_y:        float = 2     # Gaussian smoothing sigma along latitude (y) [px]
    gaussian_sigma_x:        float = 0.5   # Gaussian smoothing sigma along time (x) [px]

    # ── ROI detection ──
    amplitude_threshold:       float = 0.015  # dTEC threshold for binary masks [TECu]
    min_blob_area:             int   = 1000   # min blob area [downscaled px]; ~? km·min on the default grid
    concavity_depth_threshold: float = 30     # min bay depth to split a fork [downscaled px]

    # ── GMM splitting ──
    gmm_max_components: int = 5    # hard cap on number of Gaussian components
    # a GMM cut is undone (pieces merged back) when its row extent is >= this
    # fraction of BOTH pieces' row extents — i.e. the cut runs the full height of
    # both, meaning one stripe was sliced across rather than a fork separated.
    merge_row_overlap_frac: float = 0.8

    # ── ROI filtering ──
    min_height:   float = 0.5    # min blob height as fraction of keogram height [-]

    # ── LSTID matching ──
    lstid_time_distance:   float = 3600   # max ref_time separation between paired ROIs [s]
    lstid_max_speed:       float = 1500   # max propagation speed to consider [m/s]
    lstid_min_speed:       float = 300    # min propagation speed to consider [m/s]
    lstid_orient_diff:     float = 2      # max orientation difference between paired ROIs [deg]
    lstid_min_lat_overlap: float = 10    # min latitude overlap between paired ROIs [deg]
    lstid_min_period:      float = 1800   # min wave period to keep [s]
    lstid_min_matches:     int   = 1      # min opposite-sign companions an ROI needs, counted
                                          # on either side in time; 1 = keep any paired ROI,
                                          # raise to demand a fuller wave train [-]

    @property
    def grid_path(self) -> Path:
        return self.keo_folder / f"keogram_grid_{self.input_date}.parquet"


# ── Run identity ────────────────────────────────────────────────────────────────

# Fields that say WHERE a run reads/writes or WHICH day it covers, rather than HOW
# detection behaves. They are excluded from the run name: input_date varies per
# date within a single batch, so including it would give every date its own folder
# and defeat the grouping.
_NON_SCIENTIFIC = frozenset({"keo_folder", "out_dir", "input_date"})

DEFAULT_RUN_NAME = "roi"


def scientific_fields():
    """The tunable detection parameters, in declaration order."""
    return [f for f in fields(Config) if f.name not in _NON_SCIENTIFIC]


def _fmt(value):
    """Format a parameter value for a folder name: floats without trailing zeros
    (0.015 -> '0.015', not '0.014999999999999999'), everything else via str."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_name(cfg):
    """
    Name identifying this run by the scientific parameters that differ from the
    defaults, e.g. 'roi_amplitude_threshold=0.02_min_height=0.4'. A run that changes
    nothing is DEFAULT_RUN_NAME, so repeated default runs land in one folder
    instead of silently overwriting each other under different names. The 'roi'
    prefix keeps ROI runs apart from the vespagram results in the same folder.
    """
    defaults = Config()
    # Compare on the FORMATTED value, not the raw one: floats a hair off the
    # default (0.015000000000000001 from arithmetic or parsing) are != the default
    # yet format identically, which would otherwise name a folder
    # 'amplitude_threshold=0.015' that claims to differ from a default of 0.015.
    # If two values render the same, they are the same run.
    diffs = [f"{f.name}={_fmt(getattr(cfg, f.name))}"
             for f in scientific_fields()
             if _fmt(getattr(cfg, f.name)) != _fmt(getattr(defaults, f.name))]
    return "_".join([DEFAULT_RUN_NAME] + diffs)


def run_dir(cfg):
    """The output folder for this run: <out_dir>/<run_name>."""
    return cfg.out_dir / run_name(cfg)


def write_config(cfg, path):
    """
    Record the FULL config (not just the diffs) as config.json in the run folder.
    The folder name is a readable hint but gets unwieldy with many changes; this
    file is the authoritative record of what produced the results beside it.
    input_date is omitted: one folder spans many dates.
    """
    payload = {f.name: _json_safe(getattr(cfg, f.name))
               for f in fields(Config) if f.name != "input_date"}
    payload["run_name"] = run_name(cfg)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_safe(value):
    """Path and date aren't JSON types; render them as strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load grid
# ══════════════════════════════════════════════════════════════════════════════

SECONDS_PER_DAY = 86400


def _read_grid(path):
    """Read one keogram parquet -> (Z, lats, times_sec) with Z flipped (row 0 =
    max latitude), NaNs zeroed and clipped. times_sec is seconds since that day's
    midnight. Raises if the file is missing or the time axis is non-uniform."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Keogram grid not found: {path}")

    df        = pl.read_parquet(path)
    lats      = df["lat"].to_numpy()
    time_cols = [c for c in df.columns if c != "lat"]
    times_sec = np.array([int(c) for c in time_cols])

    steps = np.diff(times_sec)
    if steps.size and not np.allclose(steps, steps[0]):
        raise ValueError(
            f"Time axis is not uniformly spaced in {path.name}: "
            f"step ranges {steps.min()}..{steps.max()} s. "
            "Speed and period estimates assume a constant cadence.")

    Z = df.select(time_cols).to_numpy()
    Z = np.flipud(Z)              # row 0 = max latitude
    Z = np.nan_to_num(Z, nan=0.0)
    Z = np.clip(Z, -1.0, 1.0)
    return Z, lats, times_sec


def load_grid(cfg):
    """
    Load day cfg.input_date, optionally padded with the last cfg.border_pad_hours
    of the previous day and the first cfg.border_pad_hours of the next day, to
    avoid border effects on waves that cross midnight.

    Padding columns carry times relative to day D's midnight: the previous day's
    tail gets negative seconds, the next day's head gets seconds >= 86400. So a
    column belongs to day D iff its time is in [0, 86400). Missing neighbour grids
    are skipped silently. Returns (Z, lats, times_sec, day_mask) where day_mask is
    a boolean over columns marking which belong to day D.
    """
    Z, lats, times_sec = _read_grid(cfg.grid_path)
    step    = int(times_sec[1] - times_sec[0]) if len(times_sec) > 1 else 30
    pad_cols = int(round(cfg.border_pad_hours * 3600 / step))

    blocks_Z, blocks_t = [Z], [times_sec]

    if pad_cols > 0:
        # previous day's tail -> prepend, shifted by -1 day
        prev = cfg.keo_folder / f"keogram_grid_{cfg.input_date - dt.timedelta(days=1)}.parquet"
        if prev.exists():
            pZ, _, pt = _read_grid(prev)
            blocks_Z.insert(0, pZ[:, -pad_cols:])
            blocks_t.insert(0, pt[-pad_cols:] - SECONDS_PER_DAY)
        # next day's head -> append, shifted by +1 day
        nxt = cfg.keo_folder / f"keogram_grid_{cfg.input_date + dt.timedelta(days=1)}.parquet"
        if nxt.exists():
            nZ, _, nt = _read_grid(nxt)
            blocks_Z.append(nZ[:, :pad_cols])
            blocks_t.append(nt[:pad_cols] + SECONDS_PER_DAY)

    Z         = np.concatenate(blocks_Z, axis=1)
    times_sec = np.concatenate(blocks_t)
    day_mask  = (times_sec >= 0) & (times_sec < SECONDS_PER_DAY)
    return Z, lats, times_sec, day_mask


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(Z, cfg):
    """Downsample along the time axis and apply Gaussian smoothing."""
    Z_ds = downscale_local_mean(Z, (1, cfg.time_downscaling_factor))
    Z_ds = gaussian_filter(Z_ds, sigma=(cfg.gaussian_sigma_y, cfg.gaussian_sigma_x))
    return Z_ds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ROI detection
# ══════════════════════════════════════════════════════════════════════════════

def concavity_depth(p):
    """
    Depth of the deepest concavity (convexity defect) of a blob, in pixels.

    A forked / Y-shaped blob (two stripes merging) has a deep bay between the
    prongs: one convex-hull edge bridges the notch and the boundary dips far
    below it. A single coherent "I" stripe — even a curved or wavy one — has
    only shallow defects, because the hull hugs the curve in many short edges.

    Uses cv2.convexityDefects, which reports each defect's depth as the
    perpendicular distance from the farthest contour point to the bridging
    hull edge (so lengthwise curvature of an elongated stripe does NOT inflate
    it). Returns the max such depth, in pixels.
    """
    img = np.ascontiguousarray(p.image.astype(np.uint8))
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    # Lightly denoise the pixel staircase with a small fixed tolerance (1 px). A
    # perimeter-fraction eps over-simplifies long stripes — collapsing them to a
    # few points, which both erases real bays and makes convexityDefects raise —
    # so the deep concavity of a tall fork was being silently reported as 0.
    cnt = cv2.approxPolyDP(cnt, 1.0, True)
    if len(cnt) < 4:
        return 0.0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0.0
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return 0.0
    if defects is None:
        return 0.0
    # defects[:, 0, 3] is fixed-point depth in units of 1/256 px
    return defects[:, 0, 3].max() / 256.0


def split_blob(p, mask, matrix, cfg):
    """
    Split a forked blob using a GMM on its 1D projection along the major axis.

    TID stripes are elongated along the time axis (major axis ≈ time). When two
    stripes merge into one Y-shaped blob, a deep concavity opens between the
    prongs. The pixels are clustered along the major axis and the boundaries
    between clusters are erased from the binary mask, cutting the blob across
    its width. The number of clusters is increased until no resulting sub-blob
    is still forked (max concavity below cfg.concavity_depth_threshold).
    """
    coords = np.argwhere(p.image)
    X      = coords + p.bbox[:2]   # convert to full-matrix coordinates

    # project each pixel onto the blob's major axis (≈ vertical/time-tilt);
    # GMM clusters along this axis so the erased boundaries are vertical cuts
    # that separate side-by-side parallel stripes.
    major_vec = np.array([-np.sin(p.orientation), np.cos(p.orientation)])
    proj      = (X - np.array(p.centroid)) @ major_vec
    X_scaled  = minmax_scale(proj.reshape(-1, 1))

    # subsample to at most 1000 pixels for speed if the blob is large
    if len(X_scaled) > 1000:
        idx   = np.random.default_rng(0).choice(len(X_scaled), 1000, replace=False)
        X_fit = X_scaled[idx]
    else:
        X_fit = X_scaled

    def split_boundaries(model):
        """Boundary pixels (full-matrix coords) erased by this model's clusters."""
        subregions = coo_matrix((model.predict(X_scaled) + 1, X.T),
                                shape=matrix.shape).toarray()
        return find_boundaries(subregions, mode='outer')

    def max_piece_concavity(boundaries):
        """Max concavity over the sub-blobs left after erasing `boundaries`."""
        piece = p.image.copy()
        piece[boundaries[p.slice] & p.image] = False
        worst = 0.0
        for q in regionprops(label(piece)):
            worst = max(worst, concavity_depth(q))
        return worst

    # Increase the number of GMM components until no resulting sub-blob is still
    # forked (max concavity below threshold). The split criterion is thus the
    # same geometry that triggered the split in the first place.
    chosen = None
    for n in range(2, cfg.gmm_max_components + 1):
        model      = GaussianMixture(n_components=n, random_state=0).fit(X_fit)
        boundaries = split_boundaries(model)
        chosen     = boundaries
        if max_piece_concavity(boundaries) <= cfg.concavity_depth_threshold:
            break
    else:
        logging.warning("blob %s still forked after %d GMM components "
                        "(max concavity %.1f px > %.1f px threshold)",
                        getattr(p, "label", "?"), cfg.gmm_max_components,
                        max_piece_concavity(chosen), cfg.concavity_depth_threshold)

    # Erase the accepted split's boundaries, then undo any cut whose row extent is
    # similar to BOTH adjacent pieces' row extents (a spurious split: the cut runs
    # the full height of both pieces, so it sliced one stripe across — not a fork).
    # A genuine fork (e.g. a diapason joined at its base) only separates in part of
    # the latitude span, so its cut is shorter than the pieces and is kept.
    erase = chosen[p.slice] & p.image
    keep  = _spurious_split_boundaries(p, erase, cfg)   # boundary px to re-fill (merge back)
    np.putmask(mask[p.slice], erase & ~keep, 0)


def _row_extent(mask2d):
    """Number of rows spanned by a 2D boolean mask (max_row - min_row + 1)."""
    rows = np.nonzero(mask2d.any(axis=1))[0]
    return (rows.max() - rows.min() + 1) if rows.size else 0


def _spurious_split_boundaries(p, erase, cfg):
    """
    Return the boundary pixels (within p's bbox) that should be RE-FILLED because
    the cut between two adjacent sub-blobs spans nearly the full row extent of BOTH
    pieces — i.e. the GMM sliced one coherent stripe across its length rather than
    separating two side-by-side stripes.

    Criterion: let rows_split be the row extent of the erased cut, and rows1/rows2
    the row extents of the two pieces it separates. Merge back when
        rows_split / rows1 >= frac  AND  rows_split / rows2 >= frac
    (the cut is "tall" relative to both). A genuine fork keeps its split because
    the prongs join over part of the latitude, so the cut is short compared to the
    full-height pieces and at least one ratio falls below `frac`.

    Pieces are ordered along the major (split) axis; only consecutive pieces are
    adjacent.
    """
    pieces = label(p.image & ~erase)
    if pieces.max() < 2:
        return np.zeros_like(erase)

    # Each connected component of the ERASED pixels is one "cut". It is the part
    # removed from between regions; the pieces it borders are the regions it split.
    # Re-fill (merge back) a cut whose row extent is >= frac of BOTH bordering
    # pieces' row extents — it ran the full height of both, so it sliced one stripe
    # across rather than separating a fork (which only parts over some of the span).
    cuts = label(erase)
    refill = np.zeros_like(erase)
    for c in range(1, cuts.max() + 1):
        cut       = cuts == c
        neighbours = np.unique(pieces[binary_dilation(cut)])
        neighbours = neighbours[neighbours > 0]
        if len(neighbours) < 2:
            continue                                    # cut doesn't separate two pieces
        rows_split   = _row_extent(cut)
        piece_extents = [_row_extent(pieces == lbl) for lbl in neighbours]
        if all(rows_split / e >= cfg.merge_row_overlap_frac for e in piece_extents):
            refill |= cut
    return refill


def draw_concavity_debug(ax, matrix, cfg):
    """
    Draw the pre-split debug panel onto `ax`: all blobs BEFORE splitting, each
    annotated with its blob ID and concavity depth [px] (as "#id / depth"). Blobs
    that exceed cfg.concavity_depth_threshold (i.e. will be split) are coloured
    red, else white.
    """
    neg = remove_small_objects(matrix < -cfg.amplitude_threshold, max_size=cfg.min_blob_area)
    pos = remove_small_objects(matrix > cfg.amplitude_threshold, max_size=cfg.min_blob_area)
    pre = join_segmentations(label(neg), label(pos))
    ax.imshow(label2rgb(pre, bg_label=0), aspect="auto")
    for p in regionprops(pre):
        cd = concavity_depth(p)
        will_split = cd > cfg.concavity_depth_threshold
        y, x = p.centroid
        ax.text(x, y, f"#{p.label}\n{cd:.0f}", color="red" if will_split else "white",
                fontsize=7, ha="center", va="center",
                fontweight="bold" if will_split else "normal")
    ax.set_title(f"Pre-split blobs — concavity depth [px]  "
                 f"(red = will split, thresh={cfg.concavity_depth_threshold})", fontsize=13)


def detect_roi(matrix, cfg):
    """
    Detect positive and negative dTEC blobs, split forked ones, and return a
    labelled ROI array.

    Tiny blobs are discarded first (cfg.min_blob_area). A blob is then split when
    it has a concavity deeper than cfg.concavity_depth_threshold (a fork). Positive
    and negative blobs are processed independently, then merged into a single label
    image using join_segmentations.
    """
    roi_layers = []

    for mask in [matrix < -cfg.amplitude_threshold, matrix > cfg.amplitude_threshold]:
        # discard tiny blobs immediately so they neither get split nor labelled
        mask[:] = remove_small_objects(mask, max_size=cfg.min_blob_area)
        for p in regionprops(label(mask)):
            if concavity_depth(p) > cfg.concavity_depth_threshold:
                split_blob(p, mask, matrix, cfg)

        # re-apply the area threshold: splitting can leave sub-blobs (and the
        # erased boundary pixels can shed slivers) below cfg.min_blob_area
        mask[:] = remove_small_objects(mask, max_size=cfg.min_blob_area)

        roi_layers.append(relabel_sequential(label(mask))[0])

    # ensure negative-dTEC ROIs contain only negative pixels and vice versa
    np.putmask(roi_layers[0], matrix > 0, 0)
    np.putmask(roi_layers[1], matrix < 0, 0)

    return join_segmentations(roi_layers[0], roi_layers[1])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ROI filtering
# ══════════════════════════════════════════════════════════════════════════════

def filter_roi(roi, matrix, cfg):
    """
    Remove blobs that are unlikely to be TIDs:
      - too short (less than cfg.min_height fraction of the keogram height)
    """
    min_height = cfg.min_height * matrix.shape[0]
    filtered   = roi.copy()
    removed    = [p.label for p in regionprops(filtered)
                  if p.image.shape[0] < min_height]
    np.putmask(filtered, np.isin(filtered, removed), 0)
    return relabel_sequential(filtered)[0]


def draw_filter_debug(ax, roi, matrix, cfg):
    """
    Draw the pre-filter debug panel onto `ax`: all ROIs AFTER GMM splitting but
    BEFORE filtering, each annotated with its ROI ID and the filter metric
    (height [px]). The metric is red when it fails its threshold, i.e. discarded.
    """
    min_height = cfg.min_height * matrix.shape[0]
    ax.imshow(label2rgb(roi, bg_label=0), aspect="auto")
    for p in regionprops(roi):
        h    = p.image.shape[0]
        y, x = p.centroid
        fails = h < min_height
        ax.text(x, y - 6, f"#{p.label}", color="white", fontsize=7,
                ha="center", va="center", fontweight="bold")
        ax.text(x, y + 6, f"h={h}", color="red" if fails else "white",
                fontsize=7, ha="center", va="center",
                fontweight="bold" if fails else "normal")
    ax.set_title(f"Pre-filter ROIs — h<{min_height:.0f} "
                 f"(red = fails → discarded)", fontsize=13)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — RANSAC line fitting and speed estimation
# ══════════════════════════════════════════════════════════════════════════════

def fit_lines(roi, matrix, lats, times_sec, cfg):
    """
    For each ROI, fit a RANSAC line (row ~ col) and derive physical parameters:
    propagation speed, timing, latitude extent, and ref_time (time at
    cfg.ref_latitude for cross-ROI comparison).

    The RANSAC fit is restricted to inlier rows to avoid edge pixels of the
    blob distorting the speed estimate.
    """
    n_lat, n_time = matrix.shape
    lat_range     = (lats.min(), lats.max())
    deg_per_row   = (lat_range[1] - lat_range[0]) / n_lat
    sec_per_col   = (times_sec[-1] - times_sec[0]) / n_time
    t0_dt         = dt.datetime.combine(cfg.input_date, dt.time())
    results       = []

    for p in regionprops(roi):
        rows, cols = np.nonzero(roi == p.label)
        if len(rows) < 10:
            continue

        ransac = RANSACRegressor(min_samples=10, max_trials=50, random_state=0)
        ransac.fit(rows[:, None], cols)

        # use only inlier rows to avoid edge outliers stretching the speed estimate
        inlier_rows = rows[ransac.inlier_mask_]
        x = np.arange(inlier_rows.min(), inlier_rows.max() + 1)
        y = ransac.predict(x[:, None])

        start_lat  = lat_range[1] - x[y.argmin()] * deg_per_row
        end_lat    = lat_range[1] - x[y.argmax()] * deg_per_row
        start_time = t0_dt + dt.timedelta(seconds=int(times_sec[0]) + int(y.min() * sec_per_col))
        end_time   = t0_dt + dt.timedelta(seconds=int(times_sec[0]) + int(y.max() * sec_per_col))
        duration_s = (end_time - start_time).total_seconds()
        distance_m = geodesic((start_lat, cfg.longitude), (end_lat, cfg.longitude)).meters
        speed      = distance_m / duration_s if duration_s > 0 else np.nan

        # ref_time: time at cfg.ref_latitude, used as a consistent cross-ROI timestamp
        ref_row  = (lat_range[1] - cfg.ref_latitude) / deg_per_row
        ref_col  = float(ransac.predict([[ref_row]])[0])
        ref_time = t0_dt + dt.timedelta(seconds=int(times_sec[0]) + ref_col * sec_per_col)

        # robust "peak" amplitude: the 95th percentile of magnitude (5th for
        # negative-dTEC ROIs), signed to match the ROI. Avoids a single noisy
        # pixel inflating the reported peak the way a plain max() would.
        amp          = matrix[rows, cols]
        mean_amp     = amp.mean()
        peak_amp     = np.percentile(amp, 95) if mean_amp > 0 else np.percentile(amp, 5)

        results.append({
            "roi_label":      p.label,
            "start_time":     start_time,
            "end_time":       end_time,
            "ref_time":       ref_time,
            "start_lat":      start_lat,
            "end_lat":        end_lat,
            "speed_m_s":      speed,
            "orientation":    np.degrees(p.orientation),
            "solidity":       p.solidity,
            "area":           p.area,
            "mean_amplitude": mean_amp,
            "peak_amplitude": peak_amp,
            "x":              x,
            "y":              y,
        })

    results.sort(key=lambda r: r["start_time"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — LSTID matching
# ══════════════════════════════════════════════════════════════════════════════

def match_lstid(lines, cfg):
    """
    Keep only ROIs that belong to an LSTID wave by requiring each ROI to have
    at least one opposite-sign companion satisfying all three criteria:
      - ref_time within cfg.lstid_time_distance (same wave packet)
      - orientation difference within cfg.lstid_orient_diff (same propagation speed)
      - latitude overlap >= cfg.lstid_min_lat_overlap (same spatial extent)

    The wave period is estimated from the nearest opposite-sign companion on each
    side in time (the crest immediately before and the trough immediately after,
    or vice versa). The crest-to-trough half-period is the time gap to the closest
    partner; the full period is twice that. When companions exist on both sides we
    average the two nearest half-periods; when only one side has a companion we use
    that single nearest gap. This avoids conflating harmonics, which a mean over
    *all* matched partners would do.
    A final filter removes ROIs whose estimated period is below cfg.lstid_min_period.
    """
    candidates = [r for r in lines
                  if cfg.lstid_min_speed <= r["speed_m_s"] <= cfg.lstid_max_speed]

    pos = [r for r in candidates if r["mean_amplitude"] > 0]
    neg = [r for r in candidates if r["mean_amplitude"] < 0]

    result = []
    for group, opposite in [(pos, neg), (neg, pos)]:
        for r in group:
            matches = [
                o for o in opposite
                if (abs((r["ref_time"] - o["ref_time"]).total_seconds()) <= cfg.lstid_time_distance
                    and abs(r["orientation"] - o["orientation"]) <= cfg.lstid_orient_diff
                    and min(r["start_lat"], o["start_lat"]) - max(r["end_lat"], o["end_lat"]) >= cfg.lstid_min_lat_overlap)
            ]
            if not matches:
                continue

            # signed time gaps to matched opposite-sign partners
            gaps   = [(o["ref_time"] - r["ref_time"]).total_seconds() for o in matches]
            before = [-g for g in gaps if g < 0]   # partners earlier in time
            after  = [ g for g in gaps if g > 0]   # partners later in time
            # nearest half-period on each available side, averaged
            nearest = [min(side) for side in (before, after) if side]
            period  = 2 * np.mean(nearest)
            result.append({**r, "period_s": period})

    return [r for r in result if r["period_s"] >= cfg.lstid_min_period]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot(Z_ds, roi, roi_filtered, roi_lstid, lines, lstid_lines, lats, times_sec, day_mask,
         cfg, debug=False):
    """
    Diagnostic plot for one keogram. The smoothed keogram (first) and the
    LSTID-matched ROIs (last) are always drawn. When `debug` is True the
    intermediate panels — all ROIs, concavity debug, filter debug, filtered ROIs
    + speeds — are added in pipeline order. The number of rows adapts to how many
    panels are active.

    The keogram is processed padded with neighbour-day borders, but the displayed
    x-range is clipped to day D (day_mask marks the day-D columns) so each plot
    covers exactly one day.
    """
    # day-D column window, mapped from full-res `times_sec`/`day_mask` to the
    # time-downscaled image columns used for plotting
    f          = cfg.time_downscaling_factor
    day_cols   = np.nonzero(day_mask)[0]
    xlim       = (day_cols[0] / f, (day_cols[-1] + 1) / f)
    day_t0, day_t1 = times_sec[day_cols[0]], times_sec[day_cols[-1]]

    n_ticks       = 13
    time_tick_pos = np.linspace(xlim[0], xlim[1], n_ticks)
    time_tick_lbl = [
        str(dt.datetime.combine(cfg.input_date, dt.time()) + dt.timedelta(seconds=int(s)))[-8:-3]
        for s in np.linspace(day_t0, day_t1, n_ticks)
    ]
    lat_tick_pos  = np.linspace(0, Z_ds.shape[0], 5)
    lat_tick_lbl  = [f"{v:.1f}" for v in np.linspace(lats[-1], lats[0], 5)]

    def set_lat_ticks(ax):
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_yticks(lat_tick_pos)
        ax.set_yticklabels(lat_tick_lbl)

    # ── panel drawing functions, in pipeline order ──
    def panel_keogram(ax):
        im = ax.imshow(Z_ds, cmap="jet", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_title(f"Smoothed keogram  (σ_y={cfg.gaussian_sigma_y}, σ_x={cfg.gaussian_sigma_x}, "
                     f"×{cfg.time_downscaling_factor})  —  {cfg.input_date}", fontsize=13)
        return im   # only this panel returns a mappable (for the colorbar)

    def panel_concavity(ax):
        draw_concavity_debug(ax, Z_ds, cfg)

    def panel_filter(ax):
        draw_filter_debug(ax, roi, Z_ds, cfg)

    def panel_filtered(ax):
        sign_rgb = np.zeros((*roi_filtered.shape, 3))           # black background
        sign_rgb[(roi_filtered > 0) & (Z_ds > 0)] = [1, 0, 0]   # +dTEC -> red
        sign_rgb[(roi_filtered > 0) & (Z_ds < 0)] = [0, 0, 1]   # -dTEC -> blue
        ax.imshow(sign_rgb, aspect="auto")
        for r in lines:
            ax.plot(np.clip(r["y"], 0, Z_ds.shape[1] - 1), r["x"],
                    color="white", linewidth=1.5, alpha=0.8)
            mid = len(r["x"]) // 2
            ax.text(r["y"][mid], r["x"][mid], f"{r['speed_m_s']:.0f}",
                    color="white", fontsize=7, ha="center", va="center", fontweight="bold")
        ax.set_title(f"Filtered ROIs + RANSAC speeds: {roi_filtered.max()}", fontsize=13)

    def panel_lstid(ax):
        sign_rgb = np.zeros((*roi_lstid.shape, 3))           # black background
        sign_rgb[(roi_lstid > 0) & (Z_ds > 0)] = [1, 0, 0]   # +dTEC -> red
        sign_rgb[(roi_lstid > 0) & (Z_ds < 0)] = [0, 0, 1]   # -dTEC -> blue
        ax.imshow(sign_rgb, aspect="auto")
        for r in lstid_lines:
            ax.plot(np.clip(r["y"], 0, Z_ds.shape[1] - 1), r["x"],
                    color="white", linewidth=1.5, alpha=0.8)
            mid = len(r["x"]) // 2
            ax.text(r["y"][mid], r["x"][mid], f"{r['speed_m_s']:.0f} / {r['period_s']/60:.0f}m",
                    color="white", fontsize=7, ha="center", va="center", fontweight="bold")
        ax.set_title(f"LSTID ROIs: {len(lstid_lines)}", fontsize=13)

    # assemble the active panels in pipeline order: keogram, [debug...], lstid
    panels = [panel_keogram]
    if debug:
        panels += [panel_concavity, panel_filter, panel_filtered]
    panels.append(panel_lstid)

    n = len(panels)
    fig = plt.figure(figsize=(16, 5 * n))
    gs  = gridspec.GridSpec(n, 2, width_ratios=[20, 0.7],
                            height_ratios=[1] * n, wspace=0.05)

    ax0 = None
    for i, draw in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0], sharex=ax0) if ax0 else fig.add_subplot(gs[i, 0])
        ax0 = ax0 or ax
        mappable = draw(ax)

        set_lat_ticks(ax)
        ax.set_xticks(time_tick_pos)
        ax.set_xlim(xlim)               # clip display to day D (drop the padding)
        last = (i == n - 1)
        if last:
            ax.set_xlabel("Time UT", fontsize=11)
            ax.set_xticklabels(time_tick_lbl, rotation=30)
        else:
            ax.set_xticklabels([])

        # panel letter (a), (b), ... — black on the keogram, white on label images
        ax.text(0.98, 0.97, f"({chr(ord('a') + i)})", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold",
                color="black" if i == 0 else "white")

        # colorbar only for the keogram panel
        if mappable is not None:
            gs_cb = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i, 1],
                                                     height_ratios=[0.5, 1, 0.5])
            fig.colorbar(mappable, cax=fig.add_subplot(gs_cb[1])).set_label("dTEC [TECu]", fontsize=10)

    plt.tight_layout()
    out_png = cfg.out_dir / f"{cfg.input_date}_roi.png"
    # bbox_inches="tight" crops the figure to its drawn content, removing the
    # large blank margins above the first panel and below the last.
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    logging.info("Saved -> %s", out_png)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

GRID_GLOB   = "keogram_grid_*.parquet"
GRID_PREFIX = "keogram_grid_"


def discover_dates(keo_folder, selector):
    """
    Return the sorted list of dates to process, by matching the grid files in
    `keo_folder` against `selector`:
      - None / ""    -> all available grids
      - "YYYY"       -> every grid in that year
      - "YYYY-MM"    -> every grid in that month
      - "YYYY-MM-DD" -> just that day (still resolved against the files on disk)
    """
    available = []
    for f in sorted(Path(keo_folder).glob(GRID_GLOB)):
        stem = f.name[len(GRID_PREFIX):].removesuffix(".parquet")
        try:
            available.append(dt.date.fromisoformat(stem))
        except ValueError:
            logging.warning("Skipping unparseable grid filename: %s", f.name)

    if not selector:                       # all
        return available
    prefix = selector                      # YYYY, YYYY-MM, or YYYY-MM-DD
    return [d for d in available if d.isoformat().startswith(prefix)]


def _add_config_flags(parser):
    """
    Add one --flag per scientific Config field, derived from the dataclass so the
    two cannot drift apart. Defaults are left as None rather than the field's
    default: cfg_from_args can then tell "user passed the default value" from "user
    passed nothing", and only the flags actually given are applied.
    """
    group = parser.add_argument_group(
        "detection parameters",
        "Override any default; the run folder is named after whichever ones differ.")
    for f in scientific_fields():
        group.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                           type=f.type, default=None,
                           help=f"(default: {_fmt(f.default)})")


def cfg_from_args(args, date=None):
    """Build a Config from parsed args, applying only the flags actually passed."""
    overrides = {f.name: getattr(args, f.name)
                 for f in scientific_fields()
                 if getattr(args, f.name) is not None}
    return Config(keo_folder=args.keo_folder, out_dir=args.out_dir,
                  input_date=date or Config.input_date, **overrides)


def parse_args(argv=None):
    """Parse command-line arguments. The single positional `date` selector takes
    one of four forms: omitted (all), YYYY, YYYY-MM, or YYYY-MM-DD."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Detect TID/LSTID ROIs in keogram grids.")
    parser.add_argument("date", nargs="?", default=None,
                        help="date selector: omit for ALL, or YYYY / YYYY-MM / YYYY-MM-DD")
    parser.add_argument("--keo-folder", type=Path, default=defaults.keo_folder,
                        help="folder containing keogram_grid_<date>.parquet")
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir,
                        help="parent output folder; results go in a <out-dir>/<run-name> "
                             "subfolder named after the non-default parameters")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print detailed per-date logs; without it, only a progress bar "
                             "and a final summary are shown")
    parser.add_argument("--debug", action="store_true",
                        help="add all intermediate diagnostic panels to the output figure")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip plotting entirely; only write the ROI pickle (much faster "
                             "for a parameter sweep)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel worker processes for a batch (default: %(default)s; "
                             "use 1 for serial)")
    _add_config_flags(parser)
    args = parser.parse_args(argv)
    _validate_selector(parser, args.date)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.debug and args.no_plot:
        parser.error("--debug adds diagnostic panels to the figure, but --no-plot "
                     "skips the figure; pass at most one")
    return args


def _validate_selector(parser, selector):
    """Reject malformed selectors early with a clear CLI error."""
    if selector is None:
        return
    parts = selector.split("-")
    ok = (
        (len(parts) == 1 and len(parts[0]) == 4 and parts[0].isdigit()) or         # YYYY
        (len(parts) == 2 and len(parts[0]) == 4) or                                # YYYY-MM
        (len(parts) == 3)                                                          # YYYY-MM-DD
    )
    if not ok:
        parser.error(f"invalid date selector '{selector}'; "
                     "expected YYYY, YYYY-MM, or YYYY-MM-DD (or omit for all)")
    if len(parts) == 3:                    # full date: validate it is a real date
        try:
            dt.date.fromisoformat(selector)
        except ValueError as e:
            parser.error(f"invalid date '{selector}': {e}")


def run(cfg, debug=False, do_plot=True):
    """Run the full detection pipeline for one keogram, returning the matched
    LSTID lines. When `debug` is True the output figure includes all the
    intermediate diagnostic panels; when `do_plot` is False no figure is produced.
    Raises on I/O or validation errors (caught in main)."""
    t_start = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load (padded with neighbour-day borders to avoid midnight edge effects)
    logging.info("Loading grid %s ...", cfg.grid_path)
    Z, lats, times_sec, day_mask = load_grid(cfg)
    logging.info("Grid shape: %s  (lat x time)%s", Z.shape,
                 "" if day_mask.all() else f"  (+{(~day_mask).sum()} pad cols)")

    # 2. Pre-process
    Z_ds = preprocess(Z, cfg)
    logging.info("Processed shape: %s", Z_ds.shape)

    # 3 & 4. Detect and filter ROIs
    logging.info("Detecting ROIs ...")
    t0  = time.time()
    roi = detect_roi(Z_ds, cfg)
    logging.info("Unfiltered: %d  [%.1fs]", roi.max(), time.time() - t0)
    roi_filtered = filter_roi(roi, Z_ds, cfg)
    logging.info("Filtered:   %d", roi_filtered.max())

    # 5. Fit RANSAC lines on the full padded keogram (so border waves are whole).
    #    Padding ROIs are kept for now so LSTID matching can still pair a day-D ROI
    #    with a partner that lies just across midnight; they are dropped after.
    logging.info("Fitting RANSAC lines ...")
    lines      = fit_lines(roi_filtered, Z_ds, lats, times_sec, cfg)
    midnight_d = dt.datetime.combine(cfg.input_date, dt.time())
    in_day_d   = lambda r: 0 <= (r["ref_time"] - midnight_d).total_seconds() < SECONDS_PER_DAY
    lines_day  = [r for r in lines if in_day_d(r)]
    logging.info("Lines fitted: %d (day %s)%s", len(lines_day), cfg.input_date,
                 "" if len(lines) == len(lines_day) else f" + {len(lines) - len(lines_day)} pad")
    for r in lines_day:
        logging.info("  ROI %3d  %s-%s  lat %.1f->%.1f  speed=%.0f m/s",
                     r["roi_label"], r["start_time"].strftime("%H:%M"),
                     r["end_time"].strftime("%H:%M"), r["start_lat"],
                     r["end_lat"], r["speed_m_s"])

    # 6. Match LSTIDs on the full set (so a day-D ROI can pair across midnight),
    #    then keep only the matches whose own ref_time is in day D.
    logging.info("Matching LSTIDs ...")
    lstid_lines = [r for r in match_lstid(lines, cfg) if in_day_d(r)]
    logging.info("LSTID ROIs: %d", len(lstid_lines))
    for r in lstid_lines:
        logging.info("  ROI %3d  %s  amp=%s  speed=%.0f m/s  period=%.0f min",
                     r["roi_label"], r["ref_time"].strftime("%H:%M"),
                     "+" if r["mean_amplitude"] > 0 else "-",
                     r["speed_m_s"], r["period_s"] / 60)

    t_process = time.time() - t_start

    # 7. Plot — day-D ROIs only, x-axis clipped to day D
    t_plot = 0.0
    if do_plot:
        logging.info("Plotting ...")
        t_plot_start = time.time()
        lstid_labels = {r["roi_label"] for r in lstid_lines}
        roi_lstid    = roi_filtered.copy()
        np.putmask(roi_lstid, ~np.isin(roi_lstid, list(lstid_labels)), 0)
        plot(Z_ds, roi, roi_filtered, roi_lstid, lines_day, lstid_lines,
             lats, times_sec, day_mask, cfg, debug)
        t_plot = time.time() - t_plot_start

    logging.info("Timing:  processing = %.2fs   plotting/saving = %.2fs", t_process, t_plot)
    return lstid_lines


def _worker_init(verbose):
    """Initializer run once per worker process: configure logging in the child.
    Without --verbose, workers stay silent (ERROR level) so only the parent's
    progress bar is shown."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.ERROR,
                        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
                        datefmt="%H:%M:%S")


def _roi_record(date, r):
    """Reduce one LSTID line dict to picklable summary characteristics (drops
    the 'x'/'y' RANSAC pixel-coordinate arrays), tagged with its source date."""
    return {"date": date, **{k: v for k, v in r.items() if k not in ("x", "y")}}


def _process_date(job):
    """
    Top-level (picklable) worker: run the pipeline for one date in a separate
    process. Never raises — returns (date, roi_records, error_str_or_None) so a
    single bad grid cannot tear down the pool.

    `base_cfg` already carries this run's parameters and has out_dir pointed at the
    run folder; only the date changes per job.
    """
    base_cfg, date, debug, do_plot = job
    cfg = replace(base_cfg, input_date=date)
    try:
        lstid_lines = run(cfg, debug, do_plot)
        return date, [_roi_record(date, r) for r in lstid_lines], None
    except Exception as e:                            # isolate one bad grid from the rest
        return date, [], f"{type(e).__name__}: {e}"


def main(argv=None):
    args = parse_args(argv)
    # Verbose: detailed per-date logs, no progress bar. Otherwise: stay silent
    # (ERROR level) and show only a progress bar + a final summary.
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    dates = discover_dates(args.keo_folder, args.date)
    if not dates:
        # always surface this, even when quiet — there is nothing to do
        print(f"No keogram grids match selector {args.date or '(all)'!r} in {args.keo_folder}")
        return 1

    scope   = args.date or "ALL"
    workers = min(args.workers, len(dates))           # never spawn more workers than jobs

    # Resolve this run's folder from the non-default parameters and point the
    # config's out_dir at it, so every per-date output lands inside automatically.
    base_cfg = cfg_from_args(args)
    out_dir  = run_dir(base_cfg)
    base_cfg = replace(base_cfg, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(base_cfg, out_dir / "config.json")

    logging.info("Batch: %d keogram(s) for selector '%s'  (%d worker%s)",
                 len(dates), scope, workers, "" if workers == 1 else "s")
    print(f"Run '{run_name(base_cfg)}' -> {out_dir}")

    jobs = [(base_cfg, d, args.debug, not args.no_plot) for d in dates]
    n_ok = n_fail = total_lstid = 0
    failures = []
    all_records = []
    t_batch = time.time()

    # progress bar only when not verbose (a live bar and log spam fight each other)
    bar = tqdm(total=len(jobs), unit="keo", disable=args.verbose,
               desc=f"{scope} ({workers}w)")

    def record(date, roi_records, err):
        nonlocal n_ok, n_fail, total_lstid
        if err is None:
            n_ok += 1
            total_lstid += len(roi_records)
            all_records.extend(roi_records)
            logging.info("DONE  %s  (%d LSTID ROIs)", date, len(roi_records))
        else:
            n_fail += 1
            failures.append((date, err))
            logging.error("FAILED %s: %s", date, err)
        bar.update(1)
        bar.set_postfix(ok=n_ok, fail=n_fail, lstid=total_lstid)

    if workers == 1:                                  # serial path: ordered logs, no pool overhead
        for job in jobs:
            record(*_process_date(job))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_worker_init, initargs=(args.verbose,)) as ex:
            futures = {ex.submit(_process_date, job): job[2] for job in jobs}
            for fut in as_completed(futures):
                record(*fut.result())

    bar.close()

    # combined pickle of every LSTID ROI's characteristics across the whole batch;
    # only a run over all days writes the main catalogue, a partial or failed run
    # gets its own file so it can never overwrite it
    tag = (f"_{args.date}" if args.date else "") + ("_partial" if n_fail else "")
    pkl_path = out_dir / f"lstid_rois{tag}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_records, f)
    logging.info("Saved -> %s (%d ROIs)", pkl_path, len(all_records))

    # final summary: always printed (unconditional), regardless of verbosity
    print(f"Batch done in {time.time() - t_batch:.1f}s: "
          f"{n_ok} ok, {n_fail} failed, {total_lstid} total LSTID ROIs")
    print(f"Saved ROI characteristics -> {pkl_path}")
    for d, e in failures:
        print(f"  failed: {d}  ({e})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
