"""
paper/paper_numbers.py

Every number quoted in the Scientific Data draft, recomputed from the final
catalogue, so the text can be refreshed whenever the catalogue changes.

  python paper/paper_numbers.py            -> prints a summary, writes paper/numbers.json
"""

import sys
import json
import pickle
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]
import vespagram as v
from detect_roi import _read_grid, discover_dates
from catalogue_records import roi_trains, overlap_match, ROI_PKL
import compare_climatology as cc

OUT = Path(__file__).resolve().parent / "numbers.json"
HM  = ("high", "medium")


def med_iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q = np.percentile(x, [25, 50, 75])
    return {"median": round(float(q[1]), 3), "q25": round(float(q[0]), 3), "q75": round(float(q[2]), 3)}


def main():
    # keep the sections written by the other scripts (hfint, published_roi, method_checks)
    N = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    cfg = v.VespaConfig()

    # ── input grid ──
    days = discover_dates(cfg.keo_folder, None)
    Z, lats, t = _read_grid(cfg.keo_folder / f"keogram_grid_{days[len(days) // 2]}.parquet")
    N["input"] = {"n_days": len(days), "first": str(days[0]), "last": str(days[-1]),
                  "native_lat_step_deg": float(np.round(np.median(np.abs(np.diff(lats))), 4)),
                  "native_time_step_s": float(np.median(np.diff(t))),
                  "lat_range": [float(lats.min()), float(lats.max())],
                  "longitude": cfg.longitude}

    # ── catalogue ──
    tr = pl.read_parquet(ROOT / "results/vespagram/trains.parquet")
    de = pl.read_parquet(ROOT / "results/vespagram/detections.parquet")
    tr = tr.with_columns(((pl.col("t_end") - pl.col("t_start")).dt.total_minutes() / 60).alias("dur_h"))
    hm = tr.filter(pl.col("confidence").is_in(HM))
    N["catalogue"] = {
        "n_trains": tr.height, "n_detections": de.height,
        "tiers": dict(tr.group_by("confidence").len().iter_rows()),
        "n_high_medium": hm.height,
        "share_single_detection_hm": round(float((hm["n_detections"] == 1).mean()), 3),
        "share_single_array_all": round(float((tr["coverage"] <= 1 / 3 + 1e-9).mean()), 3),
        "n_hm_longer_12h": int((hm["dur_h"] > 12).sum()),
        "share_hm_longer_12h": round(float((hm["dur_h"] > 12).mean()), 3),
        "max_duration_h": round(float(tr["dur_h"].max()), 1),
        "speed_halfwidth_hm_pct": round(float(((hm["speed_hi_m_s"] - hm["speed_lo_m_s"])
                                               / 2 / hm["speed_m_s"]).median() * 100), 1),
        "trains_per_year_hm": {int(y): int(n) for y, n in
                               hm.group_by(pl.col("date").dt.year()).len().sort("date").iter_rows()},
    }
    N["characteristics_hm"] = {k: med_iqr(hm[c].to_numpy()) for k, c in
                               [("speed_m_s", "speed_m_s"), ("period_min", "period_min"),
                                ("wavelength_km", "wavelength_km"), ("amplitude_tecu", "amplitude"),
                                ("duration_h", "dur_h"), ("coverage", "coverage"),
                                ("n_detections", "n_detections")]}
    N["characteristics_by_tier"] = {
        c: {k: med_iqr(tr.filter(pl.col("confidence") == c)[k].to_numpy())["median"]
            for k in ("speed_m_s", "period_min", "amplitude", "semblance", "coverage", "dur_h")}
        for c in ("high", "medium", "low")}

    # ── occupancy and agreement with the ROI catalogue ──
    grid = cc.common_grid()
    ys = {k: cc.outcome(grid, cc.load(k, "hours")) for k in ("roi", "vespagram", "vespagram_hm")}
    yr = pd.to_datetime(grid.to_numpy(), utc=True).year.to_numpy()
    agree = {}
    for label, m in [("2015-2025", np.ones_like(yr, bool)), ("2021-2024", (yr >= 2021) & (yr <= 2024))]:
        a = {k: float(ys[k][m].mean() * 100) for k in ys}
        for k in ("vespagram", "vespagram_hm"):
            both = (ys["roi"][m] & ys[k][m]).sum()
            a[f"roi_hours_in_{k}_pct"] = float(both / ys["roi"][m].sum() * 100)
            a[f"{k}_hours_in_roi_pct"] = float(both / ys[k][m].sum() * 100)
        agree[label] = {k: round(x, 1) for k, x in a.items()}
    N["occupancy_pct_and_agreement"] = agree
    N["catalogue"]["active_h_per_year_hm"] = {int(y): int(ys["vespagram_hm"][yr == y].sum())
                                              for y in np.unique(yr)}
    N["catalogue"]["median_duration_h_per_year_hm"] = {
        int(y): round(float(d), 1) for y, d in
        hm.group_by(pl.col("date").dt.year()).agg(pl.col("dur_h").median()).sort("date").iter_rows()}

    roi = roi_trains()
    rs = [r["start_time"] for r in roi]; re_ = [r["end_time"] for r in roi]
    ry = np.array([r["ref_time"].year for r in roi])
    hs = hm["t_start"].to_list(); he = hm["t_end"].to_list()
    hy = (hm["t_start"] + (hm["t_end"] - hm["t_start"]) / 2).dt.year().to_numpy()
    roi_hit = overlap_match(rs, re_, hs, he)
    hm_hit = overlap_match(hs, he, rs, re_)
    sel_r, sel_h = (ry >= 2021) & (ry <= 2024), (hy >= 2021) & (hy <= 2024)
    N["train_matching"] = {
        "n_roi_fronts": len(pickle.load(open(ROI_PKL, "rb"))), "n_roi_trains": len(roi),
        "roi_trains_matched_by_hm_pct": round(float(roi_hit.mean() * 100), 1),
        "hm_trains_matched_by_roi_pct": round(float(hm_hit.mean() * 100), 1),
        "2021-2024": {"roi_trains": int(sel_r.sum()), "hm_trains": int(sel_h.sum()),
                      "roi_trains_matched_by_hm_pct": round(float(roi_hit[sel_r].mean() * 100), 1),
                      "hm_trains_matched_by_roi_pct": round(float(hm_hit[sel_h].mean() * 100), 1)},
    }

    ts = pd.to_datetime(grid.to_numpy(), utc=True)
    shape = {}
    for name, key in [("ut_hour", ts.hour.to_numpy()), ("month", ts.month.to_numpy()), ("year", ts.year.to_numpy())]:
        lv = np.unique(key)
        c = [[ys[k][key == x].mean() for x in lv] for k in ("roi", "vespagram_hm")]
        shape[name] = round(float(np.corrcoef(*c)[0, 1]), 2)
    N["climatology_shape_corr_roi_vs_hm"] = shape

    # ── tiers against drivers (from compare_climatology / ratio_climatology outputs) ──
    d = pd.read_csv(ROOT / "results/climatology_compare/drivers.csv")
    N["driver_effects"] = {
        drv: {r.catalogue: {"coef": round(r.coef, 2), "ci": [round(r.ci_lo, 2), round(r.ci_hi, 2)],
                            "dauc": round(r.dauc, 3)}
              for r in d[d.driver == drv].itertuples()}
        for drv in ("SME", "Hp30", "Newell", "IE", "SymH")}
    r = pd.read_csv(ROOT / "results/climatology_compare/ratio.csv")
    hp = r[r.group == "Hp30, 3 h mean"]
    N["ratio_vs_hp30"] = {row.level: {"occ_roi": round(row.occ_roi, 2), "occ_hm": round(row.occ_vespagram_hm, 2),
                                      "active_h_roi": round(row.occ_roi * row.hours / 100),
                                      "active_h_hm": round(row.occ_vespagram_hm * row.hours / 100),
                                      "ratio_hm": round(row.ratio_vespagram_hm, 2),
                                      "ci_hm": [round(row.lo_vespagram_hm, 2), round(row.hi_vespagram_hm, 2)]}
                          for row in hp.itertuples()}

    # ── quiet-time trains: vespagram-only vs also found by ROI ──
    import quiet_check as qc
    mid = (hm["t_start"] + (hm["t_end"] - hm["t_start"]) / 2).to_list()
    q = hm.with_columns(pl.Series("hp3h", qc.hp30_at(mid))).filter(pl.col("hp3h") < qc.HP30_MAX)
    q = q.with_columns(pl.Series("roi_match", overlap_match(
        q["t_start"].to_list(), q["t_end"].to_list(), rs, re_)))
    N["quiet_time_hm"] = {
        "hp30_max": qc.HP30_MAX, "n": q.height,
        **{("roi_also" if k else "vespagram_only"): {
            "n": g.height, "amplitude_med": round(float(g["amplitude"].median()), 4),
            "semblance_med": round(float(g["semblance"].median()), 3),
            "speed_med": round(float(g["speed_m_s"].median())),
            "share_high": round(float((g["confidence"] == "high").mean()), 2)}
           for k, g in ((True, q.filter(pl.col("roi_match"))), (False, q.filter(~pl.col("roi_match"))))}}

    # ── case events ──
    cases = {}
    for name, a0, a1 in [("2024-08-11/12", "2024-08-11 12:00", "2024-08-13 00:00"),   # the windows of Fig. 4
                         ("2015-03-17/18", "2015-03-17 00:00", "2015-03-18 03:00"),
                         ("2024-05-10/11", "2024-05-10 12:00", "2024-05-11 12:00")]:
        s = tr.filter((pl.col("t_end") > dt.datetime.fromisoformat(a0))
                      & (pl.col("t_start") < dt.datetime.fromisoformat(a1))).sort("t_start")
        cases[name] = [{"t": f"{a:%d %H:%M}-{b:%d %H:%M}", "speed": round(sp),
                        "speed_lo_hi": [round(lo), round(hi)], "period": round(p),
                        "conf": c, "coverage": round(cv, 2)}
                       for a, b, sp, lo, hi, p, c, cv in s.select(
                           "t_start", "t_end", "speed_m_s", "speed_lo_m_s", "speed_hi_m_s",
                           "period_min", "confidence", "coverage").iter_rows()]
    N["case_events"] = cases
    f1 = de.filter((pl.col("date") == dt.date(2024, 8, 12)) & (pl.col("train_id") == 1))
    N["fig1_first_train"] = {"n_detections": f1.height,      # the example in Methods
                             "speed_range_m_s": [round(f1["speed_m_s"].min()), round(f1["speed_m_s"].max())]}

    OUT.write_text(json.dumps(N, indent=1, default=str), encoding="utf-8")
    print(json.dumps(N, indent=1, default=str))


if __name__ == "__main__":
    main()
