import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\MarcoGuerra\Documents\PYTHON\roi-identification")
sys.path.insert(0, str(ROOT / "analysis"))
import climatology as c

records = c.load_records(ROOT / "results" / "roi" / "lstid_rois.pkl")
years = sorted({r["ref_time"].year for r in records})
hours = np.array([r["ref_time"].hour for r in records])

sme = c.load_index_by_hour(years, "sme")  # same values as the hourly plot, before _minmax

print(f"years: {years[0]}-{years[-1]}  n_rois={len(records)}")
print("hour  mean_SME[nT]  norm")
for h, (v, n) in enumerate(zip(sme, c._minmax(sme))):
    print(f"{h:4d}  {v:12.1f}  {n:.2f}")
print(f"min={np.nanmin(sme):.1f}  max={np.nanmax(sme):.1f}  "
      f"range={np.nanmax(sme) - np.nanmin(sme):.1f} nT  "
      f"(max/min = {np.nanmax(sme) / np.nanmin(sme):.2f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(np.arange(24), np.bincount(hours, minlength=24), color="tab:blue")
ax.set_xticks(np.arange(0, 24, 2))
ax.set_xlabel("Hour (UT)")
ax.set_ylabel("LSTID ROI count", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_title(f"Hourly LSTID ROIs vs mean SME, unscaled (n={len(records)})")
ax.grid(axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax2 = ax.twinx()
ax2.plot(np.arange(24), sme, color="tab:red", marker="o", markersize=4,
         linewidth=1.5, label="SME (hourly mean)")
ax2.set_ylim(0, np.nanmax(sme) * 1.1)  # zero-based so the swing's true size shows
ax2.set_ylabel("SME [nT]", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout()
out = Path(__file__).with_name("climatology_hourly_sme_raw.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
