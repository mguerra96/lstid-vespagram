import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(r"C:\Users\MarcoGuerra\Documents\PYTHON\roi-identification")
sys.path.insert(0, str(ROOT / "analysis"))
import climatology as c

records = c.load_records(ROOT / "results" / "roi" / "lstid_rois.pkl")
years = np.array([r["ref_time"].year for r in records])
year_range = np.arange(years.min(), years.max() + 1)
counts = np.array([(years == y).sum() for y in year_range])

df = (c._features(["sme"]).with_columns(pl.col("time").dt.year().alias("year"))
        .filter(pl.col("year").is_in(year_range.tolist()))
        .group_by("year").agg(pl.col("sme").mean().alias("mean"),
                              pl.col("sme").median().alias("median")))
sme_mean = dict(zip(df["year"], df["mean"]))
sme_med = dict(zip(df["year"], df["median"]))
mean_vals = np.array([sme_mean.get(y, np.nan) for y in year_range])
med_vals = np.array([sme_med.get(y, np.nan) for y in year_range])
f107 = c.load_f107_by_year(year_range.tolist())
f107_vals = np.array([f107.get(y, np.nan) for y in year_range])

print("year  rois  sme_mean  sme_median  f107")
for row in zip(year_range, counts, mean_vals, med_vals, f107_vals):
    print("{}  {:5d}  {:8.1f}  {:10.1f}  {:5.1f}".format(*row))
for name, v in [("SME mean", mean_vals), ("SME median", med_vals), ("F10.7", f107_vals)]:
    r, pr = pearsonr(counts, v)
    rho, ps = spearmanr(counts, v)
    print(f"ROI count vs {name:10s}: Pearson r={r:+.2f} (p={pr:.2f})  "
          f"Spearman rho={rho:+.2f} (p={ps:.2f})  n={len(year_range)}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(year_range, counts, color="tab:green")
ax.set_xticks(year_range)
ax.set_xticklabels(year_range, rotation=45)
ax.set_xlabel("Year")
ax.set_ylabel("LSTID ROI count", color="tab:green")
ax.tick_params(axis="y", labelcolor="tab:green")
ax.set_title(f"Yearly distribution of LSTID ROIs (n={len(records)})")
ax.grid(axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax2 = ax.twinx()
ax2.plot(year_range, mean_vals, color="tab:red", marker="o", markersize=5,
         linewidth=1.5, label="SME (mean)")
ax2.plot(year_range, med_vals, color="tab:red", marker="o", markersize=4,
         linewidth=1.2, linestyle="--", alpha=0.6, label="SME (median)")
ax2.set_ylim(0, None)
ax2.set_ylabel("SME [nT]", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.legend(loc="upper center", fontsize=8)
fig.tight_layout()
out = Path(__file__).with_name("climatology_yearly_sme.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
