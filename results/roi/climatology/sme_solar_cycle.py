import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\MarcoGuerra\Documents\PYTHON\roi-identification")
sys.path.insert(0, str(ROOT / "analysis"))
import climatology as c

df = c._features(["sme", c.F107_COL])
print(f"time span: {df['time'].min()} -> {df['time'].max()}  rows={df.height}")

monthly = (df.group_by_dynamic("time", every="1mo")
             .agg(pl.col("sme").mean().alias("sme"),
                  pl.col(c.F107_COL).mean().alias("f107"),
                  pl.col("sme").is_not_null().mean().alias("cov"))
             .sort("time")
             .with_columns(pl.col("sme").rolling_mean(13, center=True).alias("sme13"),
                           pl.col("f107").rolling_mean(13, center=True).alias("f10713")))

yearly = (df.group_by(pl.col("time").dt.year().alias("year"))
            .agg(pl.col("sme").mean().alias("sme"),
                 pl.col("sme").median().alias("sme_median"),
                 pl.col(c.F107_COL).mean().alias("f107"))
            .sort("year"))
print(yearly)

sme_min = monthly.filter(pl.col("sme13").is_not_null()).sort("sme13").row(0, named=True)
sme_max = monthly.filter(pl.col("sme13").is_not_null()).sort("sme13").row(-1, named=True)
print(f"13-mo smoothed SME min: {sme_min['sme13']:.1f} nT at {sme_min['time']:%Y-%m}")
print(f"13-mo smoothed SME max: {sme_max['sme13']:.1f} nT at {sme_max['time']:%Y-%m}")
print(f"low-coverage months (<50% SME samples): "
      f"{monthly.filter(pl.col('cov') < 0.5)['time'].dt.strftime('%Y-%m').to_list()}")

t = monthly["time"].to_numpy()
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t, monthly["sme"], color="tab:red", alpha=0.35, linewidth=1, label="SME monthly mean")
ax.plot(t, monthly["sme13"], color="tab:red", linewidth=2.2, label="SME 13-month running mean")
ax.set_ylim(0, None)
ax.set_ylabel("SME [nT]", color="tab:red")
ax.tick_params(axis="y", labelcolor="tab:red")
ax.set_xlabel("Year")
ax.grid(linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax2 = ax.twinx()
ax2.plot(t, monthly["f10713"], color="dimgray", linewidth=1.8, linestyle="--",
         label="F10.7 13-month running mean")
ax2.set_ylim(0, None)
ax2.set_ylabel("F10.7 [sfu]", color="dimgray")
ax2.tick_params(axis="y", labelcolor="dimgray")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.8)
ax.set_title("Mean SME across the solar cycle (2015-2025)")
fig.tight_layout()
out = Path(__file__).with_name("sme_solar_cycle.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
