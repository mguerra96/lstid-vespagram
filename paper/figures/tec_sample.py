"""
paper/figures/tec_sample.py

Sample the pykeo TEC observation files (not released; ~900 GB) once and cache
the small summaries that Figure 2 needs, in paper/figures/data/:

  constellations.csv   valid observations per constellation, one day per week
  ipp_density.npz      pierce-point counts on a 0.5 deg grid for two June days

A valid observation is a row with a detrended vertical TEC value.

  python paper/figures/tec_sample.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

TEC_DIR = Path("D:/pykeo/results/tec_data")
DATA    = Path(__file__).resolve().parent / "data"
STEP    = dt.timedelta(days=7)
MAP_DAYS = ["2015-06-15", "2025-06-15"]
LON_EDGES = np.arange(-15, 40.01, 0.5)
LAT_EDGES = np.arange(28, 68.01, 0.5)


def read(day, columns):
    return pl.read_parquet(TEC_DIR / f"tec_{day}.parquet", columns=columns + ["vtec_detrended"]) \
             .filter(pl.col("vtec_detrended").is_not_null() & pl.col("vtec_detrended").is_not_nan())


def constellations():
    rows, day = [], dt.date(2015, 1, 1)
    while day <= dt.date(2025, 12, 31):
        f = TEC_DIR / f"tec_{day}.parquet"
        if f.exists():
            n = (read(day, ["sv"]).group_by(pl.col("sv").str.slice(0, 1).alias("c")).len())
            rows.append({"date": day, **dict(zip(n["c"], n["len"]))})
        day += STEP
    out = pl.DataFrame(rows).fill_null(0)
    out.write_csv(DATA / "constellations.csv")
    print("wrote", DATA / "constellations.csv", out.height, "days")


def ipp_density():
    maps = {}
    for day in MAP_DAYS:
        df = read(day, ["lat_ipp", "lon_ipp"])
        H, _, _ = np.histogram2d(df["lon_ipp"].to_numpy(), df["lat_ipp"].to_numpy(),
                                 bins=[LON_EDGES, LAT_EDGES])
        maps[day] = H.T                       # rows = latitude
    np.savez(DATA / "ipp_density.npz", lon=LON_EDGES, lat=LAT_EDGES,
             days=np.array(MAP_DAYS), **{f"d{i}": maps[d] for i, d in enumerate(MAP_DAYS)})
    print("wrote", DATA / "ipp_density.npz")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    ipp_density()
    constellations()
