"""
supermag.py

The SuperMAG indices SME, SML and SMU may not be redistributed, so the driver table of
the Zenodo record comes without them. read_features() reads the driver table and, the
first time a SuperMAG column is asked for and missing, downloads the indices from
SuperMAG with the user's own account and adds them to the table:

  - a free SuperMAG registration is needed (https://supermag.jhuapl.edu); the username
    is taken from the environment variable SUPERMAG_USERID, or asked for;
  - 2015-2025 are fetched in 14-day requests (the service rejects longer ones), one
    file per year in results/supermag/, so an interrupted download resumes;
  - spikes are removed as in the released table: a point more than 10 scaled MADs from
    the 30-min centred running median is set to null.

  python supermag.py          download now (otherwise it happens on first use)

Please acknowledge SuperMAG as required by its rules of the road.
"""

import os
import json
import time
import datetime as dt
import urllib.request

import numpy as np
import pandas as pd
import polars as pl

from paths import ROOT, FEATURES

COLUMNS = ("sme", "sml", "smu")
YEARS = range(2015, 2026)
CACHE = ROOT / "results/supermag"
URL = ("https://supermag.jhuapl.edu/services/indices.php?python&nohead"
       "&start={start}&logon={user}&extent={extent:012d}&indices=sme,sml,smu")
CHUNK_DAYS, RETRIES = 14, 4


def _fetch_year(year, user):
    """One year of 1-min SME/SML/SMU as a polars frame [time, sme, sml, smu]."""
    rows, day, end = [], dt.date(year, 1, 1), dt.date(year, 12, 31)
    while day <= end:
        n = min(CHUNK_DAYS, (end - day).days + 1)
        url = URL.format(start=f"{day:%Y-%m-%d}T00:00", user=user, extent=n * 86400)
        for attempt in range(1, RETRIES + 1):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    body = r.read()
                if body.startswith(b"ERROR"):
                    raise RuntimeError(body[:200].decode(errors="replace"))
                rows += json.loads(body) if len(body) > 3 else []
                break
            except Exception as e:
                if attempt == RETRIES:
                    raise RuntimeError(f"SuperMAG request for {day} failed: {e}") from e
                time.sleep(8 * attempt)
        day += dt.timedelta(days=n)
    return (pl.DataFrame(rows)
            .select(pl.from_epoch(pl.col("tval").cast(pl.Int64)).dt.replace_time_zone("UTC").alias("time"),
                    *[pl.col(c.upper()).cast(pl.Float64).alias(c) for c in COLUMNS]))


def _despike(s, window="30min", n_mad=10.0):
    """Null points more than n_mad scaled MADs from the centred running median."""
    med = s.rolling(window, center=True, min_periods=3).median()
    dev = (s - med).abs()
    mad = 1.4826 * dev.rolling(window, center=True, min_periods=3).median()
    return s.mask((mad > 0) & (dev > n_mad * mad))


def download():
    """Fetch the indices (resuming from results/supermag/) and add them to FEATURES."""
    user = (os.environ.get("SUPERMAG_USERID") or input("SuperMAG username: ")).strip()
    CACHE.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        f = CACHE / f"supermag_{y}.parquet"
        if not f.exists():
            print(f"SuperMAG {y} ...", flush=True)
            _fetch_year(y, user).write_parquet(f)
    sm = pl.concat([pl.read_parquet(CACHE / f"supermag_{y}.parquet") for y in YEARS]).to_pandas()
    sm = sm.set_index("time").sort_index()
    sm = pl.from_pandas(sm.apply(_despike).reset_index())
    feats = pl.read_parquet(FEATURES, memory_map=False).drop(COLUMNS, strict=False)
    feats = feats.join(sm.with_columns(pl.col("time").cast(feats["time"].dtype)), on="time", how="left")
    tmp = FEATURES.with_suffix(".tmp.parquet")
    feats.write_parquet(tmp, compression="zstd")
    tmp.replace(FEATURES)                     # replace only once the new table is complete
    print(f"added {', '.join(COLUMNS)} to {FEATURES}")


def read_features(columns):
    """pl.read_parquet(FEATURES, columns=...), downloading SuperMAG first if needed."""
    if set(columns) & set(COLUMNS) and not set(COLUMNS) <= set(pl.read_parquet_schema(FEATURES)):
        download()
    return pl.read_parquet(FEATURES, columns=list(columns))


if __name__ == "__main__":
    download()
