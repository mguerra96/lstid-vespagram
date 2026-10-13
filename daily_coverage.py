"""
daily_coverage.py

The daily coverage table released with the dataset: for each day, the number of
stations, of valid observations and of valid arcs in the TEC observation file.
A valid observation is a row with a detrended vertical TEC value; an arc is one
continuous receiver-satellite pass, and its id starts with the station name.

  python daily_coverage.py                 every tec_*.parquet in TEC_DIR (paths.py); days already
                                           in the table are skipped, so an interrupted run resumes
  python daily_coverage.py tec_2024-08-12.parquet ...    only print the rows of these files

Output: results/vespagram/daily_coverage.csv
"""

import sys
from pathlib import Path

import polars as pl

from paths import ROOT, TEC_DIR

OUT = ROOT / "results/vespagram/daily_coverage.csv"


def coverage(path):
    """(date, n_stations, n_observations, n_valid_arcs) of one daily TEC file."""
    valid = pl.col("vtec_detrended").is_not_null() & pl.col("vtec_detrended").is_not_nan()
    n = (pl.scan_parquet(path).filter(valid)
           .select(pl.col("id_arc_valid").str.split("_").list.first().n_unique(),   # station name
                   pl.len(),
                   pl.col("id_arc_valid").n_unique().alias("arcs"))
           .collect().row(0))
    return (Path(path).stem[4:], *n)


def main():
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            print(",".join(map(str, coverage(f))))
        return
    if not TEC_DIR.exists():
        raise FileNotFoundError(f"TEC files not found: {TEC_DIR}")
    if not OUT.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("date,n_stations,n_observations,n_valid_arcs\n", encoding="utf-8")
    done = set(pl.read_csv(OUT, schema_overrides={"date": pl.String})["date"])
    files = [f for f in sorted(TEC_DIR.glob("tec_*.parquet")) if f.stem[4:] not in done]
    print(f"{len(done)} days already in the table, {len(files)} to do")
    with open(OUT, "a", encoding="utf-8") as fh:
        for f in files:
            fh.write(",".join(map(str, coverage(f))) + "\n")
            fh.flush()
    pl.read_csv(OUT, schema_overrides={"date": pl.String}).sort("date").write_csv(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
