"""
keogram_from_tec.py

Rebuild the daily keogram grids from daily TEC files with the pykeo keogram code,
for the figures when the keogram archive is not at hand.

  python paper/keogram_from_tec.py                       every paper/external/tec_*.parquet
  python paper/keogram_from_tec.py path/tec_2024-08-12.parquet

Output: paper/external/keo/keogram_grid_YYYY-MM-DD.parquet (and a quick-look png).
make_figures.py reads this folder when KEO_DIR points to it:
  KEO_DIR=paper/external/keo python paper/make_figures.py 1
"""

import sys
import datetime as dt
from pathlib import Path

import polars as pl

HERE  = Path(__file__).resolve().parent
PYKEO = Path(r"C:\Users\MarcoGuerra\Documents\PYTHON\pykeo\scripts")
OUT   = HERE / "external/keo"
sys.path.insert(0, str(PYKEO))
from pykeo_lib.keogram import plot_keogram


def main():
    files = [Path(f) for f in sys.argv[1:]] or sorted((HERE / "external").glob("tec_*.parquet"))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in files:
        day = dt.date.fromisoformat(f.stem[4:])
        plot_keogram(pl.read_parquet(f), OUT / f"keogram_{day}.png", input_date=day,
                     grid_out_path=OUT / f"keogram_grid_{day}.parquet")
        print("wrote", OUT / f"keogram_grid_{day}.parquet")


if __name__ == "__main__":
    main()
