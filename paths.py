"""
paths.py

Where the inputs live. By default in data/, laid out as the Zenodo record unpacks:

  data/keograms/keogram_grid_YYYY-MM-DD.parquet   (the keograms_YYYY.zip files, unzipped)
  data/drivers_1min.parquet

Set the environment variable of the same name to use another location, e.g.

  KEO_DIR=/data/keograms python vespagram.py 2024-05-11
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# daily keogram grids, keogram_grid_YYYY-MM-DD.parquet (Zenodo record)
KEO_DIR = Path(os.environ.get("KEO_DIR", DATA / "keograms"))
# 1-min driver table (Zenodo record; SuperMAG columns are added by supermag.py)
FEATURES = Path(os.environ.get("FEATURES", DATA / "drivers_1min.parquet"))
# daily TEC observation files from pykeo, tec_YYYY-MM-DD.parquet (not released, about
# 0.9 TB; only daily_coverage.py and the empty-cell checks of paper/method_checks.py)
TEC_DIR = Path(os.environ.get("TEC_DIR", DATA / "tec"))
# scripts folder of a pykeo clone next to this repository (only paper/keogram_from_tec.py)
PYKEO = Path(os.environ.get("PYKEO", ROOT.parent / "pykeo" / "scripts"))
