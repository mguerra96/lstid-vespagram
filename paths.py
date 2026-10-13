"""
paths.py

Where the inputs live. The defaults are the author's machine; set the environment
variable of the same name to use other folders, e.g.

  KEO_DIR=/data/keograms python vespagram.py 2024-05-11
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# daily keogram grids, keogram_grid_YYYY-MM-DD.parquet (released with the dataset)
KEO_DIR = Path(os.environ.get("KEO_DIR", r"D:\pykeo\results\keo_mat"))
# daily TEC observation files, tec_YYYY-MM-DD.parquet (not released, about 0.9 TB)
TEC_DIR = Path(os.environ.get("TEC_DIR", r"D:\pykeo\results\tec_data"))
# 1-min driver table (Hp30, SYM-H, SME, IMAGE, OMNI, F10.7, solar zenith angle)
FEATURES = Path(os.environ.get(
    "FEATURES", r"C:\Users\MarcoGuerra\Documents\PYTHON\lstid_forecasting\data\processed\features.parquet"))
# scripts folder of the pykeo pipeline (only paper/keogram_from_tec.py needs it)
PYKEO = Path(os.environ.get("PYKEO", r"C:\Users\MarcoGuerra\Documents\PYTHON\pykeo\scripts"))
