"""
paper/external/download.py

Download the third-party inputs that are not redistributed with this repository.

  python paper/external/download.py

- Monthly sunspot number, SILSO, Royal Observatory of Belgium (CC BY-NC 4.0)
    -> paper/figures/data/SN_m_tot_V2.0.csv                         (Fig. 6a)
- Published ROI catalogue, Guerra et al. (2025), doi:10.5281/zenodo.14988318 (CC BY 4.0)
    -> paper/external/LSTID_ROI.mat; then, in MATLAB, from paper/external/:
         S = load('LSTID_ROI.mat'); f = fieldnames(S);
         writetable(S.(f{1}), 'LSTID_ROI_published.csv')
       (the file holds a MATLAB table, which Python cannot read)      (validate_published.py)
- HF-INT catalogue, Segarra et al., doi:10.34810/data1383 (CC BY 4.0): the CORA
  repository blocks scripted downloads. Download the dataset by hand and unzip the
  Catalogue_LSTID_YYYY.txt files into paper/external/hfint/        (validate_hfint.py)
"""

import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    "https://www.sidc.be/SILSO/INFO/snmtotcsv.php": HERE.parent / "figures/data/SN_m_tot_V2.0.csv",
    "https://zenodo.org/api/records/14988318/files/LSTID_ROI.mat/content": HERE / "LSTID_ROI.mat",
}

for url, out in FILES.items():
    if out.exists():
        print("exists", out)
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    print("wrote", out)

missing = [p for p in [HERE / "LSTID_ROI_published.csv", HERE / "hfint/Catalogue_LSTID_2015.txt"] if not p.exists()]
for p in missing:
    print("still missing (see the docstring):", p)
