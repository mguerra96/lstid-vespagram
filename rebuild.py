"""
rebuild.py

Rebuild the catalogue and everything derived from it, in the order the steps
depend on each other. Each step reads what the earlier ones wrote, so after a new
catalogue run them all: a step run alone can mix numbers of the old and of the
new catalogue without any warning.

  python rebuild.py            every step
  python rebuild.py 3          from step 3 on (after fixing what stopped it)
  python rebuild.py --list     only list the steps

Needs the keograms (KEO_DIR) and the driver table (FEATURES), see paths.py. The
plots of the other analysis/ scripts are not part of it: no number depends on them.
"""

import sys
import subprocess

from paths import ROOT

STEPS = [
    "test_vespagram.py",                          # the detector passes its synthetic checks
    "vespagram.py all --no-plot --workers 12",    # the catalogue: trains and detections (.parquet, .csv)
    "analysis/catalogue_records.py",              # record pickles of both catalogues
    "injection_test.py",                          # injection test (its quiet days come from the catalogue)
    "analysis/compare_climatology.py",            # results/climatology_compare/drivers.csv
    "analysis/ratio_climatology.py",              # results/climatology_compare/ratio.csv
    "paper/paper_numbers.py",                     # paper/numbers.json (reads the two csv above)
    "paper/validate_published.py",                #   + its published_roi section
    "paper/validate_hfint.py",                    #   + its hfint section
    "paper/method_checks.py",                     #   + its method_checks section
    "paper/make_figures.py",                      # figures (fig. 3 reads numbers.json)
]


def main():
    if "--list" in sys.argv:
        print("\n".join(f"{k:2d}  python {s}" for k, s in enumerate(STEPS, 1)))
        return
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for k, step in enumerate(STEPS, 1):
        if k < first:
            continue
        print(f"\n[{k}/{len(STEPS)}] python {step}", flush=True)
        if subprocess.run([sys.executable, *step.split()], cwd=ROOT).returncode:
            sys.exit(f"step {k} failed. Fix it, then: python rebuild.py {k}")


if __name__ == "__main__":
    main()
