# LSTID catalogue from GNSS TEC keograms over Europe

Code of the paper *An eleven-year catalogue of large-scale travelling ionospheric
disturbances from GNSS TEC over Europe* (Guerra, Cesaroni & Spogli, Scientific Data,
submitted). It detects equatorward large-scale travelling ionospheric disturbances
(LSTIDs) in daily GNSS TEC keograms of 37.5–57.5°N, 2015–2025, and produces every number
and figure of the paper.

The catalogue itself (trains, detections, keograms and daily coverage) is on Zenodo:
*DOI to be added*. The keograms are built from RINEX files by
[pykeo](https://github.com/mguerra96/pykeo) (doi:[10.5281/zenodo.23495716](https://doi.org/10.5281/zenodo.23495716)).

## Method in short

The method is described in full in the paper (Methods). Each latitude row of the keogram
is treated as one sensor of a north–south array, as in array seismology.

1. **Vespagram.** The keogram is decimated to 0.25° × 2 min and each row is band-passed
   to 25–200 min. For three 10° sub-arrays (37.5–47.5, 42.5–52.5 and 47.5–57.5°N) and
   45 equatorward trial slownesses, the rows are delay-and-summed in a 30-min window,
   giving the **semblance** *S* (0–1, independent of amplitude) and the **beam RMS**
   (TECu).
2. **Ridge pixels.** Local maxima of *S* along slowness with *S* ≥ 0.6, beam RMS ≥
   0.01 TECu and speed 300–1500 m/s; slowness refined by a parabola through the peak.
3. **Detections.** Ridge pixels joined across consecutive time steps (at most three
   slowness steps apart); a ridge lasting ≥ 45 min is a detection.
4. **Trains.** Detection times are moved to 47.5°N. A detection starting less than
   60 min after the end of the current train joins it; a train needs at least one crest
   and one trough. Trains never overlap. Each day is processed with 12 h of padding,
   and a train crossing midnight belongs to the day of its midpoint.
5. **Speed** = 1 / semblance-weighted mean slowness of all the train's ridge pixels;
   the speed interval is the weighted 16th–84th percentile range.
6. **Confidence score** *C* = (*S* − 0.6) / 0.4 × coverage, with *S* the mean semblance
   of the ridge pixels and coverage the mean share of the three sub-arrays with a
   detection running. Levels: high *C* ≥ 0.35, medium 0.20 ≤ *C* < 0.35, low below. Use
   **high + medium** for science and forecasting; low is a flag, not a rejection.

Poleward propagation is not searched. The parameters are in `VespaConfig`
(`vespagram.py`) and in Table 1 of the paper.

## Outputs

`python vespagram.py all` writes to `results/vespagram/`:

| File | Content |
|---|---|
| `release/trains.{parquet,csv}` | the published train catalogue (columns: Table 2 of the paper) |
| `release/detections.{parquet,csv}` | the published detection catalogue (Table 3) |
| `trains.parquet`, `detections.parquet` | the same with the extra columns the analyses use (semblance, coverage, number of detections, slowness, ...) |

Trains and detections are joined on (`date`, `train_id`). All times are UTC. Train
times are at 47.5°N; detection times, crests and troughs are at the centre of their
sub-array.

## Installation and inputs

Python 3.11 with the pinned packages:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip
```

Download the Zenodo record and put its inputs in `data/`:

```
data/keograms/keogram_grid_YYYY-MM-DD.parquet    unzip keograms_2015.zip ... keograms_2025.zip here
data/drivers_1min.parquet                         1-min drivers: Hp30, SYM-H, IMAGE, OMNI, F10.7, SZA
```

Other locations can be set with environment variables (`paths.py`):

| Variable | Default | Needed by |
|---|---|---|
| `KEO_DIR` | `data/keograms` | everything |
| `FEATURES` | `data/drivers_1min.parquet` | driver analysis, injection test (quiet days), Figs. 3, 4, 6 |
| `TEC_DIR` | `data/tec` | `daily_coverage.py`, empty-cell checks (pykeo TEC files, not released, ~0.9 TB) |

The SuperMAG indices (SME, SML, SMU) may not be redistributed. The first script that
needs them asks for your SuperMAG username (free registration at
https://supermag.jhuapl.edu, or set `SUPERMAG_USERID`), downloads them and adds them to
the driver table (`supermag.py`; run `python supermag.py` to do it beforehand).

Third-party reference data are not redistributed here. `python
paper/external/download.py` fetches the SILSO sunspot number and the published ROI
catalogue, and its docstring explains the two steps that cannot be scripted (the HF-INT
catalogue, behind bot protection on CORA, and the MATLAB export of the published ROI
catalogue).

## How to run

```bash
python test_vespagram.py                          # synthetic checks and the midnight rule (exit code = failures)
python vespagram.py 2024-05-11                    # one day: figure + CSV in results/vespagram/
python vespagram.py all --no-plot --workers 12    # full catalogue (~6 min with 12 workers)
python rebuild.py                                 # catalogue and everything derived from it, in order
python rebuild.py --list                          # the steps of rebuild.py
python injection_test.py --summary                # summary of the saved injection test (drop --summary to rerun, ~15 min)
python daily_coverage.py                          # daily coverage table from the TEC files
```

Only a run over `all` days with no failed day writes `trains.parquet` and the release
files; any other run writes `trains_<selector>.parquet` or `trains_partial.parquet`.

`rebuild.py` runs, in order: the tests, the catalogue, `analysis/catalogue_records.py`,
the injection test, `analysis/compare_climatology.py`, `analysis/ratio_climatology.py`,
and the paper scripts `paper_numbers.py`, `validate_published.py`, `validate_hfint.py`,
`confidence_tiers.py`, `method_checks.py` (all writing `paper/numbers.json`) and
`make_figures.py`.

## Repository layout

| Path | Content |
|---|---|
| `vespagram.py` | the detector |
| `test_vespagram.py` | synthetic tests of the detector |
| `injection_test.py` | injection-recovery test; results in `results/vespagram/injection/` |
| `detect_roi.py` | the earlier image-based detector (ROI detector), adapted to the present keograms, used for comparison only |
| `analysis/` | record files of both catalogues, climatology, driver regressions, ROI vs vespagram comparison |
| `paper/` | scripts behind every number (`numbers.json`) and figure of the paper |
| `daily_coverage.py` | daily coverage table released with the data |
| `supermag.py` | downloads the SuperMAG indices into the driver table |
| `tools/` | inspection sheets and the ROI vs vespagram comparison (plots; `quiet_check.py` also feeds `paper_numbers.py`) |
| `paths.py` | input folders |

## Injection test

`injection_test.py` adds 2,592 wave packets of known speed (350–1400 m/s), period
(40–70 min) and amplitude (0.01–0.1 TECu) to 132 quiet-day keograms and runs the
detector unchanged. About two thirds of the 0.02 TECu packets and nearly all those of
0.05 TECu or more are found. Speeds are within a few per cent up to 800 m/s and about
15% too low at 1400 m/s. The full tables are in `results/vespagram/injection/summary.txt`.

## Known limitations

- **Passband.** The detrending of the arcs and the band-pass together pass mainly
  periods of 30–70 min (half power 32–57 min); LSTIDs of 2 h or more are
  under-represented.
- **Speed.** The speed is the apparent meridional phase speed; obliquely propagating
  fronts appear faster. Speeds above about 1000 m/s are underestimated (about 15% at
  1400 m/s). Waves slower than 300 m/s or faster than 1500 m/s are not catalogued.
- **Solar-cycle sensitivity.** Amplitudes in TECu scale with the background TEC, and
  detection has a fixed 0.01 TECu floor, so at the same geomagnetic activity more trains
  are found at high solar activity. The growth in the number of observations
  (Galileo, BeiDou) has no measurable effect.
- **One train at a time.** Simultaneous waves of different speed in one sub-array come
  out as one train at an intermediate speed with a wide speed interval.
- **Offline.** The band-pass is zero-phase and each day uses 12 h of the next day; an
  operational detector would need a causal version.

## Licence and citation

Code: MIT licence, © 2026 Istituto Nazionale di Geofisica e Vulcanologia (INGV).
Data: CC BY 4.0 (Zenodo record). Please cite the paper and the Zenodo record.
