# LSTID identification in TEC keograms

Detection and characterisation of large-scale travelling ionospheric disturbances
(LSTIDs) in GNSS TEC keograms over Europe (37.5–57.5°N; pierce points ≈0–25°E, no
longitude window), 2015–2025.

**The vespagram catalogue is the default** and is the one to use for climatology,
driver studies and forecasting. The older ROI catalogue is kept for comparison
only.

| | Vespagram (default) | ROI (legacy) |
|---|---|---|
| Pipeline | `vespagram.py` | `detect_roi.py` |
| Catalogue | `results/vespagram/trains.parquet` (+ `detections.parquet`) | `results/roi/lstid_rois.pkl` |
| Unit | one record per wave **train** | one record per **front** |
| Size | 6,788 equatorward trains | 10,296 fronts (2,541 trains) |
| Hours with an LSTID present | 15.3 % (high+medium), 21.7 % (all tiers) | 9.3 % |

## Method (vespagram)

Each latitude row of the keogram is treated as one sensor of a north–south array,
as in seismology. The keogram is decimated to 0.25° × 2 min and band-passed to
25–200 min. For each trial slowness *p* = 1/speed, the rows are delay-and-summed,
giving:

- **semblance** *S*(t, p), a 0–1 coherence measure that does not depend on amplitude;
- **beam RMS**, the amplitude of the stacked wave in TECu.

1. **Arrays.** The full aperture (37.5–57.5°N) plus five overlapping 10° bands
   in 2.5° steps.
2. **Detections.** Ridges of *S* along *p* with *S* ≥ 0.6, beam RMS ≥ 0.01 TECu,
   duration ≥ 45 min and 250 ≤ |v| ≤ 2000 m/s. The ridge slowness is refined
   below the grid step (parabola through the semblance peak), so speeds are not
   tied to the slowness grid.
3. **Trains.** Detections from different arrays are linked when their waveforms
   correlate (r ≥ 0.8) where the arrays overlap.
   - Two detections of the same array that overlap in time are distinct waves
     and never end up in one train. The full array also keeps apart consecutive
     detections whose speeds differ by more than it can resolve; a 10° band
     cannot, so a front whose speed changes over time stays one train.
   - LSTID trains of the same direction that still overlap in time are merged
     into one train (`merge_simultaneous`): over this array there are no two
     simultaneous LSTID trains.
   - Detections propagating in opposite directions are never linked.
   - A train then extends into neighbouring bands that line up with it but had no
     ridge of their own.
4. **Kept.** Only LSTIDs (300–1500 m/s, period ≥ 30 min) propagating
   **equatorward**.
   - Trains seen by a single band are excluded: two blind audits found most of
     them were not LSTIDs.
   - Poleward trains are excluded: they showed no auroral-driver dependence and
     peak at dawn in summer. Set `directions` in `VespaConfig` to bring them back.
5. **Midnight.** Each day is processed with 12 h of padding from its neighbours,
   so the day that owns a train (midpoint inside it) sees the whole train.
   Trains crossing midnight are settled across days (`resolve_day_boundaries`),
   so each wave is catalogued exactly once. If the two days link a wave
   differently and both own a version, the longer one is kept.

### Confidence (`confidence_score` and `confidence` columns)

Each train has a score between 0 and 1, coherence times persistence:

    score = (semblance - 0.6) / 0.4 * (1 - exp(-n_cycles / 2))

where `n_cycles` is the train's duration divided by its period. The score does
not use the amplitude (which follows background TEC and the drivers) or the
latitudinal extent: in strong storms the source region moves inside the array,
so a clear train cannot span the full 20°.

| Tier | Rule | Trains |
|---|---|---|
| high | score ≥ 0.40 | 2,524 |
| medium | 0.30 ≤ score < 0.40 | 1,427 |
| low | score < 0.30 | 2,837 |

Trains slower than 350 m/s drop one tier: just above the 300 m/s limit they mix
with slower medium-scale TIDs. Use **high+medium** for science and forecasting,
and keep low as a flag rather than deleting it.

The thresholds are where agreement with the published ROI catalogue (2021–2024)
changes: below 0.30 trains match it no more often than by chance, above 0.40
nearly three times as often. Independent checks of the tiers are the HF-INT
catalogue and the dependence on auroral activity. The score is an ordering, not
a calibrated probability. The earlier
rule (high = detected by the full 20° array) tracked both reference catalogues
and Hp30 less well, and latitudinal extent was the weakest predictor of all.

### Catalogue columns (`trains.parquet`)

**Timing**

| Column | Meaning |
|---|---|
| `date` | processing day the train is filed under |
| `train_id` | train number within that day |
| `t_start`, `t_end` | train start and end (UTC) |
| `owned` | `False` for a midnight-crossing train kept from the neighbouring day's view (its midpoint is on the adjacent day) |

**Wave properties**

| Column | Meaning |
|---|---|
| `speed_m_s`, `slowness` | propagation speed; slowness *p* > 0 means equatorward. Semblance-weighted median of the ridge slowness, from the full array if it saw the train, otherwise from the bands |
| `speed_lo_m_s`, `speed_hi_m_s` | 16th–84th percentile range of the ridge speed over the train. In the injection test it contains the true speed 74 % of the time |
| `period_min`, `wavelength_km` | wave period and wavelength |
| `amplitude` | equivalent peak amplitude, √2 × RMS of the stacked beam over each detection, maximum over the train's detections [TECu] |
| `semblance` | maximum semblance over the train's detections |
| `n_cycles` | duration divided by period |
| `confidence_score` | confidence score, 0–1, see above |
| `confidence` | tier, see above |
| `direction` | always `equatorward` in the current catalogue |

**Extent and geometry**

| Column | Meaning |
|---|---|
| `lat_lo`, `lat_hi` | latitude extent, including extended bands |
| `full_array` | the full array detected the train |
| `n_bands` | bands the train covers, including extended ones |
| `n_bands_picked` | bands with their own ridge |
| `geometry` | `planar`, or `curved` when band speeds differ by more than 1.5× |
| `speed_ratio` | fastest / slowest band speed |
| `profile_lat`, `profile_speed` | speed against latitude, one value per band |

`detections.parquet` holds the per-array detections behind each train, joined on
(`date`, `train_id`). Its columns include the crest and trough times.

## How to run

The Python environment is `ionoring-test/iono-ring-env`. Keograms are read from
`D:\pykeo\results\keo_mat`, and the drivers from
`lstid_forecasting/data/processed/features.parquet`.

```bash
python test_vespagram.py                          # synthetic checks (must be 0 unexpected failures)
python injection_test.py --summary                # injection-recovery results (drop --summary to rerun, ~15 min)
python vespagram.py 2024-05-11                    # one day: figure + CSV in results/vespagram/
python vespagram.py all --no-plot --workers 12    # full catalogue (~30 min); only 'all' writes trains.parquet
python analysis/catalogue_records.py              # records pickles for the analysis scripts (run after every catalogue run)
python paper/method_checks.py                     # passband, slowness resolution, observation-density test -> paper/numbers.json
python paper/method_checks.py --tec-file paper/external/tec_2024-08-12.parquet   # also the empty keogram cells of that day
```

**Analysis** (`analysis/`). The scripts default to the vespagram high+medium
records and write to `results/vespagram/climatology/`. Pass
`--pkl results/roi/lstid_trains.pkl` (or `lstid_hours.pkl`) to run them on
the ROI catalogue.

- **Per train:**
  - `climatology.py`: hour, month and year distributions;
  - `characteristics_temporal.py`: wave characteristics by year and UT hour;
  - `driver_characteristics.py`: wave characteristics against driver activity.
- **Per LSTID-active hour:**
  - `driver_activity.py`: occurrence against driver levels;
  - `driver_regression.py`: confounder-adjusted logistic model, one driver at a time;
  - `driver_lag.py`: response delay.
- **ROI vs vespagram comparison**, written to `results/climatology_compare/`:
  - `compare_climatology.py`: shares of hours, driver effects and tiers;
  - `ratio_climatology.py`: vespagram/ROI ratio against activity, solar cycle,
    UT and month.

**Inspection sheets**, plots only:

- `inspect_confidence.py`: random trains per confidence tier.
- `storm_overview.py`: the two strongest storms per year.
  - `--random 40` instead draws random days.
- `quiet_check.py`: quiet-time trains, comparing those found only by the
  vespagram with those also found by ROI.
- `compare.py`: side-by-side ROI vs vespagram examples. It also defines the
  front-to-train merge rule used for the ROI catalogue.

## Injection test

`injection_test.py` adds wave packets of known speed, period and amplitude to 132
real quiet-day keograms (2,592 injections) and runs the detector. Results are in
`results/vespagram/injection/` (`summary.txt`). In short:

- **Detection.** About half of the waves are found at 0.02 TECu, 80–100 % at
  0.05 TECu and nearly all at 0.1 TECu. Close to the 1500 m/s limit it is harder
  (67 % at 0.05 TECu for 1400 m/s).
- **Speed.** Unbiased, within about ±4 % for high-confidence trains and for
  trains detected by the full array. Band-only trains scatter much more (one in
  three off by more than 20 %) and weak waves above 1100 m/s read 8–14 % slow.
- **Speed interval.** `speed_lo_m_s`–`speed_hi_m_s` contains the true speed 74 %
  of the time (68 % expected).
- **Amplitude** reads about 20 % below the injected peak for strong waves (it is
  an average over the packet). **Period** is within 3 %.
- **Geometry.** 14 % of planar waves are labelled `curved`.
- **Confined waves** (37.5–50°N): found, speed unbiased, 71 % high confidence;
  the catalogued northern edge overshoots by about one 2.5° step.

Limits: single planar packets at constant speed, quiet-day backgrounds, and the
packet is added after the arc-level detrending, so that filter is not tested.

## Known limitations

- **Period passband.** Detrending (90-min Savitzky–Golay), the 10-min mean and the
  band-pass together pass a narrow band: peak at 42 min, half power at 32–57 min,
  amplitude down to 50 % at 70 min and 9 % at 2 h. Longer-period LSTIDs are
  strongly under-represented and the period distribution reflects this band.
- **Solar-cycle sensitivity.** At the same Hp30, occurrence follows solar flux
  (0.5 high+medium hours per day in 2019 against 3.0 in 2024 for Hp30 1–2): wave
  amplitude in TECu scales with background TEC and the detector has a fixed
  0.01 TECu floor. Observation density is not the cause: days that lose a fifth
  of their observations show no change (within ±11 %).
- **Speed resolution.** The speed of a single train is precise to a few percent
  (see the injection test), but two waves present at the same time are separated
  only if their speeds differ a lot: for a 55-min wave at 600 m/s the semblance
  peak is above half maximum from 430 to 1010 m/s on the full array and from 330
  to 3000 m/s on a 10° band.
- **Speed limits.**
  - Waves slower than 300 m/s are not LSTIDs by definition. These include the
    large daytime waves on quiet winter days, at about 200–270 m/s.
  - The steepest storm-time fronts, faster than 1500 m/s or nearly simultaneous
    across latitudes, are rejected. Whether that cuts off real storm LSTIDs is
    still open.
- **Simultaneous waves.** By definition there is one train at a time: a slow
  wave in the south and a fast one in the north at the same time are one
  "curved" train, and two simultaneous waves are one train whose speed is that of
  the dominant one, with a wide speed interval.
- **Quiet-time agreement with ROI.** The vespagram finds about 1.1× ROI's LSTID
  hours in storms but about 3× in quiet conditions (high+medium). Quiet-time trains
  found only by the vespagram look like real but weaker LSTIDs: median amplitude
  0.024 vs 0.037 TECu. Quiet-time occurrence is therefore method-dependent.
- **Data coverage changes over the years.** 2015 receivers lacked BeiDou and
  Galileo. No effect of observation density on the counts was found (see
  solar-cycle sensitivity).
- **Not real-time as is.** The band-pass filter runs forward and backward in
  time, and each day uses 12 h of the following day. An operational forecast needs
  a version that only uses past data. The offline catalogue is fine as training
  labels.
