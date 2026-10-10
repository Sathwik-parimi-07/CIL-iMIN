# coalqual — rapid coal quality estimation from Geotek BoxScan

**CIL-IMin Hackathon · Team AICSAQPS-2026-0005 · IIT (ISM) Dhanbad**

Predicts the five proximate / calorific indices — **moisture, ash, volatile
matter, fixed carbon, gross calorific value** — and the resulting **CIL grade**
for every scan location on a BoxScan core tray, from:

| Sensor on your BoxScan               | Used as                                          |
|--------------------------------------|--------------------------------------------------|
| Spectrophotometer, 350–2500 nm       | 216-band VIS–SWIR reflectance (the core signal)  |
| pXRF, 36 elements                    | mineral-matter chemistry + light-element channel |
| CIE L\*a\*b\* / greyscale            | darkness and hue                                 |
| Magnetic susceptibility              | pyrite / siderite indicator                      |
| Area-imaging camera (optional)       | GLCM / histogram texture                         |

The pipeline is **sensor fusion**, not spectroscopy alone. That is the
claim that distinguishes it from the published VIS–SWIR coal work.

---

## 1. Quick start

```bash
pip install -r requirements.txt

# 1. ingest the real BoxScan sessions (unzipped under data/raw/)
python scripts/01_ingest_boxscan.py --root data/raw

# 2a. WHILE THE LAB IS RUNNING: build a synthetic placeholder population,
#     anchored on the real spectra so the pipeline can be exercised
python scripts/02_make_dataset.py --n 220 --seed 42 --seams 6

# 3. train / benchmark all five indices (writes models, tables, figures)
python scripts/03_train.py --data outputs/tables/dataset_synthetic.csv

# 4. score the real scans and draw the down-core quality log
python scripts/04_predict.py --data outputs/tables/real_scan_features.csv

# 5. sensor ablation (spectra / xrf / fused) for the "why fusion" slide
python scripts/05_ablation.py

# 6. the on-site screening dashboard: one offline HTML file
python scripts/06_build_dashboard.py          # -> outputs/dashboard.html

# 7. regenerate every figure from the saved models (no retraining)
python scripts/07_make_figures.py
```

Add `--fast` to step 3 to skip Boruta and CARS (≈ 3× faster). Steps 3 and 5
accept `--targets MC Ash` so indices can be trained in batches; results merge.
Whole run on one laptop core: about 9 minutes for step 3, 5 for step 5.

## 2. Swapping in the real laboratory results

Step 1 writes `outputs/tables/LAB_LABEL_TEMPLATE.csv`, one row per scan
location, keyed by `sample_id`. Paste the lab's `MC`, `Ash`, `VM`, `FC`, `CV`
(kcal/kg, air-dried basis) into it, then:

```bash
python scripts/02_make_dataset.py \
    --real   outputs/tables/real_scan_features.csv \
    --labels outputs/tables/LAB_LABEL_TEMPLATE.csv
python scripts/03_train.py --data outputs/tables/dataset_real.csv
```

Nothing else changes. The `is_synthetic` flag flips to `False`, the
SIMULATED-DATA stamp disappears from every figure, and `model_*.joblib`
records `trained_on_synthetic: false`.

> With 28 labelled points, `run_target` will refuse below 20 samples and
> the SPXY split leaves ~7 for prediction. Report **leave-one-out CV**
> metrics at that size and say so; do not quote an R²ₚ from 7 points. The
> pipeline switches to LOOCV automatically when n < 30.

## 3. What the code does, step by step

```
POINT_<HOLE>.csv ─┐
ASD_<HOLE>.csv   ─┼─► io_boxscan.load_sessions()  ─► one row / scan location
ENVI cubes       ─┘
                        │
                        ▼
              preprocess:  SG smooth → MSC (fit on calibration only)
                           first derivative · continuum removal · noise screen
                        │
                        ▼
              features:    S<nm> MSC reflectance · D<nm> derivative
                           d_/a_/p_ absorption depth, area, position at
                             1150 1400 1700 1900 2150 2200 2300 2350 nm
                           r_ band ratios · x_ XRF wt% + clay/pyrite/carb/LE
                           col_ CIE L*a*b* · magsus · (t_ texture)
                        │
                        ▼
              validation:  Monte-Carlo CV outlier removal → SPXY split
                        │
                        ▼
              selection:   CC │ Boruta │ CARS │ SPA │ ALL      (per index)
              models:      PLSR │ ENet │ RF │ ExtraTrees │ GBR │ LSSVR
                           hyper-parameters searched by CV *inside* calibration
                        │
                        ▼
              winner chosen on CV RMSE → scored ONCE on held-out set
                        │
                        ▼
              outputs/models/model_<IDX>.joblib · tables · figures
```

Key implementation notes:

* **Leakage control.** MSC reference, scaler, band mask and wavelength
  selection are all fitted on the calibration fold and stored in the model
  file. `pipeline.predict()` replays them exactly.
* **Winner selection never touches the prediction set.** Picking the best of
  ~30 candidate pipelines on the test set inflates R² by 0.03–0.08 at this
  sample size.
* **LSSVR** is implemented from the primal/dual equations (`models.LSSVR`),
  so it is named and behaves as in the literature. It is mathematically the
  same estimator as kernel ridge regression.
* **Proximate closure** `FC = 100 − Ash − MC − VM` is re-imposed on
  predictions (`pred_FC_closure`) as a consistency check.
* **CIL grade** is derived from predicted GCV via `config.CIL_GRADES`.
* **Applicability domain** (`validation.ApplicabilityDomain`) is fitted on the
  calibration features and saved with each model; see §5.
* **Dashboard** (`scripts/06_build_dashboard.py`) embeds the tray photo, marks
  each scan location from the curation geometry, and needs no network.

## 4. About the synthetic data — read before presenting

`simulate.generate_dataset()` is a **stand-in, not a result**. It exists so
the whole pipeline can be built and demonstrated at n = 200 while the lab is
still working.

What is real in it: the spectral endmembers (extracted from your 28 scans by
ranking on the pXRF light-element channel), the band-wise noise level
(Savitzky–Golay residual method), the wavelength grid, the XRF channel list,
the proximate mass balance, the Indian GCV correlation
`GCV = 85.6[100 − (1.1A + M)] − 60M`, and IS 1350 lab repeatability on the
labels.

What is invented: the coal population itself and the coupling strengths
between chemistry and absorption depth. Two nuisances are deliberately built
in so the demo does *not* report R² = 0.999:

* **sub-sampling representativeness error** — the 1 cm optical spot and the
  1 kg lab increment never see the same ash on a chip tray (σ ≈ 2.5 %);
* **maceral brightness** — inertinite-rich coal is brighter at equal ash, so
  the models must use absorption features, not brightness.

Consequently **model rankings and the working pipeline transfer to the real
data; absolute R² values do not.** Every figure produced from synthetic data
is stamped `SIMULATED DATA`. Leave the stamp on. Show the real 28 spectra
(`outputs/figures/spectra_by_col_L.png`) alongside as proof of ingest.

## 5. What the demo run produced (simulated labels, n = 240)

| Index | Bands by | Model | R²ₚ | RMSEₚ | RPD | verdict |
|-------|----------|-------|-----|-------|-----|---------|
| Ash   | all bands | ENet  | 0.946 | 2.76 % | 4.3 | quantitative |
| VM    | all bands | LSSVR | 0.917 | 1.61 % | 3.5 | quantitative |
| MC    | CARS      | LSSVR | 0.893 | 0.44 % | 3.1 | quantitative |
| CV    | CC        | ENet  | 0.876 | 319 kcal/kg | 2.9 | screening |
| FC    | CC        | ENet  | 0.855 | 2.21 % | 2.6 | screening |

Ablation (held-out RMSE, spectra / xrf / fused): MC 0.45 / 1.41 / **0.45**,
Ash 5.26 / 2.87 / **2.70**, VM 2.45 / 2.31 / **1.55**, FC 3.85 / 3.56 /
**2.21**, CV 542 / 442 / **319**. Moisture is a spectra-only problem (XRF is
blind to it), ash is XRF-led, and the three indices that depend on the organic
matrix need both. That pattern is physics, not tuning, and it will survive
the switch to real labels even when the absolute numbers move.

### Applicability domain
`predict()` scores every new scan against the calibration cloud (Hotelling T²
and Q residual on the selected features, limit = 97.5th percentile) and
reports `ad_score` = worst ratio to the limit across the five models:

| `ad_tier` | `ad_score` | meaning                                   | dashboard |
|-----------|-----------:|-------------------------------------------|-----------|
| `in`      | ≤ 1        | inside the calibration cloud              | solid marker |
| `edge`    | ≤ 5        | mild extrapolation, quote with caution    | dashed ring |
| `out`     | > 5        | material the models never saw, send to lab| dark marker |

On the 28 real chips scored with the *synthetic* models: 20 `edge`, 8 `out`,
and the 8 are the Fe-rich and shaly pieces. That is the check doing its job.
Once the models are retrained on real labels the same chips will sit inside.

## 6. Presentation guidance

**Reframe the deck.** Your `locations.csv` shows
`Hyperspectral Camera,False,False`. The instrument has a *point*
spectrophotometer plus pXRF, not an imaging spectrometer. Replace

> "VIS–SWIR spectral imaging … pixel-level quality maps"

with

> "VIS–SWIR reflectance + pXRF sensor fusion on the Geotek BoxScan …
> continuous down-core quality log at 5 cm spacing with grade declaration"

This is a stronger claim: Mao et al. (2024) fuse spectra + texture and reach
R²ₚ 0.95–0.99; you add direct mineral-matter chemistry, which they did not
have. The ablation (`--no-xrf`, `--no-spectra`) produces the slide that
proves fusion beats either sensor alone.

**Numbers to quote.** RPD > 3 with R² > 0.90 = quantitative; 2–3 = screening
grade; < 2 = qualitative. On the simulated set ash and CV land in the first
band, moisture in the second — the same ordering the literature reports.

## 7. Module map

| File                    | Purpose                                                   |
|-------------------------|-----------------------------------------------------------|
| `coalqual/config.py`    | wavelength grid, diagnostic bands, targets, CIL grades    |
| `coalqual/io_boxscan.py`| BoxScan readers (two-row CSV header, ENVI, curation)      |
| `coalqual/preprocess.py`| SG, FD, SNV, MSC, continuum removal, noise screen         |
| `coalqual/features.py`  | spectral / XRF / colour / texture feature blocks          |
| `coalqual/selection.py` | CC, Boruta, CARS, SPA                                     |
| `coalqual/models.py`    | LSSVR + model zoo with search grids                       |
| `coalqual/validation.py`| SPXY, MCCV outliers, R²/RMSE/RPD/RPIQ                     |
| `coalqual/simulate.py`  | anchored synthetic generator (see §4)                     |
| `coalqual/pipeline.py`  | train / benchmark / persist / predict                     |
| `coalqual/visualize.py` | deck figures, all stamped when synthetic                  |
| `scripts/01–07_*.py`    | ingest · dataset · train · predict · ablation · dashboard · figures |

## 8. Requirements

Python ≥ 3.10, numpy, pandas, scipy, scikit-learn ≥ 1.3, matplotlib,
scikit-image (texture only), joblib, Pillow. No internet needed at runtime.

## 9. References

1. Cloutis E.A. (2003) *Fuel* 82, 2239–2254.
2. Fysh S.A. et al. (1985) *Appl. Spectrosc.* 39, 354–357.
3. Begum N. et al. (2020) *Fuel* 280, 118676.
4. Mao J. et al. (2024) *Appl. Sci.* 14, 7920 — spectra-texture fusion, LSSVR.
5. Li H. et al. (2009) *Anal. Chim. Acta* 648, 77–84 — CARS.
6. Kursa M.B. & Rudnicki W.R. (2010) *J. Stat. Softw.* 36(11) — Boruta.
7. Galvão R.K.H. et al. (2005) *Talanta* 67, 736–740 — SPXY.
8. Suykens J.A.K. & Vandewalle J. (1999) *Neural Process. Lett.* 9, 293–300 — LSSVR.
9. Wyman C., Sloan P.-P., Shirley P. (2013) *JCGT* 2(2) — analytic CIE CMFs.
10. IS 1350 Parts 1 & 2, Bureau of Indian Standards.
