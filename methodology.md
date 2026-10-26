# Methodology: GLOF lake inventory, prioritisation, and visual review

## 1. Objective and interpretation

This workflow consolidates multi-source lake inventories in High Mountain Asia, relates them conservatively to documented GLOF records, derives lake-evolution features, and prioritises lakes for further review. Its outputs are research evidence and descriptive prioritisation signals.

Neither an unlabeled lake nor a low indicator score means that a GLOF cannot occur. The Streamlit score is not calibrated as an event probability and must not be used as a standalone hazard warning, land-use decision, or life-safety decision.

## 2. Input data and harmonisation

The analysis uses Sentinel-1 and Sentinel-2 GLO-HMA lake products (2017–2024), annual 30 m Lake_TP shapefiles (1991–2023), and the supplied HMAGLOFDB historical event inventory. Source geometries are reprojected into `ESRI:102025` before calculating areas, perimeters, distances, intersections, or centroids. Source attributes and geometry WKT are retained in the observation layer.

The builder deliberately does not create missing observation years, interpolate lake outlines, manufacture event dates, or label every unmatched lake as GLOF=0.

## 3. Master-lake construction

### 3.1 Candidate matching

For each source pair, the pipeline searches for candidate outlines within 750 m. It calculates centroid separation, intersection-over-union (IoU), and smaller-to-larger area ratio. Each candidate and its decision rationale is written to `lake_matching.csv`, including ambiguous and rejected cases.

### 3.2 Conservative identity decision

| Criterion | Threshold |
|---|---:|
| search distance | 750 m |
| maximum centroid separation | 500 m |
| minimum IoU | 0.20 |
| minimum area ratio | 0.20 |
| minimum margin over runner-up score | 0.12 |

Accepted links are joined into master-lake components, each with a stable `ML...` identifier. The thresholds favour avoiding false links, while retaining uncertain relationships for audit.

### 3.3 Event linkage

The event inventory is not transformed into a negative/positive classification table. A documented event coordinate is associated with the nearest current inventory polygon only when it lies within 250 m of that polygon. This build contains 768 documented events, 65 of which link to current inventory lakes. The others remain as unmatched historical evidence, not rejected events.

## 4. Feature engineering

Each observation produces physical and temporal predictors in `lake_features.csv`:

- **Morphology:** polygon area and perimeter; axis-aligned bounding-box length and width; elongation `length/width`; compactness `4πA/P²`.
- **Change:** one- and three-year area differences where an earlier observation exists, plus elapsed-time growth rate.
- **Trend:** a long-term linear area trend. The UI-indicator workflow additionally recomputes a causal trend so a row uses observations only at or before its own year.

Series calculations are scoped to master ID, source dataset, and sensor. This prevents a sensor change from being treated as lake growth. Missing values remain missing; the indicator handles unavailable inputs neutrally rather than inventing measurements.

## 5. Research-model tracks

`train_glof_prediction_models.py` generates three separate evidence streams. Read each output folder’s `README.md` before interpreting scores.

### 5.1 Similarity ranking

At lake level, candidate models compare linked-event lakes with unlabeled lakes using aggregate geometry, area, and temporal features. The present decision log reports 53 unique linked event lakes and 48,971 unlabeled comparison lakes, and selects a histogram-gradient-boosting ranking model using 67 lake-level features.

This is a positive-evidence similarity ranking, not a verified binary classifier: unlabeled lakes are not confirmed controls. Proxy ROC/AUC and AP describe separation under that comparison definition, not event probabilities.

### 5.2 Pre-outburst change detection

For an event with a numeric year, the latest real lake observation strictly before that year is selected. Isolation Forest, Local Outlier Factor, and one-class SVM approaches score unusual lake-year behaviour against unlabeled background observations. The present run has three documented pre-event cases and selected one-class SVM, so this is an exploratory signal only.

### 5.3 Time-to-event analysis

The survival track uses an event only when its source year lies within the lake’s observed span; other lake histories are right-censored at their final observation. Cox proportional-hazards and Weibull accelerated-failure-time candidates use first-observation geometry/area predictors, avoiding post-event temporal leakage. The present run has three in-window events, which severely limits generalisation.

### 5.4 Combined review score

`summarize_glof_model_outputs.py` produces a manual-review score:

`combined_review_score = 0.60 × similarity percentile + 0.25 × anomaly percentile + 0.15 × survival-risk percentile`

Similarity receives the highest weight because it has the broadest matched positive evidence. The shortlist is for analyst triage, not calibrated probability or risk classification.

## 6. Fixed-weight susceptibility indicator and 3D viewer

The Streamlit app uses a separate transparent indicator built by `train_glof_susceptibility.py`. It derives fifth and ninety-fifth percentile reference limits from all 216,505 feature rows and applies predetermined weights to ten morphology and history inputs. Temporal variables receive 0.60 of total weight, with causal long-term trend at 0.18. Each feature is bounded to 0–1, compactness is inversely scored, and the weighted sum is expressed from 0 to 100.

For an uploaded raster mask, the viewer aligns the mask and DEM in local UTM, retains the largest connected lake component, computes lake and surrounding-terrain metrics, optionally finds prior inventory history by nearest master-lake centroid, and renders a Plotly 3D terrain view. A missing history feature receives neutral indicator value 0.5.

The held-out proxy evaluation must be read cautiously: it contains six documented-event rows, finds zero true positives at the High threshold, and compares documented records with unlabeled—not verified safe—records. It does not establish real-world forecast accuracy.

## 7. Quality assurance and reproducibility

Every build writes `validation_report.json`, containing record counts, source/year coverage, matching thresholds, and integrity assertions such as `missing_observation_years_are_not_created` and `glof_negative_labels_created: false`. QA should include:

1. Inspect record counts and source temporal coverage.
2. Inspect accepted, ambiguous, and rejected spatial matches.
3. Examine event-link status alongside original event metadata.
4. Check feature distributions and missingness by source/sensor.
5. Preserve source datasets, code, model bundles, and generated validation report as one reproducible run.

## 8. Limitations

- Event records are incomplete, sometimes spatially uncertain, and represent multiple failure mechanisms.
- Lake outlines and temporal density differ across sensors and inventories.
- Dam condition, ice/rock avalanches, glacier dynamics, weather, drainage, and downstream exposure are not comprehensively modelled.
- Sparse event-linked cases constrain validation, particularly pre-event and survival analyses.
- DEM terrain values are descriptive; the application is not a breach, routing, or inundation model.

Use rankings as a starting point for image review, field reconnaissance, and multidisciplinary hazard assessment.
