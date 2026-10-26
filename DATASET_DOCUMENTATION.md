# Dataset documentation

## Purpose and scope

This repository assembles a provenance-preserving High Mountain Asia (HMA) glacial-lake corpus for GLOF research, exploratory ranking, and visual review. It combines lake outlines observed over time with a supplied historical GLOF inventory. It is not a record of confirmed safe lakes: a lake without an event match is **unlabeled**, not a verified non-event.

All geometries used for matching and metric calculation are represented in `ESRI:102025` (metres). Output coordinates intended for mapping are also available in WGS84 longitude/latitude.

## Source datasets

| Source | Location | Coverage in this setup | Role |
|---|---|---:|---|
| GLO-HMA Sentinel-1 | `GLO/S1_20172024_TB_GLOID_v1_0.gpkg` | 2017–2024 | All-weather SAR lake observations |
| GLO-HMA Sentinel-2 | `GLO/S2_20172024_TB_GLOID_v1_0.gpkg` | 2017–2024 | Optical lake observations |
| 30 m Lake_TP | `Lake_TP/<year>/Lake_TP_<year>.shp` | 1991–2023 | Historical Landsat-derived lake outlines |
| HMAGLOFDB | `GLOF INVENTORY/.../Database/GLOFs/HMAGLOFDB.csv` | historical inventory supplied with project | Documented GLOF locations and metadata |
| Copernicus GLO-30 | local DEMs and optional download used by the viewer | location-dependent | Terrain visualisation and terrain metrics; not a training label source |

The GLO package also includes pre-trained deep-learning packages (`.dlpk`/`.emd`) and the repository contains segmentation checkpoints under `Segmentation Models/`. These are supplied model artifacts. They are not consumed by the master-dataset builder or the current susceptibility-indicator training script.

## Master dataset products

The reproducible builder is `build_master_lake_dataset.py`; its outputs are in `master_lake_dataset/`.

| File | Unit of analysis | Contents |
|---|---|---|
| `master_lakes.csv` | persistent candidate lake | `ML...` master ID, source IDs, representative polygon, centroid, component count, evidence status |
| `lake_observations.csv` | source observation | original source ID, date/year, original area attribute, geometry WKT, source attributes JSON |
| `lake_features.csv` | observation with derived features | shape, location, area-change, growth and trend features |
| `lake_matching.csv` | evaluated candidate pair | matching metrics, decision, and evidence rationale; use for audit |
| `glof_events.csv` | documented inventory event | original event metadata plus any master-lake linkage and linkage status |
| `validation_report.json` | build run | record counts, coverage, matching thresholds, and integrity checks |

### Build snapshot currently in the repository

| Measure | Value |
|---|---:|
| Master lakes | 49,025 |
| Source observations | 216,505 |
| GLO-HMA S1 unique lakes | 2,966 |
| GLO-HMA S2 unique lakes | 4,152 |
| 30 m TP source features | 175,699 |
| Documented GLOF events | 768 |
| Events linked to a current inventory lake | 65 |
| Master lakes with accepted cross-source evidence | 22,862 |
| Ambiguous candidate pairs retained | 2,618 |

Counts are a build snapshot, not universal properties of the upstream datasets. Regenerate the master data and consult `validation_report.json` if inputs or thresholds change.

## Identity, matching, and event linkage

`master_lake_id` is the project-level identity. GLO source IDs are retained in `glo_s1_id` and `glo_s2_id`. Lake_TP does not supply a durable lake identifier; its `tp_id` is a deterministic year-plus-source-feature reference and must not be treated as an independent persistent ID.

Candidate outline pairs are searched within 750 m. An accepted link must satisfy a centroid-distance limit of 500 m, IoU of at least 0.20, smaller-to-larger area ratio of at least 0.20, and a 0.12 score separation from the next-best candidate. Ambiguous and rejected candidates are retained rather than silently converted into matches.

GLOF events are linked when their supplied lake coordinate is within 250 m of a current inventory polygon. Events outside that tolerance remain in `glof_events.csv` with `UNMATCHED` status. Therefore, the 65 links are usable positive evidence for the current inventory; they are not a claim that the other 703 historical events are invalid.

## Feature dictionary

`lake_features.csv` is the analysis-ready feature table. Features are calculated separately within `master_lake_id + source_dataset + sensor`; do not combine sensor time series without making that choice explicit.

| Field | Definition | Unit / notes |
|---|---|---|
| `area_derived_km2` | polygon area in the working CRS | km² |
| `perimeter_derived_km` | polygon boundary length | km |
| `length_m`, `width_m` | axis-aligned bounding-box sides in `ESRI:102025` | m; a reproducible shape proxy, not a flow-path measurement |
| `elongation_ratio` | `length_m / width_m` | dimensionless |
| `compactness` | `4π × area / perimeter²` | dimensionless; values nearer one are more compact |
| `area_change_1y_km2` | current area minus previous-year area, where available | km² |
| `area_change_3y_km2` | current area minus area three years earlier, where available | km² |
| `growth_rate_km2_per_year` | change from preceding observation divided by elapsed years | km²/year |
| `long_term_area_trend_km2_per_year` | linear trend over the available series in the feature build | km²/year |

The indicator workflow creates `causal_long_term_area_trend_km2_per_year` separately. For each observation, it fits a trend using observations at or before that year only, avoiding future-area leakage into a historic row.

## Data-use rules and limitations

- Preserve `original_attributes_json`, original IDs, dates, and source names when exporting or subsetting.
- Missing years are not fabricated. A blank change feature means the required earlier observation was unavailable.
- Geometries, source resolution, acquisition conditions, and temporal density differ by source. Sensor/source effects can be substantial.
- Historical event coordinates and dates can be approximate. An unmatched event is not evidence of no GLOF.
- This corpus supports prioritisation and research; site-specific hazard assessment also needs dam condition, ice/avalanche triggers, hydrology, exposure, and expert review.

## Rebuild and audit

Run `python build_master_lake_dataset.py` from the repository root after changing inputs or match parameters. Then review `validation_report.json`, `lake_matching.csv`, and the QA figures before using regenerated outputs. The builder records its thresholds in the validation report, enabling each build to be compared and audited.
