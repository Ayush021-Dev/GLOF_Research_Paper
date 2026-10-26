# GLOF research paper and susceptibility-indicator demo

This repository contains the reproducible code, methods, lightweight model
metadata, and selected publication-ready outputs for the GLOF study. Raw
geospatial inventories, DEMs, generated master datasets, segmentation weights,
and large model tables are intentionally excluded from Git history.

## Repository contents

- Python scripts in the repository root build the dataset, train the research
  models, prepare the demo, and run the dashboard.
- `results/images/` contains the numbered figures used in the report.
- `results/tables/` contains the compact metric table, top-100 review list, and
  3D example metrics.
- `DATASET_DOCUMENTATION.md`, `methodology.md`, and
  `SYSTEM_ARCHITECTURE.md` describe the data provenance, method, and pipeline.

## Data availability

To reproduce the full pipeline, obtain the source inventories and DEMs described
in `DATASET_DOCUMENTATION.md` and place them in the paths expected by the build
scripts. These assets are excluded by `.gitignore` because they are large and/or
redistributed from third-party sources.

## Lake 3D and GLOF indicator demo

Run from this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dem_glof_app.py
```

Choose **Dataset demo lakes**, select one or more lakes (or **Select all available
lakes**), and click **Analyse**. The comparison table and chart show each lake's
score; download the table as CSV or choose **Lake to inspect in 3D** for details.
Results stay available when switching the detail view; changing the selected
lakes or analysis settings requires clicking Analyse again.
All 55 suitable 2024 GLO-HMA Sentinel-2 observations within the supplied local
`DEM.tif` coverage are included. No network access is needed in
dataset demo mode. **Upload a lake mask** retains the custom-input workflow.

The prepared masks rasterize actual inventory outlines at 10 m in local UTM;
this demo does not run a segmentation model. Geometry metrics and susceptibility
scores are calculated from the selected mask at analysis time, using the model
bundle and earlier inventory records from the same source. The indicator is
descriptive and is not an outburst probability. Scores may share the same class.

`demo_lakes/catalog.json` records master IDs, original lake IDs, observation IDs,
source, year, coordinates, and mask paths. Regenerate the masks and catalog:

```powershell
.\.venv\Scripts\python.exe prepare_demo_lakes.py
```

Check individual lakes, multiple selection, and the full catalog through the dashboard:

```powershell
.\.venv\Scripts\python.exe -m unittest test_demo_lakes.py
```
