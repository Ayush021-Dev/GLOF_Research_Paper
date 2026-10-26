"""
Interactive GLOF lake viewer.   Run:  streamlit run dem_glof_app.py     (keep dem_glof_viz.py in the same folder)

Everything is chosen in the browser: lake mask, DEM (upload or auto-download), model bundle, history source,
image year, class cut-offs. Reference limits and weights are read from the model bundle - none are built in.
"""
import glob, json, os, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import joblib
from pyproj import Transformer
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import dem_glof_viz as v

ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False, max_entries=2)
def demo_catalog(modified):
    catalog = ROOT / "demo_lakes/catalog.json"
    return json.loads(catalog.read_text(encoding="utf-8")) if catalog.is_file() else []


@st.cache_data(show_spinner=False, max_entries=64)
def analyse_files(mask_p, dem_p, mask_modified, dem_modified):
    mk, dem, T, utm, lonlat = v.align(mask_p, dem_p)
    m, comp, poly, slope = v.analyse(mk, dem, T, utm)
    return mk, dem, T, utm, lonlat, m, comp, poly, slope


def find(pattern):
    hits = sorted(glob.glob(os.path.join("**", pattern), recursive=True))
    return hits[0] if hits else ""


@st.cache_data(show_spinner="Loading inventory tables...", max_entries=4)
def load_tables(master_p, feat_p):
    ml = pd.read_csv(master_p, usecols=["master_lake_id", "centroid_x_m", "centroid_y_m"])
    ft = pd.read_csv(feat_p, usecols=["master_lake_id", "source_dataset", "observation_year", "area_derived_km2"])
    return ml, ft


def lookup_lake(ml, x, y, radius):
    d, i = cKDTree(ml[["centroid_x_m", "centroid_y_m"]].to_numpy()).query([x, y])
    return (ml.master_lake_id.iloc[i], float(d)) if d <= radius else (None, float(d))


def history_from_inventory(obs, year, a_new):
    """Change features for a new observation, using only inventory records from earlier years."""
    prev = obs[obs.observation_year < year]
    if prev.empty:
        return {}
    at = lambda y: prev.loc[prev.observation_year == y, "area_derived_km2"].mean() if (prev.observation_year == y).any() else None
    ly = prev.observation_year.max()
    la = prev.loc[prev.observation_year == ly, "area_derived_km2"].mean()
    h = {"g": float((a_new - la) / (year - ly))}
    if at(year - 1) is not None: h["c1"] = float(a_new - at(year - 1))
    if at(year - 3) is not None: h["c3"] = float(a_new - at(year - 3))
    ys = np.r_[prev.observation_year.to_numpy(float), year]
    ar = np.r_[prev.area_derived_km2.to_numpy(float), a_new]
    h["t"] = float(np.polyfit(ys, ar, 1)[0]) if len(np.unique(ys)) >= 2 else 0.0
    return h


def save_upload(up, suffix=".tif"):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(up.getbuffer()); f.close()
    return f.name


def render_result(lake, data, hist, note, score, cls, top):
    mk, dem, T, utm, lonlat, m, comp, poly, slope = data
    c1, c2, c3 = st.columns(3)
    c1.metric("Indicator score", f"{score:.1f} / 100")
    c2.metric("Class", cls)
    c3.metric("Lake area", f"{m['area_km2']:.4f} km²")
    st.caption(note + " Descriptive indicator, not a calibrated probability and not validated as a hazard predictor.")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Top contributions")
        st.bar_chart(pd.Series({n: val for n, val, _ in top}))
        if hist:
            st.write("History features used:", {k: round(x, 4) for k, x in hist.items()})
        st.subheader("Metrics")
        st.dataframe(pd.Series({k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}, name="value"))
    fig = v.figure(mk, dem, comp, poly, T, slope, m)
    with right:
        st.plotly_chart(fig, width="stretch", alt="Selected lake outline and surrounding DEM terrain in 3D")
    st.download_button("Download this view as a standalone HTML page",
                       v.page(fig, m, score, cls, top, lonlat, "", True),
                       file_name=f"{lake['master_lake_id']}_3d.html", mime="text/html")


def dataset_results(lakes, bundle, mode, man, feat_p, master_p, lo, hi, run, signature):
    if run:
        results, failures = [], []
        ft = None
        if mode == "Look up in inventory":
            if not (os.path.isfile(master_p) and os.path.isfile(feat_p)):
                st.error("Inventory tables are missing. Correct the paths or choose another history mode.")
                return
            _, ft = load_tables(master_p, feat_p)
        try:
            manual = {k: float(x) for k, x in man.items() if x.strip()}
        except ValueError:
            st.error("Manual history values must be numbers.")
            return
        progress = st.progress(0, text="Analysing selected lakes…")
        for i, lake in enumerate(lakes):
            try:
                mask_p, dem_p = str(ROOT / lake['mask_path']), str(ROOT / lake['dem_path'])
                data = analyse_files(mask_p, dem_p, os.path.getmtime(mask_p), os.path.getmtime(dem_p))
                m = data[5]
                hist, note = {}, "No history features used; they count as neutral 0.5."
                if mode == "Enter manually":
                    hist, note = manual, "Manual history applied to the selected lakes."
                elif ft is not None:
                    obs = ft[(ft.master_lake_id == lake['master_lake_id'])
                             & (ft.source_dataset == lake['source_dataset'])
                             & (ft.observation_year < lake['observation_year'])]
                    hist = history_from_inventory(obs, lake['observation_year'], m['area_km2'])
                    note = f"{len(obs)} earlier records from {lake['source_dataset']}."
                score, _, top = v.indicator(m, hist, bundle)
                cls = "LOW" if score < lo else "HIGH" if score >= hi else "MODERATE"
                results.append(dict(lake=lake, hist=hist, note=note, score=score, cls=cls, top=top,
                                    area=m['area_km2'], elevation=m['lake_elev_mean']))
            except (Exception, SystemExit) as exc:
                failures.append(f"{lake['master_lake_id']}: {exc}")
            progress.progress((i + 1) / len(lakes), text=f"Analysed {i + 1} / {len(lakes)} lakes")
        progress.empty()
        st.session_state['dataset_results'] = (signature, results, failures)
    saved = st.session_state.get('dataset_results')
    if not saved or saved[0] != signature:
        st.info("Select one or more dataset lakes, then click Analyse.")
        return
    _, results, failures = saved
    for failure in failures:
        st.warning(failure)
    if not results:
        st.error("No selected lakes could be analysed.")
        return
    summary = pd.DataFrame([dict(Lake=r['lake']['master_lake_id'], Source=r['lake']['source_dataset'],
                                Year=r['lake']['observation_year'], Score=round(r['score'], 2),
                                Class=r['cls'], Area_km2=round(r['area'], 5),
                                Elevation_m=round(r['elevation'], 1)) for r in results])
    st.subheader(f"Results for {len(results)} lakes")
    st.dataframe(summary, hide_index=True, alt="Scores and terrain metrics for the analysed lakes")
    st.download_button("Download comparison CSV", summary.to_csv(index=False), "lake_comparison.csv", "text/csv")
    if len(results) > 1:
        st.bar_chart(summary.set_index('Lake')['Score'], alt="Susceptibility indicator scores by lake")
    by_id = {r['lake']['master_lake_id']: r for r in results}
    detail = st.selectbox("Lake to inspect in 3D", list(by_id), key="detail_lake")
    r = by_id[detail]
    lake = r['lake']
    st.subheader(f"Dataset lake {detail}")
    st.caption(f"Source ID: {lake['original_lake_id']} · {lake['latitude']:.5f}° N, {lake['longitude']:.5f}° E")
    mask_p, dem_p = str(ROOT / lake['mask_path']), str(ROOT / lake['dem_path'])
    data = analyse_files(mask_p, dem_p, os.path.getmtime(mask_p), os.path.getmtime(dem_p))
    render_result(lake, data, r['hist'], r['note'], r['score'], r['cls'], r['top'])


def main():
    st.set_page_config(page_title="Lake 3D + GLOF indicator", layout="wide")
    st.title("Lake 3D viewer and GLOF susceptibility indicator")
    sb = st.sidebar
    sb.header("Inputs")
    input_mode = sb.radio("Lake input", ["Dataset demo lakes", "Upload a lake mask"], key="lake_input")
    selected = None
    mask_up = dem_up = None
    auto_dem = False
    if input_mode == "Dataset demo lakes":
        catalog_path = ROOT / "demo_lakes/catalog.json"
        catalog = demo_catalog(catalog_path.stat().st_mtime_ns if catalog_path.exists() else 0)
        if not catalog:
            st.error("Demo lakes are missing. Run: python prepare_demo_lakes.py")
            return
        all_lakes = sb.checkbox("Select all available lakes", key="all_lakes")
        choices = sb.multiselect("Select dataset lakes", range(len(catalog)), default=[0], key="demo_lakes",
                                disabled=all_lakes,
                                format_func=lambda i: f"{catalog[i]['master_lake_id']} | {catalog[i]['inventory_area_km2']:.3f} km² | 2024")
        lakes = catalog if all_lakes else [catalog[i] for i in choices]
        selected = lakes[0] if lakes else None
        st.caption(f"{len(catalog)} dataset lakes available within the local DEM coverage · "
                   f"{len(lakes)} selected · GLO-HMA Sentinel-2, 2024.")
        st.caption("Masks come from dataset lake outlines. Analyse compares the selected lakes; choose a result to inspect in 3D. Segmentation is not run here.")
        sb.caption("Uses the prepared dataset mask and local DEM; no uploads or downloads needed.")
    else:
        mask_up = sb.file_uploader("Lake mask (prediction.tif, georeferenced, 0 = background)", type=["tif", "tiff"])
        dem_up = sb.file_uploader("DEM (GeoTIFF) - optional", type=["tif", "tiff"])
        auto_dem = sb.checkbox("Download the Copernicus GLO-30 tile if no DEM is uploaded", value=True)
    bundle_up = sb.file_uploader("Model bundle (model.joblib) - or use the path below", type=["joblib"])
    bundle_path = sb.text_input("Model bundle path", value=str(ROOT / "prediction_models_ui/model.joblib"))
    sb.header("History")
    mode = sb.radio("History features", ["Look up in inventory", "Enter manually", "None (neutral 0.5)"])
    year = sb.number_input("Observation year of the mask", 1990, 2100, 2024, disabled=selected is not None)
    src = sb.selectbox("Inventory source for history", ["GLO-HMA-S2", "GLO-HMA-S1", "30m-TP", "Any"], disabled=selected is not None)
    if selected:
        year, src = selected["observation_year"], selected["source_dataset"]
    radius = sb.number_input("Match radius to inventory lake (m)", 50, 5000, 500)
    master_p = sb.text_input("master_lakes.csv path", value=str(ROOT / "master_lake_dataset/master_lakes.csv"))
    feat_p = sb.text_input("lake_features.csv path", value=str(ROOT / "master_lake_dataset/lake_features.csv"))
    man = {k: sb.text_input(l, "") for k, l in [("c1", "1-year area change (km²)"), ("c3", "3-year area change (km²)"),
                                                 ("g", "Growth rate (km²/yr)"), ("t", "Long-term trend (km²/yr)")]} if mode == "Enter manually" else {}
    sb.header("Classes")
    lo = sb.number_input("LOW below", 0.0, 100.0, 33.33)
    hi = sb.number_input("HIGH from", 0.0, 100.0, 66.67)
    go_btn = sb.button("Analyse", type="primary")

    if not go_btn and input_mode != "Dataset demo lakes":
        st.info("Choose the inputs in the sidebar and press **Analyse**.")
        return
    if lo >= hi:
        st.error("LOW threshold must be below the HIGH threshold."); return
    if input_mode == "Dataset demo lakes" and not lakes:
        st.info("Select at least one dataset lake."); return
    if input_mode != "Dataset demo lakes" and mask_up is None:
        st.error("Upload a lake mask first."); return
    if bundle_up is not None:
        bundle = joblib.load(bundle_up)
    elif bundle_path and os.path.isfile(bundle_path):
        bundle = joblib.load(bundle_path)
    else:
        st.error("Provide the model bundle (upload it or enter a valid path)."); return

    if input_mode == "Dataset demo lakes":
        signature = (tuple(lake['master_lake_id'] for lake in lakes), mode, tuple(man.items()),
                     feat_p, master_p, lo, hi,
                     bundle_up.getvalue() if bundle_up is not None else (bundle_path, os.path.getmtime(bundle_path)),
                     catalog_path.stat().st_mtime_ns)
        dataset_results(lakes, bundle, mode, man, feat_p, master_p, lo, hi, go_btn, signature)
        return

    mask_p = str(ROOT / selected["mask_path"]) if selected else save_upload(mask_up)
    if selected:
        dem_p = str(ROOT / selected["dem_path"])
    elif dem_up is not None:
        dem_p = save_upload(dem_up)
    elif auto_dem:
        dem_p = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with st.spinner("Downloading Copernicus GLO-30 tile..."):
            v.fetch_dem(mask_p, dem_p)
    else:
        st.error("Upload a DEM or tick the download option."); return

    with st.spinner("Aligning rasters and analysing the lake..."):
        if not (os.path.isfile(mask_p) and os.path.isfile(dem_p)):
            st.error("Lake mask or DEM file is missing. Run prepare_demo_lakes.py to recreate the demo masks.")
            return
        mk, dem, T, utm, lonlat, m, comp, poly, slope = analyse_files(
            mask_p, dem_p, os.path.getmtime(mask_p), os.path.getmtime(dem_p))

    hist, note = {}, "No history features used; they count as neutral 0.5."
    if mode == "Enter manually":
        hist = {k: float(x) for k, x in man.items() if x.strip()}
        note = "History entered manually."
    elif mode == "Look up in inventory":
        if not (os.path.isfile(master_p) and os.path.isfile(feat_p)):
            st.warning("Inventory tables not found; scoring without history.")
        else:
            ml, ft = load_tables(master_p, feat_p)
            cx, cy = Transformer.from_crs(utm, "ESRI:102025", always_xy=True).transform(poly.centroid.x, poly.centroid.y)
            mid, dist = (selected["master_lake_id"], 0.0) if selected else lookup_lake(ml, cx, cy, radius)
            if mid is None:
                note = f"No inventory lake within {radius} m (nearest {dist:.0f} m); scoring without history."
            else:
                obs = ft[ft.master_lake_id == mid]
                obs = obs if src == "Any" else obs[obs.source_dataset == src]
                obs = obs[obs.observation_year < int(year)]
                hist = history_from_inventory(obs, int(year), m["area_km2"])
                note = (f"Dataset lake {mid}; " if selected else f"Matched {mid} ({dist:.0f} m away); ") + f"{len(obs)} earlier records from {src}."

    score, _, top = v.indicator(m, hist, bundle)
    cls = "LOW" if score < lo else "HIGH" if score >= hi else "MODERATE"
    c1, c2, c3 = st.columns(3)
    c1.metric("Indicator score", f"{score:.1f} / 100"); c2.metric("Class", cls); c3.metric("Lake area", f"{m['area_km2']:.4f} km²")
    st.caption(note + " Descriptive indicator, not a calibrated probability and not validated as a hazard predictor.")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Top contributions")
        st.bar_chart(pd.Series({n: val for n, val, _ in top}))
        if hist: st.write("History features used:", {k: round(x, 4) for k, x in hist.items()})
        st.subheader("Metrics")
        st.dataframe(pd.Series({k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}, name="value"))
    fig = v.figure(mk, dem, comp, poly, T, slope, m)
    with right:
        st.plotly_chart(fig, width="stretch", alt="Selected lake outline and surrounding DEM terrain in 3D")
    st.download_button("Download this view as a standalone HTML page",
                       v.page(fig, m, score, cls, top, lonlat, "", True),
                       file_name=f"{selected['master_lake_id'] if selected else 'lake'}_3d.html", mime="text/html")


if __name__ == "__main__":
    main()
