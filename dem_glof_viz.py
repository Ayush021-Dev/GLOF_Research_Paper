"""
DEM.tif + prediction.tif  ->  interactive 3D lake/terrain page + GLOF susceptibility indicator.

  python dem_glof_viz.py --dem DEM.tif --mask prediction.tif --out lake_3d.html
         [--bundle prediction_models_ui/model.joblib]
         [--change1y 0.12 --change3y 0.31 --growth 0.10 --trend 0.08]   # km2 history features, optional

prediction.tif: 0 = background, >0 = lake. Both rasters must be georeferenced; CRS may differ
(they are reprojected to the local UTM zone). The indicator is the v1 fixed-weight score
(glof_predict.py logic). History features not supplied are treated as missing -> neutral 0.5.
"""
import argparse, json, math, os, urllib.request
import numpy as np
import rasterio
from rasterio import features as rfeat
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import Window
from rasterio.crs import CRS
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer
import plotly.graph_objects as go

FEATS = ["area_derived_km2", "perimeter_derived_km", "length_m", "width_m", "elongation_ratio", "compactness",
         "area_change_1y_km2", "area_change_3y_km2", "growth_rate_km2_per_year",
         "causal_long_term_area_trend_km2_per_year"]
NAMES = ["Lake area", "Perimeter", "Length", "Width", "Elongation", "Compactness",
         "1-year area change", "3-year area change", "Growth rate", "Long-term trend"]
WEIGHTS = dict(zip(FEATS, [.08, .06, .05, .05, .08, .08, .14, .14, .14, .18]))
# fallback copy of v1 reference limits (5th/95th percentile); replaced by --bundle when given
LIMITS = dict(zip(FEATS, [(0.0079, 14.48021), (0.4536, 29.47845), (127.37415, 6973.63317), (95.07861, 4667.27671),
                          (1.03179, 2.74161), (0.1081, 0.76397), (-0.2338, 0.35252), (-0.31413, 0.79811),
                          (-0.23001, 0.34694), (-0.04289, 0.17186)]))


def tile_name(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    return f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"


def fetch_dem(mask_p, out):
    """Download the Copernicus GLO-30 1-degree tile (public AWS bucket) that contains the lake mask."""
    with rasterio.open(mask_p) as m:
        w, s, e, n = transform_bounds(m.crs, "EPSG:4326", *m.bounds)
    if math.floor(w) != math.floor(e) or math.floor(s) != math.floor(n):
        print("WARNING: the mask spans more than one 1-degree tile; only the centre tile is downloaded.")
    name = tile_name((s + n) / 2, (w + e) / 2)
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, out)


def align(mask_p, dem_p):
    """Crop mask around the lake, reproject mask (nearest) and DEM (bilinear) to a common UTM grid."""
    with rasterio.open(mask_p) as m, rasterio.open(dem_p) as d:
        if m.crs is None or d.crs is None:
            raise SystemExit("Both rasters must be georeferenced (CRS missing).")
        a = m.read(1) > 0
        if not a.any():
            raise SystemExit("prediction.tif contains no lake pixels.")
        ys, xs = np.nonzero(a)
        pad = int(max(30, 1.0 * max(np.ptp(ys), np.ptp(xs))))
        r0, r1 = max(0, ys.min() - pad), min(m.height, ys.max() + pad + 1)
        c0, c1 = max(0, xs.min() - pad), min(m.width, xs.max() + pad + 1)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        sub = a[r0:r1, c0:c1].astype(np.uint8)
        tr, bounds = m.window_transform(win), rasterio.windows.bounds(win, m.transform)
        lon, lat = (lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))(transform_bounds(m.crs, "EPSG:4326", *bounds))
        utm = CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)
        T, W, H = calculate_default_transform(m.crs, utm, sub.shape[1], sub.shape[0], *bounds)
        mk = np.zeros((H, W), np.uint8)
        reproject(sub, mk, src_transform=tr, src_crs=m.crs, dst_transform=T, dst_crs=utm, resampling=Resampling.nearest)
        dm = np.full((H, W), np.nan, np.float32)
        reproject(d.read(1).astype(np.float32), dm, src_transform=d.transform, src_crs=d.crs, src_nodata=d.nodata,
                  dst_transform=T, dst_crs=utm, dst_nodata=np.nan, resampling=Resampling.bilinear)
    if np.isnan(dm).all():
        raise SystemExit("DEM does not cover the lake extent.")
    return mk.astype(bool), dm, T, utm, (lon, lat)


def analyse(mk, dem, T, utm):
    res = abs(T.a)
    lab, n = ndi.label(mk, structure=np.ones((3, 3)))
    sizes = ndi.sum(mk, lab, range(1, n + 1))
    comp = lab == (1 + int(np.argmax(sizes)))
    poly = max((shape(g) for g, v in rfeat.shapes(comp.astype(np.uint8), mask=comp, transform=T) if v == 1),
               key=lambda p: p.area)
    to_eq = Transformer.from_crs(utm, "ESRI:102025", always_xy=True).transform
    pe = shp_transform(to_eq, poly)
    pes = shp_transform(to_eq, poly.simplify(res * 0.75))
    area_km2, per_km = pe.area / 1e6, pes.length / 1e3
    x0, y0, x1, y1 = pe.bounds
    length, width = max(x1 - x0, y1 - y0), min(x1 - x0, y1 - y0)
    zl = dem[comp]
    gy, gx = np.gradient(dem, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    ring = ndi.binary_dilation(comp, iterations=max(1, int(500 / res))) & ~mk
    m = dict(area_km2=area_km2, perimeter_km=per_km, length_m=length, width_m=width,
             elongation=length / max(width, 1e-9), compactness=4 * math.pi * area_km2 * 1e6 / (per_km * 1e3) ** 2,
             lake_elev_mean=float(np.nanmean(zl)), lake_elev_min=float(np.nanmin(zl)), lake_elev_max=float(np.nanmax(zl)),
             lake_elev_range=float(np.nanmax(zl) - np.nanmin(zl)),
             surround_elev_mean=float(np.nanmean(dem[ring])), surround_elev_max=float(np.nanmax(dem[ring])),
             lake_slope_mean_deg=float(np.nanmean(slope[comp])), surround_slope_mean_deg=float(np.nanmean(slope[ring])),
             n_lake_components=int(n), pixel_m=float(res))
    m["relief_above_lake_m"] = m["surround_elev_max"] - m["lake_elev_mean"]
    m["surround_minus_lake_mean_m"] = m["surround_elev_mean"] - m["lake_elev_mean"]
    return m, comp, poly, slope


def indicator(m, hist, bundle):
    lim = bundle["stats"] if bundle else {k: dict(q05=v[0], q95=v[1]) for k, v in LIMITS.items()}
    w = bundle["weights"] if bundle else WEIGHTS
    vals = dict(zip(FEATS, [m["area_km2"], m["perimeter_km"], m["length_m"], m["width_m"], m["elongation"], m["compactness"],
                            hist.get("c1"), hist.get("c3"), hist.get("g"), hist.get("t")]))
    comp = {}
    for k in FEATS:
        v, lo, hi = vals[k], lim[k]["q05"], lim[k]["q95"]
        s = 0.5 if (v is None or hi <= lo) else float(np.clip((v - lo) / (hi - lo), 0, 1))
        comp[k] = 1 - s if k == "compactness" else s
    score = float(np.clip(100 * sum(comp[k] * w[k] for k in w), 0, 100))
    cls = "LOW" if score < 33.33 else "MODERATE" if score < 66.67 else "HIGH"
    contrib = sorted(((NAMES[FEATS.index(k)], 100 * comp[k] * w[k], vals[k] is None) for k in w), key=lambda t: -t[1])
    return score, cls, contrib[:5]


def figure(mk, dem, comp, poly, T, slope, m):
    rows, cols = np.nonzero(comp)
    pad = int(max(300 / m["pixel_m"], 1.5 * max(np.ptp(rows), np.ptp(cols))))
    r0, r1 = max(0, rows.min() - pad), min(dem.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(dem.shape[1], cols.max() + pad + 1)
    st = max(1, math.ceil(max(r1 - r0, c1 - c0) / 220))
    sl = (slice(r0, r1, st), slice(c0, c1, st))
    Z, S, L = dem[sl], slope[sl], mk[sl]
    E = T.c + (np.arange(c0, c1, st) + .5) * T.a
    N = T.f + (np.arange(r0, r1, st) + .5) * T.e
    ex = 1.0
    zmin = float(np.nanmin(Z))
    lake_z = np.where(L, Z + 2.0, np.nan)
    surf = go.Surface(x=E, y=N, z=Z, surfacecolor=Z, colorscale="Earth", name="Terrain", showscale=True,
                      colorbar=dict(title="Elevation (m)", len=.6), lighting=dict(ambient=.55, diffuse=.8, specular=.1),
                      hovertemplate="E %{x:.0f}<br>N %{y:.0f}<br>%{z:.0f} m<extra>terrain</extra>")
    lake = go.Surface(x=E, y=N, z=lake_z, surfacecolor=np.zeros_like(Z), colorscale=[[0, "#00c8ff"], [1, "#0077cc"]],
                      showscale=False, opacity=.95, name="Lake (prediction)", showlegend=True,
                      hovertemplate=f"Lake<br>{m['lake_elev_mean']:.0f} m mean<extra></extra>")
    bx, by = poly.exterior.xy
    bz = ndi.map_coordinates(np.nan_to_num(dem, nan=zmin), [(np.array(by) - T.f) / T.e - .5, (np.array(bx) - T.c) / T.a - .5], order=1) + 4
    foot = go.Scatter3d(x=list(bx), y=list(by), z=list(bz), mode="lines", line=dict(color="yellow", width=5), name="Lake footprint")
    fig = go.Figure([surf, lake, foot])
    btn = lambda lab, z: dict(label=lab, method="relayout", args=[{"scene.aspectratio.z": z}])
    asp = max(E.max() - E.min(), N.max() - N.min())
    base = float(np.nanmax(Z) - zmin) / asp
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0d1b2a", font=dict(color="#e0e6ed"), height=640,
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=(N.max() - N.min()) / (E.max() - E.min()), z=base * 2),
                   xaxis=dict(title="Easting (m)", backgroundcolor="#0d1b2a"), yaxis=dict(title="Northing (m)", backgroundcolor="#0d1b2a"),
                   zaxis=dict(title="Elevation (m)", backgroundcolor="#0d1b2a")),
        legend=dict(x=.01, y=.99, bgcolor="rgba(13,27,42,.7)"),
        updatemenus=[dict(type="buttons", direction="right", x=.5, y=1.0, xanchor="center", bgcolor="#1b2a3a",
                          buttons=[btn("Vertical ×1", base), btn("×2", base * 2), btn("×3", base * 3), btn("×5", base * 5)]),
                     dict(type="buttons", direction="right", x=.99, y=1.0, xanchor="right", bgcolor="#1b2a3a",
                          buttons=[dict(label="Colour: elevation", method="restyle", args=[{"surfacecolor": [Z], "colorscale": "Earth"}, [0]]),
                                   dict(label="Colour: slope", method="restyle", args=[{"surfacecolor": [S], "colorscale": "Viridis"}, [0]])])])
    return fig


def page(fig, m, score, cls, top, lonlat, banner, bundle_used):
    col = {"LOW": "#2ecc71", "MODERATE": "#f1c40f", "HIGH": "#e74c3c"}[cls]
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in [
        ("Centroid (lon, lat)", f"{lonlat[0]:.4f}, {lonlat[1]:.4f}"), ("Area", f"{m['area_km2']:.4f} km²"),
        ("Perimeter", f"{m['perimeter_km']:.2f} km"), ("Length × width", f"{m['length_m']:.0f} × {m['width_m']:.0f} m"),
        ("Elongation / compactness", f"{m['elongation']:.2f} / {m['compactness']:.2f}"),
        ("Lake elevation mean", f"{m['lake_elev_mean']:.0f} m"), ("Lake elevation min / max", f"{m['lake_elev_min']:.0f} / {m['lake_elev_max']:.0f} m"),
        ("Elevation range (lake)", f"{m['lake_elev_range']:.1f} m"), ("Surrounding elevation mean / max", f"{m['surround_elev_mean']:.0f} / {m['surround_elev_max']:.0f} m"),
        ("Relief above lake (max − mean)", f"{m['relief_above_lake_m']:.0f} m"), ("Slope, lake / 500 m surroundings", f"{m['lake_slope_mean_deg']:.1f}° / {m['surround_slope_mean_deg']:.1f}°"),
        ("Lake components in mask", m["n_lake_components"])])
    bars = "".join(f"<div class='f'><span>{n}{' <i>(missing → neutral)</i>' if miss else ''}</span><b>{v:.1f}</b>"
                   f"<div class='bar'><div style='width:{min(100, v / 18 * 100):.0f}%'></div></div></div>" for n, v, miss in top)
    ban = f"<div class='ban'>{banner}</div>" if banner else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Lake 3D + GLOF indicator</title><style>
body{{margin:0;background:#0d1b2a;color:#e0e6ed;font:14px system-ui,sans-serif}}.wrap{{display:grid;grid-template-columns:340px 1fr;gap:12px;padding:12px}}
.card{{background:#16263a;border-radius:8px;padding:12px;margin-bottom:12px}}h3{{margin:0 0 8px;font-size:15px}}table{{width:100%;border-collapse:collapse}}td{{padding:3px 0;border-bottom:1px solid #223}}td:last-child{{text-align:right}}
.score{{font-size:44px;font-weight:700;color:{col}}}.badge{{background:{col};color:#111;border-radius:4px;padding:2px 10px;font-weight:700}}
.scale{{position:relative;height:10px;border-radius:5px;background:linear-gradient(90deg,#2ecc71 33%,#f1c40f 33% 67%,#e74c3c 67%);margin:8px 0}}.scale i{{position:absolute;top:-4px;left:{score:.1f}%;width:3px;height:18px;background:#fff}}
.f{{margin:6px 0}}.f b{{float:right}}.bar{{background:#223;height:6px;border-radius:3px}}.bar div{{height:6px;border-radius:3px;background:#00c8ff}}
.note{{font-size:12px;color:#aab}}.ban{{background:#c0392b;color:#fff;padding:6px 12px;font-weight:700;text-align:center}}
@media(max-width:900px){{.wrap{{grid-template-columns:1fr}}}}</style></head><body>{ban}<div class='wrap'><div>
<div class='card'><h3>GLOF susceptibility indicator</h3><span class='score'>{score:.1f}</span> / 100 &nbsp;<span class='badge'>{cls}</span>
<div class='scale'><i></i></div><div class='note'>LOW &lt; 33.3 ≤ MODERATE &lt; 66.7 ≤ HIGH</div><h3 style='margin-top:10px'>Top contributions</h3>{bars}
<p class='note'>Descriptive indicator from fixed weights ({'model bundle limits' if bundle_used else 'built-in v1 limits'}). Not a calibrated probability and not validated as a hazard predictor: on documented GLOF lakes (Sentinel-2) it did not beat lake area alone (AUC 0.55 vs 0.73). History features not supplied count as neutral. With the current reference limits, Sentinel-2 lakes rarely leave the LOW class.</p></div>
<div class='card'><h3>Terrain and lake metrics</h3><table>{rows}</table></div></div>
<div>{fig.to_html(full_html=False, include_plotlyjs=True, div_id='plot')}<p class='note'>Drag to rotate, scroll to zoom. Lake = prediction mask (cyan), footprint = yellow outline. Elevation: Copernicus GLO-30 (a surface model; lake values include DEM noise).</p></div></div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True); ap.add_argument("--mask", required=True); ap.add_argument("--out", default="lake_3d.html")
    ap.add_argument("--bundle"); ap.add_argument("--banner", default="")
    ap.add_argument("--fetch-dem", action="store_true", help="download the Copernicus GLO-30 tile to --dem if that file is missing")
    for k in ("change1y", "change3y", "growth", "trend"):
        ap.add_argument(f"--{k}", type=float)
    a = ap.parse_args()
    if not os.path.isfile(a.mask):
        raise SystemExit(f"mask file not found: {a.mask}\nCurrent folder: {os.getcwd()}")
    if not os.path.isfile(a.dem):
        if not a.fetch_dem:
            raise SystemExit(f"DEM file not found: {a.dem}\nCurrent folder: {os.getcwd()}\nAdd --fetch-dem to download the Copernicus GLO-30 tile for this lake.")
        fetch_dem(a.mask, a.dem)
    bundle = None
    if a.bundle:
        import joblib
        bundle = joblib.load(a.bundle)
    mk, dem, T, utm, lonlat = align(a.mask, a.dem)
    m, comp, poly, slope = analyse(mk, dem, T, utm)
    hist = dict(c1=a.change1y, c3=a.change3y, g=a.growth, t=a.trend)
    score, cls, top = indicator(m, hist, bundle)
    fig = figure(mk, dem, comp, poly, T, slope, m)
    open(a.out, "w", encoding="utf-8").write(page(fig, m, score, cls, top, lonlat, a.banner, bool(bundle)))
    json.dump(dict(metrics=m, score=score, cls=cls, centroid_lonlat=lonlat, history_given=hist),
              open(a.out.replace(".html", "_metrics.json"), "w"), indent=2)
    print(f"{cls} {score:.1f} | area {m['area_km2']:.4f} km2 | mean elev {m['lake_elev_mean']:.0f} m -> {a.out}")


if __name__ == "__main__":
    main()