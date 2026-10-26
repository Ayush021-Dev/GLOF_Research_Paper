"""Build a provenance-preserving HMA lake integration dataset.

No imagery, masks, models, imputation, negative GLOF labels, or inferred event
dates are used.  All measurements are from source polygons in ESRI:102025.
"""
from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "master_lake_dataset"
CRS = "ESRI:102025"
WGS84 = "EPSG:4326"

# Conservative, published-in-output rules.  They admit only pairs with
# substantial polygon agreement; otherwise they remain ambiguous/unmatched.
SEARCH_DISTANCE_M = 750.0
MAX_CENTROID_DISTANCE_M = 500.0
MIN_IOU = 0.20
MIN_AREA_RATIO = 0.20                 # smaller/larger area
SCORE_MARGIN = 0.12                   # required separation from runner-up
EVENT_NEAR_DISTANCE_M = 250.0         # source coordinate can be any lake point


class DSU:
    def __init__(self): self.p = {}
    def add(self, x): self.p.setdefault(x, x)
    def find(self, x):
        self.add(x)
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def valid_geom(g):
    if g is None or g.is_empty: return None
    if not g.is_valid: g = g.buffer(0)
    return None if g.is_empty else g


def make_record(node, source, source_id, year, geom, product, attrs, sensor=None):
    geom = valid_geom(geom)
    if geom is None: return None
    return {"node": node, "source_dataset": source, "source_id": str(source_id),
            "year": int(year) if pd.notna(year) else None, "geometry": geom,
            "product_reference": str(product), "attributes": attrs, "sensor": sensor}


def metric(a, b):
    ga, gb = a["geometry"], b["geometry"]
    aa, ab = ga.area, gb.area
    inter = ga.intersection(gb).area
    union = ga.union(gb).area
    cd = ga.centroid.distance(gb.centroid)
    ratio = min(aa, ab) / max(aa, ab) if aa and ab else 0.0
    return {"centroid_distance_m": cd, "intersection_area_m2": inter,
            "union_area_m2": union, "iou": inter / union if union else 0.0,
            "area_ratio": ratio, "area_difference_m2": abs(aa-ab),
            "score": (inter / union if union else 0.0) * 0.8 + ratio * 0.2}


def candidates(left, right, pair_type, search=SEARCH_DISTANCE_M):
    """Generate every geographically nearby candidate and its objective metrics."""
    # The acceptance rule already rejects centroid distances >500 m.  Indexing
    # centroids (rather than buffered, potentially very complex polygons) is
    # mathematically lossless for accepted pairs and makes dense HMA clusters
    # tractable without relaxing any scientific criterion.
    rg = gpd.GeoSeries([r["geometry"].centroid for r in right], crs=CRS)
    idx = rg.sindex
    out = []
    for li, a in enumerate(left):
        ac = a["geometry"].centroid
        possible = idx.query(ac.buffer(MAX_CENTROID_DISTANCE_M), predicate="intersects")
        for ri in possible:
            b = right[int(ri)]
            m = metric(a, b)
            if m["centroid_distance_m"] <= MAX_CENTROID_DISTANCE_M:
                out.append({"left": a, "right": b, "pair_type": pair_type, **m})
    return out


def classify_and_union(rows, dsu):
    """Mutual-best, score-separated, strong geometric matches only."""
    by_l, by_r = defaultdict(list), defaultdict(list)
    for r in rows:
        strong = (r["iou"] >= MIN_IOU and r["area_ratio"] >= MIN_AREA_RATIO and
                  r["centroid_distance_m"] <= MAX_CENTROID_DISTANCE_M)
        r["strong"] = strong
        by_l[r["left"]["node"]].append(r)
        by_r[r["right"]["node"]].append(r)
    for group in (by_l, by_r):
        for vals in group.values(): vals.sort(key=lambda x: x["score"], reverse=True)
    accepted = set()
    for r in rows:
        lvals, rvals = by_l[r["left"]["node"]], by_r[r["right"]["node"]]
        lmargin = r["score"] - (lvals[1]["score"] if len(lvals) > 1 else 0)
        rmargin = r["score"] - (rvals[1]["score"] if len(rvals) > 1 else 0)
        if r["strong"] and lvals[0] is r and rvals[0] is r and lmargin >= SCORE_MARGIN and rmargin >= SCORE_MARGIN:
            accepted.add(id(r)); dsu.union(r["left"]["node"], r["right"]["node"])
    result = []
    for r in rows:
        if id(r) in accepted: status, reason = "MATCHED", "mutual_best_strong_geometric_match"
        elif r["strong"] or (r["iou"] >= 0.05 and r["centroid_distance_m"] <= MAX_CENTROID_DISTANCE_M):
            status, reason = "AMBIGUOUS", "competing_or_insufficiently_separated_candidate"
        else: status, reason = "UNMATCHED", "does_not_meet_strong_geometric_thresholds"
        result.append({"left_node":r["left"]["node"], "right_node":r["right"]["node"],
                       "left_dataset":r["left"]["source_dataset"], "right_dataset":r["right"]["source_dataset"],
                       "left_original_lake_id":r["left"]["source_id"], "right_original_lake_id":r["right"]["source_id"],
                       "left_year":r["left"]["year"], "right_year":r["right"]["year"], "pair_type":r["pair_type"],
                       "match_status":status, "decision_reason":reason,
                       **{k:r[k] for k in ["centroid_distance_m","intersection_area_m2","union_area_m2","iou","area_ratio","area_difference_m2","score"]}})
    return result


def read_glo(sensor):
    all_path = ROOT / "GLO" / f"{sensor}_20172024_TB_GLOID_v1_0.gpkg"
    uni_path = ROOT / "GLO" / f"{sensor}_20172024_TB_GLOID_UniqueLakes_v1_0.gpkg"
    allg, unig = gpd.read_file(all_path).to_crs(CRS), gpd.read_file(uni_path).to_crs(CRS)
    records, uniques = [], []
    tag = sensor.lower()
    for _, row in unig.iterrows():
        sid = str(row.GLO_ID); node = f"{tag}|{sid}"
        r = make_record(node, f"GLO-HMA-{sensor}", sid, None, row.geometry, uni_path,
                        row.drop(labels="geometry").to_dict(), sensor)
        if r: uniques.append(r)
    for ix, row in allg.iterrows():
        sid = str(row.GLO_ID)
        r = make_record(f"{tag}|{sid}", f"GLO-HMA-{sensor}", sid, row.AREA_YEAR, row.geometry, all_path,
                        row.drop(labels="geometry").to_dict(), sensor)
        if r: records.append(r)
    return records, uniques, allg.crs


def read_tp():
    records, by_year = [], defaultdict(list)
    for path in sorted((ROOT / "Lake_TP").glob("*/Lake_TP_*.shp"), key=lambda p:int(p.parent.name)):
        year = int(path.parent.name); g = gpd.read_file(path).to_crs(CRS)
        for fid, row in g.iterrows():
            # Dataset has no lake ID field: this is solely the original Shapefile FID reference.
            sid = f"tp:{year}:{fid}"
            r = make_record(f"tp|{year}|{fid}", "30m-TP", sid, year, row.geometry, path,
                            {"area": row.get("area"), "source_feature_fid": int(fid)}, "Landsat-derived 30m")
            if r: records.append(r); by_year[year].append(r)
    return records, by_year


def source_area(rec):
    a = rec["attributes"]
    if rec["source_dataset"].startswith("GLO-HMA"): return a.get("AREA")
    return a.get("area")


def obs_rows(records, master):
    rows = []
    for n, r in enumerate(records, 1):
        a = r["attributes"]
        start = a.get("START_DATE") if r["source_dataset"].startswith("GLO-HMA") else None
        end = a.get("END_DATE") if r["source_dataset"].startswith("GLO-HMA") else None
        rows.append({"master_lake_id": master[r["node"]], "observation_id": f"OBS{n:09d}",
                     "source_dataset":r["source_dataset"], "sensor":r["sensor"],
                     "original_lake_id":r["source_id"], "observation_year":r["year"],
                     "observation_start_date":start, "observation_end_date":end,
                     "lake_area":source_area(r), "lake_area_unit":"km2",
                     "lake_area_value_type":"original_source_attribute",
                     "geometry_wkt":r["geometry"].wkt, "_geometry":r["geometry"], "geometry_crs":CRS,
                     "image_product_reference":r["product_reference"],
                     "original_attributes_json":json.dumps(a, default=str, ensure_ascii=False)})
    return rows


def add_features(obs):
    transformer = Transformer.from_crs(CRS, WGS84, always_xy=True)
    # Vectorised calculations over the original projected source geometries.
    # Length/width are the two axis-aligned bounding-box sides (not an inferred
    # physical lake axis); the definition travels with every output row.
    gs = gpd.GeoSeries(obs["_geometry"].tolist(), crs=CRS)
    cent = gs.centroid; bounds = gs.bounds
    dx=(bounds.maxx-bounds.minx).to_numpy(); dy=(bounds.maxy-bounds.miny).to_numpy()
    length=np.maximum(dx,dy); width=np.minimum(dx,dy)
    area=gs.area.to_numpy(); perimeter=gs.length.to_numpy()
    lon,lat=transformer.transform(cent.x.to_numpy(),cent.y.to_numpy())
    f=obs[["observation_id","master_lake_id","source_dataset","sensor","observation_year"]].copy()
    f["area_derived_km2"]=area/1e6; f["perimeter_derived_km"]=perimeter/1000
    f["centroid_x_m"]=cent.x.to_numpy(); f["centroid_y_m"]=cent.y.to_numpy(); f["centroid_lon"]=lon; f["centroid_lat"]=lat
    f["length_m"]=length; f["width_m"]=width; f["elongation_ratio"]=np.divide(length,width,out=np.full(len(f),np.nan),where=width!=0)
    f["compactness"]=np.divide(4*math.pi*area,perimeter**2,out=np.full(len(f),np.nan),where=perimeter!=0)
    f["shape_metric_definition"]="length_width=axis_aligned_bounding_box_sides_in_ESRI_102025; compactness=4*pi*area/perimeter^2"
    f["area_change_1y_km2"]=np.nan; f["area_change_3y_km2"]=np.nan; f["growth_rate_km2_per_year"]=np.nan; f["long_term_area_trend_km2_per_year"]=np.nan
    f["temporal_series_scope"]="master_lake_id + source_dataset + sensor"
    group_cols=["master_lake_id","source_dataset","sensor"]
    for _, idx in f.groupby(group_cols, dropna=False).groups.items():
        if len(idx) < 2: continue
        q=f.loc[idx].sort_values("observation_year")
        years=q.observation_year.astype(float).to_numpy(); vals=q.area_derived_km2.to_numpy()
        byyear=dict(zip(years, vals))
        trend=np.polyfit(years,vals,1)[0] if len(q)>=2 and len(set(years))>=2 else np.nan
        for i, (rowidx, yr, val) in enumerate(zip(q.index, years, vals)):
            f.loc[rowidx,"area_change_1y_km2"] = val-byyear[yr-1] if yr-1 in byyear else np.nan
            f.loc[rowidx,"area_change_3y_km2"] = val-byyear[yr-3] if yr-3 in byyear else np.nan
            if i and yr>years[i-1]: f.loc[rowidx,"growth_rate_km2_per_year"]=(val-vals[i-1])/(yr-years[i-1])
            f.loc[rowidx,"long_term_area_trend_km2_per_year"] = trend
    return f


def match_events(events, reps, master, match_rows):
    """Source-lake point matches only; event dates remain exactly as provided."""
    geoms=gpd.GeoSeries([r["geometry"] for r in reps], crs=CRS); sidx=geoms.sindex
    out=[]
    to_crs=Transformer.from_crs(WGS84, CRS, always_xy=True)
    for _, e in events.iterrows():
        event_id=str(e.GF_ID); lon,lat=pd.to_numeric(e.Lon_lake,errors="coerce"),pd.to_numeric(e.Lat_lake,errors="coerce")
        meta=e.to_dict(); candidates_event=[]
        if pd.notna(lon) and pd.notna(lat):
            x,y=to_crs.transform(lon,lat); p=Point(x,y)
            for i in sidx.query(p.buffer(EVENT_NEAR_DISTANCE_M), predicate="intersects"):
                r=reps[int(i)]; d=p.distance(r["geometry"]); inside=r["geometry"].covers(p)
                candidates_event.append((r,d,inside))
        candidates_event.sort(key=lambda z:(not z[2],z[1]))
        if len(candidates_event)==1 and (candidates_event[0][2] or candidates_event[0][1] <= EVENT_NEAR_DISTANCE_M):
            chosen, status = candidates_event[0][0], "MATCHED"
            reason = "unique_source_lake_point_spatial_candidate"
        elif candidates_event:
            chosen, status, reason = None, "AMBIGUOUS", "multiple_source_lake_point_spatial_candidates"
        else: chosen, status, reason = None, "UNMATCHED", "no_current_inventory_polygon_within_250m_of_source_lake_coordinate"
        # All documented events are retained, including unlinked ones.
        event_date = "-".join(str(e[c]) for c in ["Year_exact","Month","Day"] if str(e[c]) not in ("NA","nan","None")) or None
        out.append({"master_lake_id":master[chosen["node"]] if chosen else None, "glof_event_id":event_id,
                    "event_date_as_provided":event_date, "event_year_exact":e.Year_exact,
                    "event_year_approx":e.Year_approx, "event_location_lon":lon, "event_location_lat":lat,
                    "event_location_crs":WGS84, "event_match_status":status, "event_match_reason":reason,
                    "original_event_metadata_json":json.dumps(meta,default=str,ensure_ascii=False),
                    "source":"HMAGLOFDB (Steiner and Shrestha; supplied original CSV)"})
        if candidates_event:
            for r,d,inside in candidates_event:
                match_rows.append({"left_node":f"glof|{event_id}","right_node":r["node"],"left_dataset":"HMAGLOFDB",
                    "right_dataset":r["source_dataset"],"left_original_lake_id":event_id,"right_original_lake_id":r["source_id"],
                    "left_year":e.Year_exact,"right_year":r["year"],"pair_type":"glof_source_point_to_lake_polygon",
                    "match_status":status if chosen is r else "AMBIGUOUS","decision_reason":reason,
                    "centroid_distance_m":np.nan,"intersection_area_m2":np.nan,"union_area_m2":np.nan,"iou":np.nan,"area_ratio":np.nan,
                    "area_difference_m2":np.nan,"score":-d,"event_point_distance_to_polygon_m":d,"event_point_within_polygon":inside})
        else:
            match_rows.append({"left_node":f"glof|{event_id}","right_node":None,"left_dataset":"HMAGLOFDB","right_dataset":None,
              "left_original_lake_id":event_id,"right_original_lake_id":None,"left_year":e.Year_exact,"right_year":None,
              "pair_type":"glof_source_point_to_lake_polygon","match_status":"UNMATCHED","decision_reason":reason,
              "centroid_distance_m":np.nan,"intersection_area_m2":np.nan,"union_area_m2":np.nan,"iou":np.nan,"area_ratio":np.nan,"area_difference_m2":np.nan,"score":np.nan})
    return pd.DataFrame(out)


def qa_plot(match_df, allnodes, out):
    for status, fname in [("MATCHED","qa_high_confidence_matches.png"),("AMBIGUOUS","qa_ambiguous_matches.png"),("UNMATCHED","qa_rejected_matches.png")]:
        q=match_df[(match_df.match_status==status)&match_df.left_node.notna()&match_df.right_node.notna()].head(12)
        fig,ax=plt.subplots(figsize=(9,7)); ax.set_title(status.replace("_"," ")+" polygon candidate sample")
        for _,r in q.iterrows():
            a,b=allnodes.get(r.left_node),allnodes.get(r.right_node)
            if not a or not b: continue
            gpd.GeoSeries([a["geometry"]]).boundary.plot(ax=ax,color="#2474a6",linewidth=1)
            gpd.GeoSeries([b["geometry"]]).boundary.plot(ax=ax,color="#e88c30",linewidth=1)
            ax.plot([a["geometry"].centroid.x,b["geometry"].centroid.x],[a["geometry"].centroid.y,b["geometry"].centroid.y],color="gray",alpha=.5)
        ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(out/fname,dpi=180); plt.close(fig)


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir()
    s1obs,s1uni,crs1=read_glo("S1"); s2obs,s2uni,crs2=read_glo("S2")
    tpobs,tpbyyear=read_tp()
    dsu=DSU(); allnodes={}
    # Prefer each source's documented UniqueLakes geometry as the representative
    # node, but retain annual GLO IDs absent from that auxiliary layer as
    # independent real observations (rather than discarding or guessing them).
    for r in s1uni+s2uni: allnodes[r["node"]] = r
    for r in s1obs+s2obs+tpobs: allnodes.setdefault(r["node"], r)
    for n in allnodes: dsu.add(n)
    match_rows=[]
    # Independent GLO sensor identities are linked from polygons, never ID equality.
    match_rows += classify_and_union(candidates(s1uni,s2uni,"glo_s1_unique_to_glo_s2_unique"),dsu)
    years=sorted(tpbyyear)
    for y0,y1 in zip(years,years[1:]):
        # only consecutive source years: no invented intermediate observations.
        if y1==y0+1: match_rows += classify_and_union(candidates(tpbyyear[y0],tpbyyear[y1],f"tp_{y0}_to_tp_{y1}"),dsu)
    # Temporally contemporaneous TP/GLO comparisons, preserving sensor evidence.
    for y in range(2017,2024):
        t=tpbyyear.get(y,[])
        for sensor, obs, uni in [("s1",s1obs,s1uni),("s2",s2obs,s2uni)]:
            right=[r for r in obs if r["year"]==y]
            match_rows += classify_and_union(candidates(t,right,f"tp_{y}_to_{sensor}_{y}"),dsu)
    components=defaultdict(list)
    for n in allnodes: components[dsu.find(n)].append(n)
    # Stable IDs based on each component's sorted member identifiers.
    ordered=sorted(components.values(),key=lambda xs:tuple(sorted(xs)))
    master={n:f"ML{ix:07d}" for ix,xs in enumerate(ordered,1) for n in xs}
    master_members={f"ML{ix:07d}": xs for ix,xs in enumerate(ordered,1)}
    observations=pd.DataFrame(obs_rows(s1obs+s2obs+tpobs,master))
    features=add_features(observations)
    # Representative source polygon (S2 unique preferred, then S1, then most recent TP).
    master_rows=[]; representatives=[]
    for mid, nodes in master_members.items():
        recs=[allnodes[n] for n in nodes]
        pref=sorted(recs,key=lambda r:(0 if r["source_dataset"]=="GLO-HMA-S2" else 1 if r["source_dataset"]=="GLO-HMA-S1" else 2,-(r["year"] or -1)))[0]
        representatives.append(pref)
        ids=defaultdict(list)
        for r in recs: ids[r["source_dataset"]].append(r["source_id"])
        c=pref["geometry"].centroid
        master_rows.append({"master_lake_id":mid,"glo_s1_id":";".join(sorted(set(ids["GLO-HMA-S1"]))) or None,
          "glo_s2_id":";".join(sorted(set(ids["GLO-HMA-S2"]))) or None,"tp_id":";".join(sorted(set(ids["30m-TP"]))) or None,
          "representative_source_dataset":pref["source_dataset"],"representative_original_lake_id":pref["source_id"],
          "representative_geometry_wkt":pref["geometry"].wkt,"geometry_crs":CRS,"centroid_x_m":c.x,"centroid_y_m":c.y,
          "component_node_count":len(nodes),"source_id_note":"30m-TP has no supplied persistent lake ID; tp_id values are year+source-FID references."})
    # Supplied metadata declares UTF-8, but the distributed CSV includes
    # Windows-1252 bytes in textual reference fields; retain those source bytes.
    events=pd.read_csv(ROOT/"GLOF INVENTORY"/"fidelsteiner-HMAGLOFDB-1d975de"/"Database"/"GLOFs"/"HMAGLOFDB.csv",keep_default_na=False,encoding="cp1252")
    glof=match_events(events,representatives,master,match_rows)
    matching=pd.DataFrame(match_rows)
    # Correct mapping of rows collected only as GLO observation aliases: all master nodes exist.
    matching["left_master_lake_id"]=matching.left_node.map(master)
    matching["right_master_lake_id"]=matching.right_node.map(master)
    master_df=pd.DataFrame(master_rows)
    # A documented event ID appears here only when its supplied source-lake
    # coordinate passed the explicit spatial event-link rule; there is no
    # complementary negative-label field.
    event_ids=(glof[glof.event_match_status=="MATCHED"].dropna(subset=["master_lake_id"])
               .groupby("master_lake_id").glof_event_id.agg(lambda s:";".join(sorted(set(s.astype(str))))))
    master_df["glof_event_id"]=master_df.master_lake_id.map(event_ids)
    accepted_masters=set(matching.loc[matching.match_status=="MATCHED","left_master_lake_id"].dropna()) | set(matching.loc[matching.match_status=="MATCHED","right_master_lake_id"].dropna())
    master_df["master_match_evidence_status"]=np.where(master_df.master_lake_id.isin(accepted_masters),"MATCHED","UNMATCHED")
    master_df.to_csv(OUT/"master_lakes.csv",index=False)
    observations.drop(columns="_geometry").to_csv(OUT/"lake_observations.csv",index=False)
    features.to_csv(OUT/"lake_features.csv",index=False)
    glof.to_csv(OUT/"glof_events.csv",index=False)
    matching.to_csv(OUT/"lake_matching.csv",index=False)
    qa_plot(matching,allnodes,OUT)
    # GLOF and time series QA plots.
    matched_g=glof[glof.event_match_status=="MATCHED"].head(12)
    fig,ax=plt.subplots(figsize=(9,7)); ax.set_title("Matched documented GLOF source-location QA sample")
    rep_by_mid={master[r["node"]]:r for r in representatives}
    tr=Transformer.from_crs(WGS84,CRS,always_xy=True)
    for _,e in matched_g.iterrows():
        r=rep_by_mid[e.master_lake_id]; gpd.GeoSeries([r["geometry"]]).boundary.plot(ax=ax,color="#d24b40")
        x,y=tr.transform(e.event_location_lon,e.event_location_lat); ax.plot(x,y,"k*",ms=6)
    ax.set_aspect("equal");fig.tight_layout();fig.savefig(OUT/"qa_glof_lake_matches.png",dpi=180);plt.close(fig)
    sample=features.groupby("master_lake_id").filter(lambda z:len(z)>=6).groupby("master_lake_id").head(1).master_lake_id.unique()[:9]
    fig,ax=plt.subplots(figsize=(10,6));
    for m in sample:
        z=features[features.master_lake_id==m].sort_values("observation_year")
        ax.plot(z.observation_year,z.area_derived_km2,marker="o",ms=2,label=m)
    ax.set(title="Selected real-observation lake area time series",xlabel="Observation year",ylabel="Derived polygon area (km²)")
    if len(sample): ax.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(OUT/"qa_selected_lake_time_series.png",dpi=180);plt.close(fig)
    # Validation report with observed coverage only.
    counts={"unique_GLO_S1_lakes":len(s1uni),"unique_GLO_S2_lakes":len(s2uni),"unique_TP_source_features":len(tpobs),
      "master_lakes":len(master_rows),"matched_lakes_with_accepted_evidence":len(accepted_masters),"unmatched_master_lakes_without_accepted_evidence":int(len(master_rows)-len(accepted_masters)),"matched_candidate_pairs":int((matching.match_status=="MATCHED").sum()),
      "ambiguous_candidate_pairs":int((matching.match_status=="AMBIGUOUS").sum()),"unmatched_candidate_pairs":int((matching.match_status=="UNMATCHED").sum()),
      "documented_GLOF_events":len(glof),"GLOF_events_successfully_matched":int((glof.event_match_status=="MATCHED").sum()),
      "observations":len(observations),"observations_per_master_lake":observations.groupby("master_lake_id").size().describe().to_dict(),
      "temporal_coverage_by_source":observations.groupby(["source_dataset","sensor"],dropna=False).observation_year.agg(["min","max","count"]).reset_index().to_dict("records"),
      "missing_observation_years_are_not_created":True,"glof_negative_labels_created":False,"source_crs":{"GLO_S1":str(crs1),"GLO_S2":str(crs2),"TP":CRS},
      "thresholds":{"search_distance_m":SEARCH_DISTANCE_M,"max_centroid_distance_m":MAX_CENTROID_DISTANCE_M,"min_iou":MIN_IOU,"min_area_ratio":MIN_AREA_RATIO,"score_margin":SCORE_MARGIN,"glof_event_point_distance_m":EVENT_NEAR_DISTANCE_M}}
    (OUT/"validation_report.json").write_text(json.dumps(counts,indent=2,default=str),encoding="utf-8")
    (OUT/"README.md").write_text("# Master HMA Lake Dataset\n\nBuilt only from supplied original vector/event datasets. `lake_matching.csv` retains all evaluated candidates and decision reasons. `glof_events.csv` retains all documented GLOF events; it contains no negative labels. Geometries and derived metrics use ESRI:102025. 30m-TP has no native persistent lake-ID field, so its `tp_id` is a deterministic year+Shapefile FID source reference only.\n",encoding="utf-8")
    print(json.dumps(counts,indent=2,default=str))

if __name__ == "__main__": main()
