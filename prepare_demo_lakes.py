"""Prepare 2024 Sentinel-2 inventory lakes covered by the local DEM."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely import wkt
from shapely.ops import transform

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "demo_lakes"
    output.mkdir(exist_ok=True)
    with rasterio.open(ROOT / "DEM.tif") as dem:
        west, south, east, north = transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)
    features = pd.read_csv(ROOT / "master_lake_dataset/lake_features.csv")
    candidates = features[
        (features.source_dataset == "GLO-HMA-S2")
        & (features.observation_year == 2024)
        & features.centroid_lon.between(west + .03, east - .03)
        & features.centroid_lat.between(south + .03, north - .03)
    ].sort_values("area_derived_km2", ascending=False).drop_duplicates("master_lake_id")
    if candidates.empty:
        raise ValueError("No 2024 Sentinel-2 lakes within the local DEM coverage.")
    needed = set(candidates.observation_id)
    observations = {}
    for chunk in pd.read_csv(ROOT / "master_lake_dataset/lake_observations.csv", chunksize=5000):
        for row in chunk[chunk.observation_id.isin(needed)].to_dict("records"):
            observations[row["observation_id"]] = row
        if needed <= observations.keys():
            break
    manifest = []
    for row in candidates.to_dict("records"):
        obs = observations[row["observation_id"]]
        lon, lat = row["centroid_lon"], row["centroid_lat"]
        crs = f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"
        polygon = transform(Transformer.from_crs(obs["geometry_crs"], crs, always_xy=True).transform,
                            wkt.loads(obs["geometry_wkt"]))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        footprint = transform(Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform,
                              polygon.buffer(500))
        fw, fs, fe, fn = footprint.bounds
        if fw < west or fs < south or fe > east or fn > north:
            continue
        resolution, padding = 10, 2000
        x0, y0, x1, y1 = polygon.bounds
        affine = from_origin(x0 - padding, y1 + padding, resolution, resolution)
        width = math.ceil((x1 - x0 + 2 * padding) / resolution)
        height = math.ceil((y1 - y0 + 2 * padding) / resolution)
        mask = rasterize([(polygon, 1)], out_shape=(height, width), transform=affine, dtype="uint8")
        if not mask.any():
            continue
        mask_path = output / f"{row['master_lake_id']}_2024.tif"
        with rasterio.open(mask_path, "w", driver="GTiff", height=height, width=width, count=1,
                           dtype=np.uint8, crs=crs, transform=affine, compress="deflate") as dst:
            dst.write(mask, 1)
        manifest.append(dict(
            master_lake_id=row["master_lake_id"], observation_id=row["observation_id"],
            original_lake_id=obs["original_lake_id"], source_dataset=obs["source_dataset"],
            observation_year=2024, longitude=lon, latitude=lat,
            inventory_area_km2=row["area_derived_km2"],
            mask_path=mask_path.relative_to(ROOT).as_posix(), dem_path="DEM.tif",
            mask_origin="Rasterized inventory outline; segmentation is not run by this demo.",
        ))
    (output / "catalog.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} dataset lakes in {output}")


if __name__ == "__main__":
    main()
