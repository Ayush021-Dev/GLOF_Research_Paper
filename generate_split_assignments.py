from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "master_lake_dataset/lake_features.csv"
OUTPUT = ROOT / "split_assignments.csv"

RANDOM_STATE = 42
TEST_FRACTION = 0.20
N_CLUSTERS = 10

df = pd.read_csv(INPUT)

required = ["master_lake_id", "centroid_lon", "centroid_lat"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required].copy()
df = df.dropna(subset=["master_lake_id", "centroid_lon", "centroid_lat"])
df = df.drop_duplicates("master_lake_id")

if len(df) < 10:
    raise ValueError("Not enough lakes to create a spatial split.")

n_clusters = min(N_CLUSTERS, len(df))

coords = df[["centroid_lon", "centroid_lat"]].to_numpy()

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=RANDOM_STATE,
    n_init=20
)

df["spatial_group"] = kmeans.fit_predict(coords)

groups = df["spatial_group"].drop_duplicates().to_numpy().copy()

rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(groups)

target_test = max(1, round(len(df) * TEST_FRACTION))

test_groups = []
test_count = 0

for group in groups:
    test_groups.append(group)
    test_count += (df["spatial_group"] == group).sum()
    if test_count >= target_test:
        break

df["split"] = np.where(
    df["spatial_group"].isin(test_groups),
    "test",
    "train"
)

result = df[["master_lake_id", "split"]].copy()

if result["split"].value_counts().get("test", 0) == 0:
    raise ValueError("No test lakes were assigned.")

if result["split"].value_counts().get("train", 0) == 0:
    raise ValueError("No training lakes were assigned.")

result.to_csv(OUTPUT, index=False)

print(f"Created: {OUTPUT}")
print(f"Total lakes : {len(result)}")
print(f"Train lakes : {(result['split'] == 'train').sum()}")
print(f"Test lakes  : {(result['split'] == 'test').sum()}")
print()
print(result["split"].value_counts())
