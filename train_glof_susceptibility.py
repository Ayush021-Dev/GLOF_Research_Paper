from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(".")
MASTER = ROOT / "master_lake_dataset"
OUT = ROOT / "prediction_models_ui"

FEATURE_COLS = [
    "area_derived_km2",
    "perimeter_derived_km",
    "length_m",
    "width_m",
    "elongation_ratio",
    "compactness",
    "area_change_1y_km2",
    "area_change_3y_km2",
    "growth_rate_km2_per_year",
    "causal_long_term_area_trend_km2_per_year",
]

RISK_HIGHER = [
    "area_derived_km2",
    "perimeter_derived_km",
    "length_m",
    "width_m",
    "elongation_ratio",
    "area_change_1y_km2",
    "area_change_3y_km2",
    "growth_rate_km2_per_year",
    "causal_long_term_area_trend_km2_per_year",
]


def add_causal_trend(df):
    """
    Calculate the long-term area trend using only observations available
    up to the current observation year.

    This prevents future observations from influencing an earlier row.
    """
    df = df.copy()

    df["causal_long_term_area_trend_km2_per_year"] = np.nan

    for lake_id, group in df.groupby("master_lake_id", sort=False):
        group = group.sort_values("observation_year")

        years = pd.to_numeric(group["observation_year"], errors="coerce")
        areas = pd.to_numeric(group["area_derived_km2"], errors="coerce")

        valid = years.notna() & areas.notna()

        valid_idx = group.index[valid]

        for idx in valid_idx:
            current_year = years.loc[idx]

            history_mask = (
                valid
                & (years <= current_year)
            )

            history_years = years.loc[group.index[history_mask]]
            history_areas = areas.loc[group.index[history_mask]]

            if len(history_years) >= 2 and history_years.nunique() >= 2:
                x = history_years.to_numpy(dtype=float)
                y = history_areas.to_numpy(dtype=float)

                try:
                    slope = np.polyfit(x, y, 1)[0]
                    df.loc[idx, "causal_long_term_area_trend_km2_per_year"] = slope
                except Exception:
                    pass
            else:
                df.loc[idx, "causal_long_term_area_trend_km2_per_year"] = 0.0

    return df


def load_data():
    path = MASTER / "lake_features.csv"

    if not path.exists():
        raise RuntimeError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_base = [
        "master_lake_id",
        "observation_year",
        "area_derived_km2",
        "perimeter_derived_km",
        "length_m",
        "width_m",
        "elongation_ratio",
        "compactness",
        "area_change_1y_km2",
        "area_change_3y_km2",
        "growth_rate_km2_per_year",
    ]

    for col in required_base:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    for col in required_base:
        if col != "master_lake_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["master_lake_id"]).copy()
    df = df.sort_values(
        ["master_lake_id", "observation_year"]
    )

    # Create the missing causal trend feature.
    df = add_causal_trend(df)

    return df


def make_reference_stats(df):
    stats = {}

    for col in FEATURE_COLS:
        values = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if values.empty:
            stats[col] = {
                "q05": 0.0,
                "q95": 1.0,
                "median": 0.0,
            }
        else:
            stats[col] = {
                "q05": float(values.quantile(0.05)),
                "q95": float(values.quantile(0.95)),
                "median": float(values.median()),
            }

    return stats


def main():
    df = load_data()

    stats = make_reference_stats(df)

    OUT.mkdir(parents=True, exist_ok=True)

    weights = {
        "area_derived_km2": 0.08,
        "perimeter_derived_km": 0.06,
        "length_m": 0.05,
        "width_m": 0.05,
        "elongation_ratio": 0.08,
        "compactness": 0.08,
        "area_change_1y_km2": 0.14,
        "area_change_3y_km2": 0.14,
        "growth_rate_km2_per_year": 0.14,
        "causal_long_term_area_trend_km2_per_year": 0.18,
    }

    bundle = {
        "feature_cols": FEATURE_COLS,
        "stats": stats,
        "weights": weights,
        "version": "glof_susceptibility_indicator_v1",
        "target_type": "indicator",
        "validated_supervised_probability": False,
    }

    joblib.dump(
        bundle,
        OUT / "model.joblib"
    )

    metadata = {
        "name": "GLOF Susceptibility Indicator",
        "version": "v1",
        "target_type": "susceptibility_indicator",
        "validated_supervised_probability": False,
        "classes": ["LOW", "MODERATE", "HIGH"],
        "score_range": [0, 100],
        "normalization": "5th-95th percentile of lake_features.csv",
        "rows": int(len(df)),
        "unique_lakes": int(df["master_lake_id"].nunique()),
        "features": FEATURE_COLS,
        "note": (
            "Indicator derived from available spatial and temporal "
            "lake characteristics. It is not a calibrated probability "
            "of GLOF occurrence."
        ),
    }

    with open(
        OUT / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    # Save the engineered dataset so it can be inspected/debugged.
    df.to_csv(
        OUT / "engineered_lake_features.csv",
        index=False
    )

    print()
    print("GLOF SUSCEPTIBILITY INDICATOR CREATED")
    print("-------------------------------------")
    print(f"Rows        : {len(df)}")
    print(f"Unique lakes: {df['master_lake_id'].nunique()}")
    print()
    print("Created:")
    print(f"  {OUT / 'model.joblib'}")
    print(f"  {OUT / 'model_metadata.json'}")
    print(f"  {OUT / 'engineered_lake_features.csv'}")
    print()
    print(
        "Causal trend feature successfully calculated:"
    )
    print(
        "  causal_long_term_area_trend_km2_per_year"
    )


if __name__ == "__main__":
    main()
