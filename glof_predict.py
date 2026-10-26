from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("prediction_models_ui/model.joblib")


def normalize(value, q05, q95):
    if pd.isna(value):
        return 0.5

    if q95 <= q05:
        return 0.5

    x = (float(value) - q05) / (q95 - q05)
    return float(np.clip(x, 0.0, 1.0))


def predict_lake(row, bundle):
    feature_cols = bundle["feature_cols"]
    stats = bundle["stats"]
    weights = bundle["weights"]

    values = {}

    for col in feature_cols:
        if col not in row:
            raise RuntimeError(f"Missing prediction feature: {col}")

        values[col] = pd.to_numeric(row[col], errors="coerce")

    component_scores = {}

    for col in feature_cols:
        score = normalize(
            values[col],
            stats[col]["q05"],
            stats[col]["q95"],
        )

        if col == "compactness":
            score = 1.0 - score

        component_scores[col] = score

    raw = sum(
        component_scores[col] * weights[col]
        for col in weights
    )

    score = float(np.clip(raw * 100.0, 0.0, 100.0))

    if score < 33.33:
        level = "LOW"
    elif score < 66.67:
        level = "MODERATE"
    else:
        level = "HIGH"

    contributions = {
        col: round(component_scores[col] * weights[col] * 100, 2)
        for col in weights
    }

    ranked = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    factor_names = {
        "area_derived_km2": "Lake area",
        "perimeter_derived_km": "Lake perimeter",
        "length_m": "Lake length",
        "width_m": "Lake width",
        "elongation_ratio": "Elongation",
        "compactness": "Compactness",
        "area_change_1y_km2": "1-year area change",
        "area_change_3y_km2": "3-year area change",
        "growth_rate_km2_per_year": "Growth rate",
        "causal_long_term_area_trend_km2_per_year": "Long-term area trend",
    }

    factors = [
        {
            "feature": col,
            "name": factor_names.get(col, col),
            "contribution": value,
        }
        for col, value in ranked[:5]
    ]

    return {
        "susceptibility_score": round(score, 2),
        "susceptibility_class": level,
        "risk_factors": factors,
        "model_type": "GLOF susceptibility indicator",
        "validated_supervised_probability": False,
        "interpretation": "Indicator derived from available spatial and temporal lake characteristics; not a calibrated probability of GLOF occurrence.",
    }


def main():
    bundle = joblib.load(MODEL_PATH)

    row = {
        "area_derived_km2": 1.2,
        "perimeter_derived_km": 5.1,
        "length_m": 2100,
        "width_m": 900,
        "elongation_ratio": 2.33,
        "compactness": 0.62,
        "area_change_1y_km2": 0.12,
        "area_change_3y_km2": 0.31,
        "growth_rate_km2_per_year": 0.10,
        "causal_long_term_area_trend_km2_per_year": 0.08,
    }

    result = predict_lake(row, bundle)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
