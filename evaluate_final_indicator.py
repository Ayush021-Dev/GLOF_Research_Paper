"""Evaluate the final susceptibility indicator on the supplied held-out lake set.

The reported comparison is documented GLOF records versus unlabeled records.
Unlabeled records are not assumed to be confirmed non-events.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from glof_predict import predict_lake


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "prediction_models_ui" / "model.joblib"
TEST_PATH = ROOT / "glof_test_dataset" / "test_lakes.csv"
OUT_DIR = ROOT / "prediction_models_ui" / "evaluation"


def main():
    bundle = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)

    if "documented_glof" not in test.columns:
        raise RuntimeError("test_lakes.csv has no documented_glof column.")

    required = bundle["feature_cols"]
    missing = [col for col in required if col not in test.columns]
    if missing:
        raise RuntimeError(f"Missing indicator features: {missing}")

    predictions = [predict_lake(row, bundle) for _, row in test.iterrows()]
    scored = test.copy()
    scored["susceptibility_score"] = [p["susceptibility_score"] for p in predictions]
    scored["susceptibility_class"] = [p["susceptibility_class"] for p in predictions]
    scored["high_indicator"] = (scored["susceptibility_score"] >= 66.67).astype(int)
    y = scored["documented_glof"].astype(int)
    high = scored["high_indicator"]

    if y.nunique() != 2:
        raise RuntimeError("The held-out set must contain documented and unlabeled rows.")

    tn, fp, fn, tp = confusion_matrix(y, high, labels=[0, 1]).ravel()
    metrics = {
        "evaluation_set": "glof_test_dataset/test_lakes.csv",
        "comparison_definition": (
            "documented GLOF record (1) versus unlabeled record (0); "
            "unlabeled does not mean confirmed non-event"
        ),
        "indicator_bundle": "prediction_models_ui/model.joblib",
        "test_rows": int(len(scored)),
        "documented_glof_rows": int(y.sum()),
        "unlabeled_rows": int((y == 0).sum()),
        "documented_glof_lakes": int(
            scored.loc[y == 1, "master_lake_id"].nunique()
        ),
        "unlabeled_lakes": int(
            scored.loc[y == 0, "master_lake_id"].nunique()
        ),
        "roc_auc_documented_vs_unlabeled": float(
            roc_auc_score(y, scored["susceptibility_score"])
        ),
        "average_precision_documented_vs_unlabeled": float(
            average_precision_score(y, scored["susceptibility_score"])
        ),
        "high_threshold": 66.67,
        "accuracy_at_high_threshold": float(accuracy_score(y, high)),
        "precision_at_high_threshold": float(precision_score(y, high, zero_division=0)),
        "recall_at_high_threshold": float(recall_score(y, high, zero_division=0)),
        "f1_at_high_threshold": float(f1_score(y, high, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "limitations": [
            "This is a retrospective proxy comparison, not a validated forecast.",
            "The indicator reference percentiles were calculated from lake_features.csv, which includes the supplied study corpus.",
            "The documented class contains only the records available in the supplied held-out set.",
            "Unlabeled records are not confirmed non-events, so accuracy and precision must not be interpreted as real-world safety or event probabilities.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_DIR / "heldout_indicator_scores.csv", index=False)
    (OUT_DIR / "heldout_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Scored rows: {OUT_DIR / 'heldout_indicator_scores.csv'}")
    print(f"Metrics: {OUT_DIR / 'heldout_metrics.json'}")


if __name__ == "__main__":
    main()
