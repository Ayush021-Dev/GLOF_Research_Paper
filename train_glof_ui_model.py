from pathlib import Path
import json
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, ClassifierMixin


class PositiveCentroidSimilarity(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.positive_centroid_ = X[y == 1].mean(axis=0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        dist = np.linalg.norm(X - self.positive_centroid_, axis=1)
        finite = dist[np.isfinite(dist)]
        scale = np.median(finite) if len(finite) else 1.0
        scale = scale if scale > 0 else 1.0
        p = np.exp(-dist / scale)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "master_lake_dataset"
SPLIT = ROOT / "split_assignments.csv"
OUT = ROOT / "prediction_models_ui"
TEST_OUT = ROOT / "glof_test_dataset"
RANDOM_STATE = 42

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


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_data():
    feature_cols = [
        "observation_id",
        "master_lake_id",
        "source_dataset",
        "sensor",
        "observation_year",
        "centroid_lon",
        "centroid_lat",
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

    features = pd.read_csv(
        MASTER / "lake_features.csv",
        usecols=feature_cols,
        low_memory=False,
    )

    events = pd.read_csv(
        MASTER / "glof_events.csv",
        usecols=[
            "master_lake_id",
            "glof_event_id",
            "event_year_exact",
            "event_year_approx",
            "event_match_status",
        ],
        low_memory=False,
    )

    events = events[
        (events["event_match_status"] == "MATCHED")
        & events["master_lake_id"].notna()
    ].copy()

    events["event_year_exact"] = pd.to_numeric(
        events["event_year_exact"], errors="coerce"
    )
    events["event_year_approx"] = pd.to_numeric(
        events["event_year_approx"], errors="coerce"
    )
    events["event_year"] = events["event_year_exact"].fillna(
        events["event_year_approx"]
    )

    positive_ids = set(events["master_lake_id"].astype(str))

    for c in ["master_lake_id", "source_dataset", "sensor"]:
        features[c] = features[c].astype(str)

    for c in [
        "observation_year",
        "centroid_lon",
        "centroid_lat",
        "area_derived_km2",
        "perimeter_derived_km",
        "length_m",
        "width_m",
        "elongation_ratio",
        "compactness",
        "area_change_1y_km2",
        "area_change_3y_km2",
        "growth_rate_km2_per_year",
    ]:
        features[c] = pd.to_numeric(features[c], errors="coerce")

    features = add_causal_trend(features)

    return features, events, positive_ids

def add_causal_trend(features):
    features = features.sort_values(
        ["master_lake_id", "observation_year"]
    ).copy()

    features["causal_long_term_area_trend_km2_per_year"] = np.nan

    for lake_id, idx in features.groupby("master_lake_id", sort=False).groups.items():
        g = features.loc[idx].sort_values("observation_year")

        yearly = (
            g.groupby("observation_year", as_index=False)["area_derived_km2"]
            .mean()
            .dropna()
        )

        if len(yearly) < 2:
            continue

        x = yearly["observation_year"].to_numpy(dtype=float)
        y = yearly["area_derived_km2"].to_numpy(dtype=float)

        slopes = []

        for i in range(len(yearly)):
            if i < 1:
                slopes.append(np.nan)
                continue

            xx = x[:i + 1]
            yy = y[:i + 1]

            if len(np.unique(xx)) < 2:
                slopes.append(np.nan)
                continue

            slope = np.polyfit(xx, yy, 1)[0]
            slopes.append(slope)

        yearly["causal_long_term_area_trend_km2_per_year"] = slopes

        trend_map = dict(
            zip(
                yearly["observation_year"],
                yearly["causal_long_term_area_trend_km2_per_year"],
            )
        )

        features.loc[
            idx,
            "causal_long_term_area_trend_km2_per_year"
        ] = g["observation_year"].map(trend_map).to_numpy()

    return features

def build_causal_lake_rows(features, events, positive_ids):
    """
    Build causal training rows.

    For documented GLOF lakes:
      - use only observations strictly before the first matched event year
      - keep multiple pre-event observations, not only the final one

    For comparison lakes:
      - keep their latest available observation

    This preserves the no-post-event rule while giving each documented
    GLOF lake multiple pre-event examples when the source data supports it.
    """
    event_year = (
        events.groupby("master_lake_id")["event_year"]
        .min()
        .to_dict()
    )

    rows = []

    for lake_id, group in features.groupby("master_lake_id", sort=False):
        group = group.sort_values("observation_year").copy()

        if lake_id in positive_ids:
            year = event_year.get(lake_id)

            if pd.isna(year):
                continue

            # Strictly causal: nothing at or after the event year.
            group = group[group["observation_year"] < year].copy()

            if group.empty:
                continue

            # Keep all usable pre-event observations.
            for _, row in group.iterrows():
                row = row.copy()
                row["documented_glof"] = 1
                row["prediction_reference_year"] = row["observation_year"]
                rows.append(row)

        else:
            # Unlabeled comparison lake: use latest available observation.
            if group.empty:
                continue

            row = group.iloc[-1].copy()
            row["documented_glof"] = 0
            row["prediction_reference_year"] = row["observation_year"]
            rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No usable lake rows were created.")

    return result

def load_split():
    if not SPLIT.exists():
        raise FileNotFoundError(
            "split_assignments.csv not found. Run generate_split_assignments.py first."
        )

    split = pd.read_csv(SPLIT)
    split["master_lake_id"] = split["master_lake_id"].astype(str)
    split["split"] = split["split"].astype(str).str.lower()

    required = {"master_lake_id", "split"}
    if not required.issubset(split.columns):
        raise ValueError("split_assignments.csv must contain master_lake_id and split.")

    return split[["master_lake_id", "split"]].drop_duplicates("master_lake_id")


def make_test_dataset(data, split):
    """
    Attach the existing spatial split.

    The split is assigned at lake level, so observations from the same lake
    cannot appear in both train and test.

    If the provided spatial split contains no positive GLOF lake in test,
    move one positive lake group from train to test. This is a lake-level
    adjustment, not an observation-level random split.
    """
    data = data.merge(split, on="master_lake_id", how="inner")

    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()

    if train.empty or test.empty:
        raise RuntimeError("Train/test split is empty.")

    train_positive_ids = set(
        train.loc[train["documented_glof"] == 1, "master_lake_id"].astype(str)
    )
    test_positive_ids = set(
        test.loc[test["documented_glof"] == 1, "master_lake_id"].astype(str)
    )

    # Ensure the held-out test set contains at least one documented GLOF lake
    # when the source data has positives available in training.
    if len(test_positive_ids) == 0 and len(train_positive_ids) > 0:
        lake_id = sorted(train_positive_ids)[0]

        moved = train[train["master_lake_id"].astype(str) == lake_id].copy()
        train = train[train["master_lake_id"].astype(str) != lake_id].copy()

        moved["split"] = "test"
        test = pd.concat([test, moved], ignore_index=True)

        print(
            "\nAdjusted spatial holdout: moved documented GLOF lake "
            f"{lake_id} from train to test so test contains a positive lake."
        )

    if train["documented_glof"].nunique() < 2:
        raise RuntimeError(
            "Training split must contain both documented GLOF and comparison lakes."
        )

    if test["documented_glof"].nunique() < 2:
        print(
            "\nWARNING: Test split still does not contain both classes. "
            "Test discrimination metrics will be unavailable."
        )

    TEST_OUT.mkdir(parents=True, exist_ok=True)

    test_export_cols = [
        "master_lake_id",
        "prediction_reference_year",
        "centroid_lon",
        "centroid_lat",
        *FEATURE_COLS,
        "documented_glof",
        "split",
    ]

    test[test_export_cols].to_csv(
        TEST_OUT / "test_lakes.csv", index=False
    )

    test[test["documented_glof"] == 1][test_export_cols].to_csv(
        TEST_OUT / "test_documented_glof_lakes.csv", index=False
    )

    test[test["documented_glof"] == 0][test_export_cols].to_csv(
        TEST_OUT / "test_unlabeled_lakes.csv", index=False
    )

    return train, test

def train(train, test):
    available = [c for c in FEATURE_COLS if c in train.columns]

    missing = [c for c in FEATURE_COLS if c not in train.columns]
    if missing:
        print("Missing optional features:", missing)

    X_train = train[available]
    y_train = train["documented_glof"].astype(int)
    X_test = test[available]
    y_test = test["documented_glof"].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError("Training split does not contain both classes.")

    train_positive_lakes = train.loc[
        train["documented_glof"] == 1, "master_lake_id"
    ].nunique()

    if train_positive_lakes < 2:
        raise RuntimeError(
            "Training data contains fewer than 2 documented GLOF lake groups. "
            "At least two positive lakes are required for leakage-controlled CV."
        )

    models = {
        "positive_centroid_similarity": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", PositiveCentroidSimilarity()),
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    min_class = int(y_train.value_counts().min())
    n_splits = min(5, min_class)

    if n_splits < 2:
        raise RuntimeError("Not enough positive/background lakes for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = []
    fitted = {}

    for name, model in models.items():
        fold_ap = []
        fold_auc = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)

            score = model.predict_proba(X_val)[:, 1]

            if y_val.nunique() == 2:
                fold_auc.append(roc_auc_score(y_val, score))
                fold_ap.append(average_precision_score(y_val, score))

        cv_results.append({
            "model": name,
            "cv_folds": n_splits,
            "cv_mean_roc_auc": float(np.mean(fold_auc)),
            "cv_mean_average_precision": float(np.mean(fold_ap)),
        })

    cv_df = pd.DataFrame(cv_results).sort_values(
        ["cv_mean_average_precision", "cv_mean_roc_auc"],
        ascending=False,
        na_position="last",
    )

    best_name = cv_df.iloc[0]["model"]
    best_model = models[best_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_train, y_train)

    test_score = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "selected_model": best_name,
        "train_lakes": int(len(train)),
        "test_lakes": int(len(test)),
        "train_documented_glof": int(y_train.sum()),
        "train_documented_glof_lakes": int(
            train.loc[train["documented_glof"] == 1, "master_lake_id"].nunique()
        ),
        "test_documented_glof": int(y_test.sum()),
        "test_documented_glof_lakes": int(
            test.loc[test["documented_glof"] == 1, "master_lake_id"].nunique()
        ),
        "test_roc_auc": (
            float(roc_auc_score(y_test, test_score))
            if y_test.nunique() == 2 else None
        ),
        "test_average_precision": (
            float(average_precision_score(y_test, test_score))
            if y_test.nunique() == 2 else None
        ),
    }

    test_output = test.copy()
    test_output["susceptibility_score"] = test_score
    test_output["susceptibility_class"] = pd.cut(
        test_output["susceptibility_score"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Moderate", "High"],
    ).astype(str)

    OUT.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": available,
            "selected_model": best_name,
            "score_type": "documented_GLOF_susceptibility_proxy",
            "target_definition": (
                "1 = lake with a matched documented GLOF event; "
                "0 = unlabeled comparison lake, not a confirmed non-GLOF lake"
            ),
        },
        OUT / "model.joblib",
    )

    cv_df.to_csv(OUT / "cv_metrics.csv", index=False)
    save_json(OUT / "test_metrics.json", test_metrics)

    test_output.to_csv(TEST_OUT / "test_predictions.csv", index=False)

    test_output[
        [
            "master_lake_id",
            "documented_glof",
            "prediction_reference_year",
            "susceptibility_score",
            "susceptibility_class",
        ]
    ].sort_values("susceptibility_score", ascending=False).to_csv(
        TEST_OUT / "ui_test_cases.csv",
        index=False,
    )

    save_json(
        OUT / "model_metadata.json",
        {
            "selected_model": best_name,
            "candidate_models": [
                "positive_centroid_similarity",
                "logistic_regression",
                "random_forest",
                "hist_gradient_boosting",
            ],
            "selection_rule": "mean average precision first, then mean ROC-AUC, using training-only CV",
            "feature_columns": available,
            "cv_folds": n_splits,
            "train_rows": len(train),
            "train_documented_glof": int(y_train.sum()),
            "train_documented_glof_lakes": int(train_positive_lakes),
            "test_rows": len(test),
            "test_documented_glof": int(y_test.sum()),
            "test_documented_glof_lakes": int(
                test.loc[test["documented_glof"] == 1, "master_lake_id"].nunique()
            ),
            "leakage_control": {
                "spatial_holdout": True,
                "test_used_for_model_selection": False,
                "test_used_only_for_final_evaluation": True,
                "latitude_longitude_used_as_model_features": False,
                "observation_year_used_as_model_feature": False,
                "source_counts_used": False,
                "post_event_features_for_positive_lakes": False,
            },
        },
    )

    return cv_df, test_metrics, best_name



def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    if TEST_OUT.exists():
        shutil.rmtree(TEST_OUT)

    features, events, positive_ids = load_data()

    data = build_causal_lake_rows(
        features,
        events,
        positive_ids,
    )

    split = load_split()

    train_data, test_data = make_test_dataset(
        data,
        split,
    )
    print("\nLabel distribution:")
    print(train_data["documented_glof"].value_counts(dropna=False))

    print(
        "\nTrain positive observations:",
        int((train_data["documented_glof"] == 1).sum()),
    )
    print(
        "Train positive lakes:",
        train_data.loc[
            train_data["documented_glof"] == 1, "master_lake_id"
        ].nunique(),
    )
    print(
        "Test positive observations:",
        int((test_data["documented_glof"] == 1).sum()),
    )
    print(
        "Test positive lakes:",
        test_data.loc[
            test_data["documented_glof"] == 1, "master_lake_id"
        ].nunique(),
    )

    results, test_metrics, best_name = train(
        train_data,
        test_data,
    )

    print("\nGLOF UI MODEL TRAINING COMPLETE")
    print("Selected model:", best_name)
    print("\nModel metrics:")
    print(results.to_string(index=False))
    print("\nModel:", OUT / "model.joblib")
    print("Test dataset:", TEST_OUT / "test_lakes.csv")
    print("Test predictions:", TEST_OUT / "test_predictions.csv")
    print("UI test cases:", TEST_OUT / "ui_test_cases.csv")


if __name__ == "__main__":
    main()
