"""
Leakage-controlled GLOF modelling pipeline.

Required files inside ROOT:
    master_lake_dataset/lake_features.csv
    master_lake_dataset/glof_events.csv
    split_assignments.csv

Required columns in split_assignments.csv:
    master_lake_id
    split
Optional:
    cv_fold

Expected split values:
    train / test
or:
    TRAIN / TEST

Recommended split_assignments.csv structure:
    master_lake_id,split,cv_fold
    L001,train,0
    L002,train,1
    ...
    L999,test,

IMPORTANT:
- Do NOT put centroid latitude/longitude, observation year, observation counts,
  source counts, sensor counts, or future-looking aggregate statistics into the
  model features.
- CAUSAL_TREND_COL must be a trend calculated only from observations available
  before the prediction cutoff/event window. Do not replace it with a
  full-history/future-looking trend.
- This script does not create true GLOF=0 labels. Non-event lakes are treated
  as an unlabeled comparison/background population.
- Test data are used exactly once for final evaluation after CV/model selection.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "master_lake_dataset"
OUT = ROOT / "prediction_models_fixed"

RANDOM_STATE = 42
N_CV_SPLITS = 5

# Change ONLY if your actual causal trend column has a different name.
CAUSAL_TREND_COL = "causal_long_term_area_trend_km2_per_year"

# Your split file must already contain a frozen spatial split.
SPLIT_FILE = ROOT / "split_assignments.csv"

# Sentinel-2 is the only sensor/source family used for the model.
SENTINEL_TOKENS = ("sentinel-2", "sentinel2", "sentinel 2", "s2")

# These are allowed model features.
# NO latitude, longitude, year, counts, source IDs, or future-history aggregates.
BASE_FEATURE_COLS = [
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


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def reset_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)


def write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )


def safe_auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def safe_ap(y, score):
    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(average_precision_score(y, score))
    except Exception:
        return np.nan


def precision_at_k(y, score, k):
    y = np.asarray(y)
    score = np.asarray(score)

    if len(y) == 0:
        return np.nan

    k = max(1, min(int(k), len(y)))
    order = np.argsort(score)[::-1][:k]
    return float(np.mean(y[order]))


def plot_curves(y, scores, out_dir, prefix):
    y = np.asarray(y)

    # ROC
    plt.figure(figsize=(8, 5))

    if len(np.unique(y)) >= 2:
        for name, score in scores.items():
            fpr, tpr, _ = roc_curve(y, score)
            auc = roc_auc_score(y, score)
            plt.plot(
                fpr,
                tpr,
                label=f"{name} AUC={auc:.3f}",
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="0.7",
        )

    plt.xlabel("False positive rate")
    plt.ylabel("Positive-event lake recall")
    plt.title("Frozen spatial test ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{prefix}_roc.png",
        dpi=180,
    )
    plt.close()

    # PR
    plt.figure(figsize=(8, 5))

    if len(np.unique(y)) >= 2:
        for name, score in scores.items():
            precision, recall, _ = precision_recall_curve(y, score)
            ap = average_precision_score(y, score)
            plt.plot(
                recall,
                precision,
                label=f"{name} AP={ap:.3f}",
            )

    plt.xlabel("Positive-event lake recall")
    plt.ylabel("Precision")
    plt.title("Frozen spatial test precision-recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{prefix}_pr.png",
        dpi=180,
    )
    plt.close()


# ============================================================
# 3. LOAD AND VALIDATE SPLIT
# ============================================================

def load_split_assignments():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"""
Missing required file:
    {SPLIT_FILE}

Create a frozen spatial split file before running this script.

Required columns:
    master_lake_id
    split

Optional:
    cv_fold
"""
        )

    split = pd.read_csv(SPLIT_FILE, low_memory=False)

    required = {"master_lake_id", "split"}
    missing = required - set(split.columns)

    if missing:
        raise ValueError(
            f"split_assignments.csv is missing columns: {sorted(missing)}"
        )

    split["master_lake_id"] = split["master_lake_id"].astype(str)
    split["split"] = (
        split["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {"train", "test"}

    if not set(split["split"].dropna()).issubset(allowed):
        raise ValueError(
            "split column must contain only 'train' and 'test'."
        )

    if split["master_lake_id"].duplicated().any():
        dupes = (
            split.loc[
                split["master_lake_id"].duplicated(keep=False),
                "master_lake_id",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Each master_lake_id must occur once in split_assignments.csv. "
            f"Duplicates include: {dupes[:10]}"
        )

    train_count = int((split["split"] == "train").sum())
    test_count = int((split["split"] == "test").sum())

    if train_count == 0 or test_count == 0:
        raise ValueError(
            f"Frozen split must contain both train and test lakes. "
            f"Got train={train_count}, test={test_count}."
        )

    return split


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data(split):
    features_path = MASTER / "lake_features.csv"
    events_path = MASTER / "glof_events.csv"

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    if not events_path.exists():
        raise FileNotFoundError(events_path)

    feature_columns = [
        "observation_id",
        "master_lake_id",
        "source_dataset",
        "sensor",
        "observation_year",
        *BASE_FEATURE_COLS,
        CAUSAL_TREND_COL,
    ]

    feature_columns = list(dict.fromkeys(feature_columns))

    # Read the header first so a missing causal column gives a useful error.
    header = pd.read_csv(
        features_path,
        nrows=0,
    ).columns.tolist()

    missing_features = [
        c for c in feature_columns
        if c not in header
    ]

    if missing_features:
        raise ValueError(
            "\nlake_features.csv is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing_features)
            + "\n\nDo not substitute a full-history trend for the causal trend. "
              "Create the causal trend first and then rerun."
        )

    features = pd.read_csv(
        features_path,
        usecols=feature_columns,
        low_memory=False,
    )

    features["master_lake_id"] = (
        features["master_lake_id"]
        .astype(str)
    )

    # Numeric conversion.
    numeric_cols = [
        "observation_year",
        *BASE_FEATURE_COLS,
        CAUSAL_TREND_COL,
    ]

    for col in numeric_cols:
        features[col] = pd.to_numeric(
            features[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Sentinel-2 ONLY
    # --------------------------------------------------------
    sensor_text = (
        features["sensor"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("_", "-", regex=False)
    )

    sentinel_mask = sensor_text.apply(
        lambda value: any(token in value for token in SENTINEL_TOKENS)
    )

    before = len(features)
    features = features.loc[sentinel_mask].copy()
    after = len(features)

    if after == 0:
        raise ValueError(
            "No Sentinel-2 rows were found. "
            "Inspect unique values in lake_features.csv['sensor']."
        )

    # Keep only lakes represented in the frozen split.
    features = features.merge(
        split[["master_lake_id", "split"]],
        on="master_lake_id",
        how="inner",
        validate="many_to_one",
    )

    events = pd.read_csv(
        events_path,
        usecols=[
            "master_lake_id",
            "glof_event_id",
            "event_year_exact",
            "event_year_approx",
            "event_match_status",
        ],
        low_memory=False,
    )

    events["master_lake_id"] = (
        events["master_lake_id"]
        .astype(str)
    )

    events = events[
        (events["event_match_status"] == "MATCHED")
        & events["master_lake_id"].notna()
    ].copy()

    events["event_year_exact_num"] = pd.to_numeric(
        events["event_year_exact"],
        errors="coerce",
    )

    events["event_year_approx_num"] = pd.to_numeric(
        events["event_year_approx"],
        errors="coerce",
    )

    events["event_year_for_model"] = (
        events["event_year_exact_num"]
        .fillna(events["event_year_approx_num"])
    )

    positive_masters = set(
        events["master_lake_id"].dropna().astype(str)
    )

    # Only events whose lakes exist in the modelling dataset.
    positive_masters &= set(
        features["master_lake_id"].unique()
    )

    print(f"Original feature rows: {before:,}")
    print(f"Sentinel-2 feature rows: {after:,}")
    print(f"Model lakes after frozen split: {features['master_lake_id'].nunique():,}")
    print(f"Documented matched event lakes in model data: {len(positive_masters):,}")

    return features, events, positive_masters


# ============================================================
# 5. BUILD CAUSAL LAKE-LEVEL FEATURES
# ============================================================

def build_lake_level(features, positive_masters):
    """
    Build one row per lake.

    Only causal/latest available features are used.

    We deliberately DO NOT create:
        first_observation_year
        last_observation_year
        observation_count
        observed_year_count
        observation_year_span
        source_dataset_count
        sensor_count
        max/min/std over the full history

    Those variables can encode inventory coverage, duration, or future
    information and are therefore excluded.
    """

    rows = []

    for master_id, g in features.groupby(
        "master_lake_id",
        sort=False,
    ):
        g = g.sort_values("observation_year")

        # Latest available Sentinel-2 row.
        # This is used only as a current descriptive state, not as a
        # future-looking full-history statistic.
        latest_year = g["observation_year"].max()
        latest = g[
            g["observation_year"] == latest_year
        ]

        row = {
            "master_lake_id": str(master_id),
            "is_documented_glof_lake": int(
                str(master_id) in positive_masters
            ),
        }

        # Current/latest physical descriptors.
        for col in [
            "area_derived_km2",
            "perimeter_derived_km",
            "length_m",
            "width_m",
            "elongation_ratio",
            "compactness",
        ]:
            row[col] = latest[col].mean()

        # Causal temporal descriptors.
        for col in [
            "area_change_1y_km2",
            "area_change_3y_km2",
            "growth_rate_km2_per_year",
            CAUSAL_TREND_COL,
        ]:
            row[col] = latest[col].mean()

        rows.append(row)

    lake = pd.DataFrame(rows)

    if lake.empty:
        raise ValueError("No lake-level rows were produced.")

    return lake


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

def make_models():
    return {
        "balanced_logistic_regression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "balanced_random_forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "hist_gradient_boosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# 7. CV MODEL SELECTION
# ============================================================

def cross_validate_models(train_df, feature_cols):
    """
    CV is performed ONLY inside the frozen training block.

    The frozen test block is never touched here.
    """

    y = train_df["is_documented_glof_lake"].astype(int).to_numpy()
    X = train_df[feature_cols]

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    if n_pos < N_CV_SPLITS:
        raise ValueError(
            f"Training block has only {n_pos} positive lakes. "
            f"Need at least {N_CV_SPLITS} for {N_CV_SPLITS}-fold stratified CV."
        )

    if n_neg < N_CV_SPLITS:
        raise ValueError(
            f"Training block has only {n_neg} comparison lakes. "
            f"Need at least {N_CV_SPLITS} for {N_CV_SPLITS}-fold stratified CV."
        )

    cv = StratifiedKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for model_name in make_models().keys():
        fold_aucs = []
        fold_aps = []

        for fold, (tr_idx, val_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            model = make_models()[model_name]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    X.iloc[tr_idx],
                    y[tr_idx],
                )

            score = model.predict_proba(
                X.iloc[val_idx]
            )[:, 1]

            auc = safe_auc(
                y[val_idx],
                score,
            )

            ap = safe_ap(
                y[val_idx],
                score,
            )

            fold_aucs.append(auc)
            fold_aps.append(ap)

            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "roc_auc": auc,
                    "average_precision": ap,
                    "validation_rows": len(val_idx),
                    "validation_positives": int(
                        y[val_idx].sum()
                    ),
                }
            )

    cv_df = pd.DataFrame(rows)

    summary = (
        cv_df.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_average_precision=(
                "average_precision",
                "mean",
            ),
            std_average_precision=(
                "average_precision",
                "std",
            ),
        )
        .sort_values(
            [
                "mean_average_precision",
                "mean_roc_auc",
            ],
            ascending=False,
        )
    )

    best_model_name = summary.iloc[0]["model"]

    return cv_df, summary, best_model_name


# ============================================================
# 8. FINAL FROZEN TEST EVALUATION
# ============================================================

def evaluate_on_frozen_test(
    fitted_models,
    test_df,
    feature_cols,
    out,
):
    y_test = (
        test_df["is_documented_glof_lake"]
        .astype(int)
        .to_numpy()
    )

    X_test = test_df[feature_cols]

    if len(np.unique(y_test)) < 2:
        raise ValueError(
            "Frozen test block must contain both documented-positive "
            "and comparison/unlabeled lakes for ROC-AUC."
        )

    scores = {}
    rows = []

    k = max(
        1,
        min(
            int(y_test.sum()),
            len(y_test),
        ),
    )

    for name, model in fitted_models.items():
        score = model.predict_proba(
            X_test
        )[:, 1]

        scores[name] = score

        rows.append(
            {
                "model": name,
                "evaluation_split": "FROZEN_SPATIAL_TEST",
                "test_lakes": len(test_df),
                "test_documented_positive_lakes": int(
                    y_test.sum()
                ),
                "test_comparison_lakes": int(
                    (y_test == 0).sum()
                ),
                "roc_auc": safe_auc(
                    y_test,
                    score,
                ),
                "average_precision": safe_ap(
                    y_test,
                    score,
                ),
                "precision_at_test_positive_count": precision_at_k(
                    y_test,
                    score,
                    k,
                ),
            }
        )

    test_metrics = pd.DataFrame(rows)

    test_metrics.to_csv(
        out / "frozen_test_metrics.csv",
        index=False,
    )

    plot_curves(
        y_test,
        scores,
        out,
        "frozen_test",
    )

    return test_metrics


# ============================================================
# 9. AREA-ONLY BASELINE
# ============================================================

def evaluate_area_baseline(test_df, out):
    """
    Non-ML baseline:
    larger lakes receive higher scores.

    This is intentionally evaluated on the same untouched test block.
    """

    y = (
        test_df["is_documented_glof_lake"]
        .astype(int)
        .to_numpy()
    )

    score = (
        test_df["area_derived_km2"]
        .fillna(
            test_df["area_derived_km2"].median()
        )
        .to_numpy()
    )

    result = pd.DataFrame(
        [
            {
                "model": "area_only_baseline",
                "evaluation_split": "FROZEN_SPATIAL_TEST",
                "test_lakes": len(test_df),
                "test_documented_positive_lakes": int(y.sum()),
                "test_comparison_lakes": int((y == 0).sum()),
                "roc_auc": safe_auc(y, score),
                "average_precision": safe_ap(y, score),
            }
        ]
    )

    result.to_csv(
        out / "area_only_baseline.csv",
        index=False,
    )

    return result


# ============================================================
# 10. FIT FINAL MODEL AND CREATE RANKINGS
# ============================================================

def fit_final_and_rank(
    train_df,
    all_lakes,
    feature_cols,
    best_model_name,
    out,
):
    """
    The final model is trained ONLY on the frozen training lakes.

    Rankings are therefore NOT generated by a model trained on the GLOF
    test lakes.

    For deployment on genuinely unseen future lakes, use model.joblib.
    """

    model = make_models()[best_model_name]

    X_train = train_df[feature_cols]
    y_train = (
        train_df["is_documented_glof_lake"]
        .astype(int)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train,
            y_train,
        )

    all_scores = model.predict_proba(
        all_lakes[feature_cols]
    )[:, 1]

    ranking = all_lakes[
        [
            "master_lake_id",
            "is_documented_glof_lake",
            *feature_cols,
        ]
    ].copy()

    ranking["glof_similarity_score"] = all_scores

    ranking = ranking.sort_values(
        "glof_similarity_score",
        ascending=False,
    )

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1,
    )

    ranking.to_csv(
        out / "glof_similarity_rankings.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "selected_model": best_model_name,
            "training_definition": {
                "split": "frozen spatial train block",
                "sensor": "Sentinel-2 only",
                "lat_lon_used_as_model_features": False,
                "observation_year_used_as_model_feature": False,
                "count_features_used": False,
                "full_history_max_min_std_used": False,
                "causal_trend_column": CAUSAL_TREND_COL,
                "test_lakes_used_for_training": False,
            },
        },
        out / "model.joblib",
    )

    return model, ranking


# ============================================================
# 11. REPORT
# ============================================================

def write_report(
    split,
    features,
    train_df,
    test_df,
    cv_df,
    cv_summary,
    test_metrics,
    area_baseline,
    best_model_name,
    out,
):
    report = []

    report.append("# Leakage-Controlled GLOF Similarity Model")
    report.append("")
    report.append("## Dataset and split")
    report.append("")
    report.append(
        f"- Sentinel-2 feature rows: {len(features):,}"
    )
    report.append(
        f"- Unique modelling lakes: "
        f"{features['master_lake_id'].nunique():,}"
    )
    report.append(
        f"- Frozen training lakes: {len(train_df):,}"
    )
    report.append(
        f"- Frozen test lakes: {len(test_df):,}"
    )
    report.append(
        f"- Training documented-positive lakes: "
        f"{int(train_df['is_documented_glof_lake'].sum()):,}"
    )
    report.append(
        f"- Test documented-positive lakes: "
        f"{int(test_df['is_documented_glof_lake'].sum()):,}"
    )
    report.append("")
    report.append("## Leakage controls")
    report.append("")
    report.append(
        "- Frozen spatial train/test assignment is read from split_assignments.csv."
    )
    report.append(
        "- Latitude and longitude are excluded from model features."
    )
    report.append(
        "- Observation year is excluded from model features."
    )
    report.append(
        "- Observation/source/sensor counts are excluded."
    )
    report.append(
        "- Full-history max/min/std aggregates are excluded."
    )
    report.append(
        "- Sentinel-2 rows are used only."
    )
    report.append(
        f"- Causal trend column: `{CAUSAL_TREND_COL}`."
    )
    report.append(
        "- Model selection occurs inside the training block using cross-validation."
    )
    report.append(
        "- Frozen test metrics are calculated only after model selection."
    )
    report.append(
        "- The final ranking model is trained only on the frozen training block."
    )
    report.append("")
    report.append("## Allowed model features")
    report.append("")
    for col in BASE_FEATURE_COLS + [CAUSAL_TREND_COL]:
        report.append(f"- `{col}`")
    report.append("")
    report.append("## Cross-validation")
    report.append("")
    report.append(cv_summary.to_string(index=False))
    report.append("")
    report.append(
        f"Selected model from training-only CV: `{best_model_name}`"
    )
    report.append("")
    report.append("## Frozen spatial test")
    report.append("")
    report.append(test_metrics.to_string(index=False))
    report.append("")
    report.append("## Area-only baseline")
    report.append("")
    report.append(area_baseline.to_string(index=False))
    report.append("")
    report.append("## Interpretation")
    report.append("")
    report.append(
        "These are documented-positive-versus-unlabeled comparison metrics. "
        "They are not specificity, calibrated GLOF probabilities, or proof that "
        "the model predicts physical GLOF occurrence."
    )
    report.append("")
    report.append(
        "The area-only baseline is included because a model should be interpreted "
        "relative to this simple physical-size ranking."
    )
    report.append("")
    report.append(
        "Do not report the old random-split 0.90-0.95 results as model performance "
        "for this leakage-controlled pipeline."
    )

    (out / "README.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():
    reset_output()

    print("=" * 70)
    print("LEAKAGE-CONTROLLED GLOF MODELING")
    print("=" * 70)

    split = load_split_assignments()

    features, events, positive_masters = load_data(
        split
    )

    lake = build_lake_level(
        features,
        positive_masters,
    )

    lake = lake.merge(
        split[["master_lake_id", "split"]],
        on="master_lake_id",
        how="inner",
        validate="one_to_one",
    )

    feature_cols = BASE_FEATURE_COLS + [
        CAUSAL_TREND_COL
    ]

    missing = [
        c for c in feature_cols
        if c not in lake.columns
    ]

    if missing:
        raise ValueError(
            f"Missing final model features: {missing}"
        )

    # --------------------------------------------------------
    # Frozen train/test split
    # --------------------------------------------------------
    train_df = lake[
        lake["split"] == "train"
    ].copy()

    test_df = lake[
        lake["split"] == "test"
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Frozen split produced an empty train or test block."
        )

    print()
    print("Frozen spatial split:")
    print(
        f"  Train lakes: {len(train_df):,}"
    )
    print(
        f"  Test lakes : {len(test_df):,}"
    )
    print(
        f"  Train positives: "
        f"{int(train_df['is_documented_glof_lake'].sum()):,}"
    )
    print(
        f"  Test positives : "
        f"{int(test_df['is_documented_glof_lake'].sum()):,}"
    )

    # --------------------------------------------------------
    # CV ONLY on training block
    # --------------------------------------------------------
    print()
    print("Running training-only cross-validation...")

    cv_df, cv_summary, best_model_name = (
        cross_validate_models(
            train_df,
            feature_cols,
        )
    )

    cv_df.to_csv(
        OUT / "cross_validation_folds.csv",
        index=False,
    )

    cv_summary.to_csv(
        OUT / "cross_validation_summary.csv",
        index=False,
    )

    print()
    print(cv_summary.to_string(index=False))
    print()
    print(
        f"Selected model: {best_model_name}"
    )

    # --------------------------------------------------------
    # Fit every candidate on TRAIN ONLY
    # --------------------------------------------------------
    fitted_models = {}

    for name in make_models():
        model = make_models()[name]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                train_df[feature_cols],
                train_df["is_documented_glof_lake"].astype(int),
            )

        fitted_models[name] = model

    # --------------------------------------------------------
    # ONE final evaluation on frozen test
    # --------------------------------------------------------
    print()
    print("Evaluating once on untouched frozen test...")

    test_metrics = evaluate_on_frozen_test(
        fitted_models,
        test_df,
        feature_cols,
        OUT,
    )

    area_baseline = evaluate_area_baseline(
        test_df,
        OUT,
    )

    print()
    print(test_metrics.to_string(index=False))
    print()
    print(area_baseline.to_string(index=False))

    # --------------------------------------------------------
    # Final model + ranking
    # --------------------------------------------------------
    final_model, ranking = fit_final_and_rank(
        train_df,
        lake,
        feature_cols,
        best_model_name,
        OUT,
    )

    # --------------------------------------------------------
    # Save the actual split used
    # --------------------------------------------------------
    split.to_csv(
        OUT / "split_assignments_used.csv",
        index=False,
    )

    # Save feature definition.
    write_json(
        OUT / "feature_definition.json",
        {
            "sensor_filter": "Sentinel-2 only",
            "features": feature_cols,
            "excluded": [
                "centroid_lon",
                "centroid_lat",
                "observation_year",
                "observation_count",
                "observed_year_count",
                "observation_year_span",
                "source_dataset_count",
                "sensor_count",
                "full_history_max",
                "full_history_min",
                "full_history_std",
            ],
            "causal_trend_column": CAUSAL_TREND_COL,
        },
    )

    write_report(
        split,
        features,
        train_df,
        test_df,
        cv_df,
        cv_summary,
        test_metrics,
        area_baseline,
        best_model_name,
        OUT,
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Output: {OUT}")
    print()
    print("Important files:")
    print("  cross_validation_folds.csv")
    print("  cross_validation_summary.csv")
    print("  frozen_test_metrics.csv")
    print("  area_only_baseline.csv")
    print("  glof_similarity_rankings.csv")
    print("  model.joblib")
    print("  feature_definition.json")
    print("  README.md")
    print()
    print(
        "The frozen test lakes were NOT used to select or fit the final model."
    )


if __name__ == "__main__":
    main()
