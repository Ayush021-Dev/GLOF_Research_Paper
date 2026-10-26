"""Train conservative GLOF outburst modeling artifacts from the master dataset.

The models in this script deliberately do not create true GLOF=0 labels.  They
use documented GLOF matches as positives and treat other lakes as unlabeled,
comparison, background, or censored observations depending on the model family.
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "master_lake_dataset"
OUT = ROOT / "prediction_models"
RANDOM_STATE = 42

FEATURE_COLS = [
    "area_derived_km2",
    "perimeter_derived_km",
    "centroid_lon",
    "centroid_lat",
    "length_m",
    "width_m",
    "elongation_ratio",
    "compactness",
    "area_change_1y_km2",
    "area_change_3y_km2",
    "growth_rate_km2_per_year",
    "long_term_area_trend_km2_per_year",
]


class PositiveCentroidSimilarity(BaseEstimator, ClassifierMixin):
    """Similarity score based on distance to the positive class centroid."""

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        self.positive_centroid_ = x[y == 1].mean(axis=0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        dist = np.linalg.norm(x - self.positive_centroid_, axis=1)
        finite = dist[np.isfinite(dist)]
        scale = np.median(finite) if len(finite) else 1.0
        scale = scale if scale > 0 else 1.0
        p = np.exp(-dist / scale)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])


@dataclass
class DataBundle:
    features: pd.DataFrame
    lake_year: pd.DataFrame
    events: pd.DataFrame
    positive_masters: set[str]


def reset_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_log(lines: list[str]) -> None:
    (OUT / "decision_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer for model reports."""
    if df.empty:
        return "_No rows._"
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else x)
    header = "| " + " | ".join(map(str, clean.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.to_numpy()]
    return "\n".join([header, sep, *rows])


def safe_metric(fn, y, score):
    try:
        if len(np.unique(y)) < 2:
            return None
        return float(fn(y, score))
    except Exception:
        return None


def precision_at_k(y, score, k):
    k = max(1, min(int(k), len(y)))
    order = np.argsort(score)[::-1][:k]
    return float(np.mean(np.asarray(y)[order]))


def plot_binary_curves(y, curves: dict[str, np.ndarray], out_dir: Path, prefix: str) -> None:
    plt.figure(figsize=(8, 5))
    for name, score in curves.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.7", linestyle="--")
    plt.xlabel("False positive rate against unlabeled/comparison lakes")
    plt.ylabel("Positive-event lake recall")
    plt.title("Proxy ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_proxy_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, score in curves.items():
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        plt.plot(recall, precision, label=f"{name} AP={ap:.3f}")
    plt.xlabel("Positive-event lake recall")
    plt.ylabel("Precision against unlabeled/comparison lakes")
    plt.title("Proxy precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_proxy_pr.png", dpi=180)
    plt.close()


def read_data(log: list[str]) -> DataBundle:
    usecols = ["observation_id", "master_lake_id", "source_dataset", "sensor", "observation_year", *FEATURE_COLS]
    features = pd.read_csv(MASTER / "lake_features.csv", usecols=usecols, low_memory=False)
    for col in FEATURE_COLS + ["observation_year"]:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    events = pd.read_csv(
        MASTER / "glof_events.csv",
        usecols=[
            "master_lake_id",
            "glof_event_id",
            "event_year_exact",
            "event_year_approx",
            "event_match_status",
            "event_date_as_provided",
        ],
        low_memory=False,
    )
    events = events[(events["event_match_status"] == "MATCHED") & events["master_lake_id"].notna()].copy()
    events["event_year_exact_num"] = pd.to_numeric(events["event_year_exact"], errors="coerce")
    events["event_year_approx_num"] = pd.to_numeric(events["event_year_approx"], errors="coerce")
    events["event_year_for_model"] = events["event_year_exact_num"].fillna(events["event_year_approx_num"])
    events["event_year_source"] = np.where(
        events["event_year_exact_num"].notna(),
        "exact_source_year",
        np.where(events["event_year_approx_num"].notna(), "approximate_source_year", "missing_numeric_year"),
    )
    positive_masters = set(events["master_lake_id"].dropna().astype(str))

    lake_year = (
        features.groupby(["master_lake_id", "observation_year"], as_index=False)
        .agg({**{c: "mean" for c in FEATURE_COLS}, "observation_id": "count"})
        .rename(columns={"observation_id": "observations_in_year"})
    )

    log.append(f"- Loaded {len(features):,} feature observations.")
    log.append(f"- Loaded {len(lake_year):,} master-lake/year feature rows.")
    log.append(f"- Documented matched GLOF events: {len(events):,}; unique matched event lakes: {len(positive_masters):,}.")
    log.append(
        "- Event-year handling: exact source year is used when numeric; numeric approximate source year is used only when exact year is missing; "
        "non-numeric ranges such as `2001-2003` are retained in source metadata but excluded from time-based training matrices."
    )
    log.append("- No GLOF=0 labels were created. Unmatched or non-event-inventory lakes remain unlabeled/comparison/censored.")
    return DataBundle(features=features, lake_year=lake_year, events=events, positive_masters=positive_masters)


def build_lake_level(features: pd.DataFrame, positive_masters: set[str]) -> pd.DataFrame:
    rows = []
    for master_id, g in features.groupby("master_lake_id", sort=False):
        g = g.sort_values("observation_year")
        latest_year = g["observation_year"].max()
        latest = g[g["observation_year"] == latest_year]
        row = {
            "master_lake_id": master_id,
            "is_documented_glof_lake": int(master_id in positive_masters),
            "first_observation_year": g["observation_year"].min(),
            "last_observation_year": latest_year,
            "observation_count": len(g),
            "observed_year_count": g["observation_year"].nunique(),
            "observation_year_span": g["observation_year"].max() - g["observation_year"].min(),
            "source_dataset_count": g["source_dataset"].nunique(),
            "sensor_count": g["sensor"].nunique(),
        }
        for col in FEATURE_COLS:
            row[f"latest_{col}"] = latest[col].mean()
            row[f"mean_{col}"] = g[col].mean()
            row[f"max_{col}"] = g[col].max()
            row[f"min_{col}"] = g[col].min()
            row[f"std_{col}"] = g[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"master_lake_id", "is_documented_glof_lake"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def train_similarity_ranking(data: DataBundle, log: list[str]) -> None:
    out = OUT / "01_glof_similarity_ranking"
    out.mkdir()
    lake = build_lake_level(data.features, data.positive_masters)
    xcols = model_feature_columns(lake)
    y = lake["is_documented_glof_lake"].astype(int).to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(lake)),
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    x_train, y_train = lake.iloc[train_idx][xcols], y[train_idx]
    x_test, y_test = lake.iloc[test_idx][xcols], y[test_idx]

    models = {
        "positive_centroid_similarity": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", PositiveCentroidSimilarity()),
            ]
        ),
        "balanced_logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "balanced_random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
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
                ("impute", SimpleImputer(strategy="median")),
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

    metrics = []
    curves = {}
    fitted = {}
    k = max(1, int(y_test.sum()))
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        curves[name] = score
        metrics.append(
            {
                "candidate_model": name,
                "evaluation_type": "documented_positive_vs_unlabeled_proxy",
                "test_rows": int(len(y_test)),
                "test_documented_positive_lakes": int(y_test.sum()),
                "proxy_roc_auc": safe_metric(roc_auc_score, y_test, score),
                "proxy_average_precision": safe_metric(average_precision_score, y_test, score),
                "precision_at_test_positive_count": precision_at_k(y_test, score, k),
            }
        )
        fitted[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["proxy_average_precision", "proxy_roc_auc"], ascending=False, na_position="last"
    )
    best_name = metrics_df.iloc[0]["candidate_model"]
    best_model = models[best_name]
    best_model.fit(lake[xcols], y)
    lake["glof_similarity_score"] = best_model.predict_proba(lake[xcols])[:, 1]
    lake["rank"] = lake["glof_similarity_score"].rank(ascending=False, method="first").astype(int)
    lake.sort_values("rank").to_csv(out / "glof_similarity_rankings.csv", index=False)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    write_json(out / "metrics.json", metrics)
    joblib.dump({"model": best_model, "feature_columns": xcols, "selected_model": best_name}, out / "model.joblib")
    plot_binary_curves(y_test, curves, out, "similarity_ranking")

    readme = f"""# GLOF Similarity Ranking Model

## Purpose

Ranks lakes by feature similarity to lakes with documented, spatially matched GLOF events.

## Why this model family

This was chosen before binary GLOF classification because the dataset contains documented positives but does not contain proven negatives. Other lakes are unlabeled, not non-events. The selected artifact is therefore a susceptibility ranking model, not a calibrated outburst probability model.

## Candidate models compared

- Positive centroid similarity: transparent baseline measuring distance to the positive-lake feature centroid.
- Balanced logistic regression: interpretable linear positive-vs-unlabeled proxy model.
- Balanced random forest: nonlinear tabular model with class balancing.
- Histogram gradient boosting: nonlinear boosting baseline.

## Selected model

`{best_name}`

Selection used proxy average precision first, then proxy ROC-AUC. Metrics are proxy metrics against unlabeled lakes, not true specificity.

## Key metrics

{markdown_table(metrics_df)}

## Curves

- `similarity_ranking_proxy_roc.png`
- `similarity_ranking_proxy_pr.png`

## Outputs

- `model.joblib`
- `glof_similarity_rankings.csv`
- `metrics.csv`
- `metrics.json`
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    best_metrics = metrics_df.iloc[0].to_dict()
    log.append(
        f"- Similarity ranking target: {int(y.sum()):,} documented-positive master lakes and {int((y == 0).sum()):,} unlabeled comparison lakes."
    )
    log.append(f"- Similarity ranking features: {len(xcols)} lake-level aggregate temporal/shape/area columns.")
    log.append(
        f"- Similarity ranking trained. Selected `{best_name}` from {len(models)} candidates "
        f"(proxy AP={best_metrics.get('proxy_average_precision')}, proxy ROC-AUC={best_metrics.get('proxy_roc_auc')})."
    )


def train_pre_outburst(data: DataBundle, log: list[str]) -> None:
    out = OUT / "02_pre_outburst_change_detection"
    out.mkdir()
    events = data.events[data.events["event_year_for_model"].notna()].copy()
    positives = []
    for _, e in events.iterrows():
        rows = data.lake_year[
            (data.lake_year["master_lake_id"] == e["master_lake_id"])
            & (data.lake_year["observation_year"] < e["event_year_for_model"])
        ]
        if not rows.empty:
            r = rows.sort_values("observation_year").iloc[-1].to_dict()
            r["glof_event_id"] = e["glof_event_id"]
            r["event_year_for_model"] = e["event_year_for_model"]
            positives.append(r)
    pos = pd.DataFrame(positives)
    bg = data.lake_year[~data.lake_year["master_lake_id"].isin(data.positive_masters)].copy()
    xcols = [c for c in FEATURE_COLS if c in bg.columns] + ["observations_in_year", "observation_year"]

    bg_sample = bg.sample(n=min(len(bg), max(5000, len(pos) * 200)), random_state=RANDOM_STATE)
    eval_df = pd.concat([pos.assign(proxy_label=1), bg_sample.assign(proxy_label=0)], ignore_index=True)
    y_eval = eval_df["proxy_label"].astype(int).to_numpy()

    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    x_bg = prep.fit_transform(bg[xcols])
    x_eval = prep.transform(eval_df[xcols])

    contamination = min(0.05, max(0.005, (len(pos) or 1) / max(len(bg_sample), 1)))
    models = {
        "isolation_forest": IsolationForest(
            n_estimators=500,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "one_class_svm": OneClassSVM(nu=contamination, kernel="rbf", gamma="scale"),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        ),
    }

    metrics = []
    curves = {}
    fitted = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_bg)
        # Higher score should mean more anomalous/pre-outburst-like.
        score = -model.decision_function(x_eval)
        curves[name] = score
        metrics.append(
            {
                "candidate_model": name,
                "evaluation_type": "pre_event_documented_positives_vs_unlabeled_background_proxy",
                "pre_event_positive_rows": int(len(pos)),
                "background_rows_for_eval": int(len(bg_sample)),
                "proxy_roc_auc": safe_metric(roc_auc_score, y_eval, score),
                "proxy_average_precision": safe_metric(average_precision_score, y_eval, score),
                "precision_at_positive_count": precision_at_k(y_eval, score, max(1, len(pos))),
            }
        )
        fitted[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["proxy_average_precision", "proxy_roc_auc"], ascending=False, na_position="last"
    )
    best_name = metrics_df.iloc[0]["candidate_model"]
    best_model = fitted[best_name]
    all_x = prep.transform(data.lake_year[xcols])
    data.lake_year.copy().assign(pre_outburst_anomaly_score=-best_model.decision_function(all_x)).sort_values(
        "pre_outburst_anomaly_score", ascending=False
    ).to_csv(out / "pre_outburst_anomaly_scores_by_lake_year.csv", index=False)
    if not pos.empty:
        pos.to_csv(out / "documented_pre_event_training_cases.csv", index=False)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    write_json(out / "metrics.json", metrics)
    joblib.dump(
        {"preprocessor": prep, "model": best_model, "feature_columns": xcols, "selected_model": best_name},
        out / "model.joblib",
    )
    plot_binary_curves(y_eval, curves, out, "pre_outburst")

    caveat = (
        "Only "
        f"{len(pos)} documented matched events have source polygon observations before the event year. "
        "This artifact is therefore a feasibility baseline and anomaly-ranker, not a validated warning model."
    )
    readme = f"""# Pre-Outburst Change Detection Model

## Purpose

Detects lake-year records whose measured area/shape/change features look anomalous relative to the unlabeled background lake-year population.

## Why this model family

This was chosen because very few documented GLOF events have pre-event polygon observations in the supplied temporal record. A supervised classifier would overfit. Unsupervised anomaly detection can still produce a defensible diagnostic ranking while keeping labels conservative.

## Critical limitation

{caveat}

## Candidate models compared

- Isolation forest: robust tabular anomaly detector and selected default for sparse positive validation.
- One-class SVM: kernel boundary baseline.
- Local outlier factor with novelty mode: local-density anomaly baseline.

## Selected model

`{best_name}`

## Key metrics

{markdown_table(metrics_df)}

Metrics are proxy metrics using the few documented pre-event rows against unlabeled background rows. They should not be read as operational GLOF prediction accuracy.

## Curves

- `pre_outburst_proxy_roc.png`
- `pre_outburst_proxy_pr.png`

## Outputs

- `model.joblib`
- `pre_outburst_anomaly_scores_by_lake_year.csv`
- `documented_pre_event_training_cases.csv`
- `metrics.csv`
- `metrics.json`
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    best_metrics = metrics_df.iloc[0].to_dict()
    log.append(
        "- Pre-outburst target construction: for each documented event with a numeric source year, selected the latest real lake-year observation strictly before that event year."
    )
    log.append(
        f"- Pre-outburst anomaly model trained with {len(pos)} documented pre-event cases and unlabeled background lake-year rows; "
        f"selected `{best_name}` (proxy AP={best_metrics.get('proxy_average_precision')}, proxy ROC-AUC={best_metrics.get('proxy_roc_auc')})."
    )


def train_survival(data: DataBundle, log: list[str]) -> None:
    out = OUT / "03_time_to_event"
    out.mkdir()
    span = data.lake_year.groupby("master_lake_id")["observation_year"].agg(["min", "max", "count"]).reset_index()
    event_year = (
        data.events[data.events["event_year_for_model"].notna()]
        .groupby("master_lake_id")["event_year_for_model"]
        .min()
        .rename("event_year_for_model")
        .reset_index()
    )
    surv = span.merge(event_year, on="master_lake_id", how="left")
    surv["event_observed"] = (
        surv["event_year_for_model"].notna()
        & (surv["event_year_for_model"] >= surv["min"])
        & (surv["event_year_for_model"] <= surv["max"])
    ).astype(int)
    surv["duration_years"] = np.where(
        surv["event_observed"].eq(1),
        np.maximum(0.5, surv["event_year_for_model"] - surv["min"]),
        np.maximum(0.5, surv["max"] - surv["min"] + 1.0),
    )

    first_rows = (
        data.lake_year.sort_values(["master_lake_id", "observation_year"])
        .groupby("master_lake_id", as_index=False)
        .first()
    )
    base_cols = [
        "area_derived_km2",
        "perimeter_derived_km",
        "centroid_lon",
        "centroid_lat",
        "length_m",
        "width_m",
        "elongation_ratio",
        "compactness",
    ]
    surv = surv.merge(first_rows[["master_lake_id", *base_cols]], on="master_lake_id", how="left")
    xcols = base_cols + ["count"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(surv[xcols])
    x_scaled = scaler.fit_transform(x_imp)
    xnames = [f"x_{i}_{c}" for i, c in enumerate(xcols)]
    fit_df = pd.DataFrame(x_scaled, columns=xnames)
    fit_df["duration_years"] = surv["duration_years"].to_numpy()
    fit_df["event_observed"] = surv["event_observed"].to_numpy()
    fit_df["master_lake_id"] = surv["master_lake_id"].to_numpy()

    candidate_defs = {
        "cox_ph_penalizer_0_1": ("cox", 0.1),
        "cox_ph_penalizer_1_0": ("cox", 1.0),
        "weibull_aft_penalizer_0_1": ("weibull", 0.1),
    }
    metrics = []
    fitted = {}
    model_df = fit_df.drop(columns=["master_lake_id"]).copy()
    for name, (kind, penalizer) in candidate_defs.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "cox":
                    model = CoxPHFitter(penalizer=penalizer)
                    model.fit(model_df, duration_col="duration_years", event_col="event_observed")
                    risk = model.predict_partial_hazard(model_df).to_numpy().reshape(-1)
                else:
                    model = WeibullAFTFitter(penalizer=penalizer)
                    model.fit(model_df, duration_col="duration_years", event_col="event_observed")
                    risk = -model.predict_median(model_df).to_numpy().reshape(-1)
            cindex = float(concordance_index(surv["duration_years"], -risk, surv["event_observed"]))
            metrics.append(
                {
                    "candidate_model": name,
                    "evaluation_type": "apparent_concordance_due_low_event_count",
                    "rows": int(len(surv)),
                    "observed_events_inside_observation_window": int(surv["event_observed"].sum()),
                    "concordance_index": cindex,
                    "fit_status": "fit",
                }
            )
            fitted[name] = (model, kind, risk)
        except Exception as exc:
            metrics.append(
                {
                    "candidate_model": name,
                    "evaluation_type": "apparent_concordance_due_low_event_count",
                    "rows": int(len(surv)),
                    "observed_events_inside_observation_window": int(surv["event_observed"].sum()),
                    "concordance_index": None,
                    "fit_status": f"failed: {exc}",
                }
            )

    metrics_df = pd.DataFrame(metrics).sort_values("concordance_index", ascending=False, na_position="last")
    best_name = metrics_df[metrics_df["fit_status"].eq("fit")].iloc[0]["candidate_model"]
    best_model, best_kind, risk = fitted[best_name]
    pred = surv[["master_lake_id", "min", "max", "count", "event_year_for_model", "event_observed", "duration_years"]].copy()
    pred["survival_risk_score"] = risk
    pred.sort_values("survival_risk_score", ascending=False).to_csv(out / "time_to_event_risk_scores.csv", index=False)
    fit_df.to_csv(out / "survival_training_matrix.csv", index=False)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    write_json(out / "metrics.json", metrics)
    joblib.dump(
        {
            "model": best_model,
            "model_kind": best_kind,
            "feature_columns": xcols,
            "scaled_feature_columns": xnames,
            "imputer": imputer,
            "scaler": scaler,
            "selected_model": best_name,
        },
        out / "model.joblib",
    )

    plt.figure(figsize=(8, 5))
    if best_kind == "cox":
        representative = pred.sort_values("survival_risk_score")
        sample_ids = [
            representative.iloc[0]["master_lake_id"],
            representative.iloc[len(representative) // 2]["master_lake_id"],
            representative.iloc[-1]["master_lake_id"],
        ]
        for label, mid in zip(["low risk", "median risk", "high risk"], sample_ids):
            row = fit_df[fit_df["master_lake_id"] == mid].drop(columns=["master_lake_id", "duration_years", "event_observed"])
            sf = best_model.predict_survival_function(row)
            plt.plot(sf.index, sf.iloc[:, 0], label=f"{label} {mid}")
    else:
        representative = pred.sort_values("survival_risk_score")
        sample_ids = [
            representative.iloc[0]["master_lake_id"],
            representative.iloc[len(representative) // 2]["master_lake_id"],
            representative.iloc[-1]["master_lake_id"],
        ]
        for label, mid in zip(["low risk", "median risk", "high risk"], sample_ids):
            row = fit_df[fit_df["master_lake_id"] == mid].drop(columns=["master_lake_id", "duration_years", "event_observed"])
            sf = best_model.predict_survival_function(row)
            plt.plot(sf.index, sf.iloc[:, 0], label=f"{label} {mid}")
    plt.xlabel("Years since first observation")
    plt.ylabel("Estimated survival probability")
    plt.title("Representative survival curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "survival_curves.png", dpi=180)
    plt.close()

    caveat = (
        f"Only {int(surv['event_observed'].sum())} documented events occur inside an observed lake time span. "
        "The model is trained as a structural survival baseline; the apparent concordance is not a stable validation estimate."
    )
    readme = f"""# Time-to-Event GLOF Model

## Purpose

Estimates relative event hazard over observed lake lifetimes using source event years and first-observation lake geometry features.

## Why this model family

Time-to-event modeling is more appropriate than simple binary classification when lakes are observed for unequal durations and many lakes have no documented event by the end of the record. Those lakes are treated as censored, not as permanent non-events.

## Critical limitation

{caveat}

## Candidate models compared

- Cox proportional hazards with penalizer 0.1.
- Cox proportional hazards with penalizer 1.0.
- Weibull accelerated failure time with penalizer 0.1.

## Selected model

`{best_name}`

## Key metrics

{markdown_table(metrics_df)}

## Curves

- `survival_curves.png`

## Outputs

- `model.joblib`
- `time_to_event_risk_scores.csv`
- `survival_training_matrix.csv`
- `metrics.csv`
- `metrics.json`
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    best_metrics = metrics_df[metrics_df["fit_status"].eq("fit")].iloc[0].to_dict()
    log.append(
        "- Time-to-event construction: event lakes are events only when the source event year falls inside that lake's observed year span; all other lake histories are right-censored at last observation."
    )
    log.append("- Time-to-event predictors use first-observation lake geometry/area features to avoid using post-event temporal change as a baseline survival predictor.")
    log.append(
        f"- Time-to-event model trained with {int(surv['event_observed'].sum())} observed in-window event lakes; "
        f"selected `{best_name}` (apparent concordance={best_metrics.get('concordance_index')})."
    )


def write_root_readme(log: list[str]) -> None:
    readme = """# GLOF Prediction Models

Three separate model artifacts were trained from the master lake dataset:

1. `01_glof_similarity_ranking`: ranks lakes by similarity to documented GLOF lakes.
2. `02_pre_outburst_change_detection`: anomaly-ranks lake-year records for pre-outburst-like change behavior.
3. `03_time_to_event`: survival/time-to-event baseline using censored lake histories.

No model in this folder uses synthetic imagery, generated masks, segmentation output, fabricated observations, inferred GLOF events, or blanket GLOF=0 labels.

Read each subfolder `README.md` before using scores. The first model is the most usable immediately. The second and third are constrained by the small number of matched documented events with usable pre-event or in-window timing.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    write_log(log)


def main() -> None:
    reset_output()
    log = [
        "# GLOF Model Decision Log",
        "",
        "- Input folder: `master_lake_dataset`.",
        "- Output folder: `prediction_models`.",
        "- Modeling order followed: similarity ranking, pre-outburst change detection, time-to-event.",
    ]
    data = read_data(log)
    train_similarity_ranking(data, log)
    train_pre_outburst(data, log)
    train_survival(data, log)
    write_root_readme(log)
    print(f"Wrote model artifacts to {OUT}")


if __name__ == "__main__":
    main()
