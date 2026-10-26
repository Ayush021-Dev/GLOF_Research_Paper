"""Create consolidated summaries from trained GLOF model outputs.

This script does not retrain models.  It normalizes existing score outputs,
creates a combined review shortlist, and writes compact comparison figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "prediction_models"
OUT = MODELS / "model_summary"


def percentile_rank_high_good(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def read_metrics() -> pd.DataFrame:
    frames = []
    mapping = {
        "similarity_ranking": MODELS / "01_glof_similarity_ranking" / "metrics.csv",
        "pre_outburst_change_detection": MODELS / "02_pre_outburst_change_detection" / "metrics.csv",
        "time_to_event": MODELS / "03_time_to_event" / "metrics.csv",
    }
    for model_family, path in mapping.items():
        df = pd.read_csv(path)
        df.insert(0, "model_family", model_family)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def create_shortlist() -> pd.DataFrame:
    master = pd.read_csv(
        ROOT / "master_lake_dataset" / "master_lakes.csv",
        usecols=[
            "master_lake_id",
            "glo_s1_id",
            "glo_s2_id",
            "tp_id",
            "glof_event_id",
            "master_match_evidence_status",
            "representative_source_dataset",
            "representative_original_lake_id",
        ],
        low_memory=False,
    )
    sim = pd.read_csv(
        MODELS / "01_glof_similarity_ranking" / "glof_similarity_rankings.csv",
        usecols=[
            "master_lake_id",
            "is_documented_glof_lake",
            "glof_similarity_score",
            "rank",
            "first_observation_year",
            "last_observation_year",
            "observation_count",
            "observed_year_count",
            "latest_area_derived_km2",
            "latest_centroid_lon",
            "latest_centroid_lat",
            "latest_area_change_1y_km2",
            "latest_area_change_3y_km2",
            "latest_growth_rate_km2_per_year",
            "latest_long_term_area_trend_km2_per_year",
        ],
        low_memory=False,
    ).rename(columns={"rank": "similarity_rank"})

    pre = pd.read_csv(
        MODELS / "02_pre_outburst_change_detection" / "pre_outburst_anomaly_scores_by_lake_year.csv",
        usecols=["master_lake_id", "observation_year", "pre_outburst_anomaly_score"],
        low_memory=False,
    )
    pre_best = (
        pre.sort_values("pre_outburst_anomaly_score", ascending=False)
        .groupby("master_lake_id", as_index=False)
        .first()
        .rename(
            columns={
                "observation_year": "max_anomaly_observation_year",
                "pre_outburst_anomaly_score": "max_pre_outburst_anomaly_score",
            }
        )
    )

    surv = pd.read_csv(
        MODELS / "03_time_to_event" / "time_to_event_risk_scores.csv",
        usecols=[
            "master_lake_id",
            "event_observed",
            "duration_years",
            "survival_risk_score",
        ],
        low_memory=False,
    )

    out = (
        sim.merge(master, on="master_lake_id", how="left")
        .merge(pre_best, on="master_lake_id", how="left")
        .merge(surv, on="master_lake_id", how="left")
    )
    out["similarity_percentile"] = percentile_rank_high_good(out["glof_similarity_score"])
    out["pre_outburst_anomaly_percentile"] = percentile_rank_high_good(out["max_pre_outburst_anomaly_score"])
    out["survival_risk_percentile"] = percentile_rank_high_good(out["survival_risk_score"])

    # Similarity is weighted highest because it has the broadest usable positive
    # evidence.  The other two scores are retained as secondary signals because
    # their documented event overlap is small.
    out["combined_review_score"] = (
        0.60 * out["similarity_percentile"].fillna(0)
        + 0.25 * out["pre_outburst_anomaly_percentile"].fillna(0)
        + 0.15 * out["survival_risk_percentile"].fillna(0)
    )
    out["combined_review_rank"] = out["combined_review_score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("combined_review_rank")


def plot_metrics(metrics: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = metrics[metrics["proxy_average_precision"].notna()].copy()
    if not candidates.empty:
        plt.figure(figsize=(9, 5))
        labels = candidates["model_family"] + "\n" + candidates["candidate_model"]
        plt.bar(labels, candidates["proxy_average_precision"])
        plt.ylabel("Proxy average precision")
        plt.xticks(rotation=35, ha="right")
        plt.title("Proxy AP by candidate model")
        plt.tight_layout()
        plt.savefig(OUT / "candidate_proxy_average_precision.png", dpi=180)
        plt.close()

    surv = metrics[metrics["concordance_index"].notna()].copy()
    if not surv.empty:
        plt.figure(figsize=(7, 4))
        plt.bar(surv["candidate_model"], surv["concordance_index"])
        plt.ylabel("Apparent concordance index")
        plt.xticks(rotation=25, ha="right")
        plt.title("Survival candidate comparison")
        plt.tight_layout()
        plt.savefig(OUT / "survival_candidate_concordance.png", dpi=180)
        plt.close()


def plot_shortlist(shortlist: pd.DataFrame) -> None:
    top = shortlist.head(500).copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(
        top["similarity_percentile"],
        top["pre_outburst_anomaly_percentile"],
        c=top["survival_risk_percentile"],
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    plt.colorbar(label="Survival risk percentile")
    plt.xlabel("Similarity percentile")
    plt.ylabel("Pre-outburst anomaly percentile")
    plt.title("Top 500 combined review candidates")
    plt.tight_layout()
    plt.savefig(OUT / "top500_score_space.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    shortlist["combined_review_score"].hist(bins=50)
    plt.xlabel("Combined review score")
    plt.ylabel("Lake count")
    plt.title("Combined score distribution")
    plt.tight_layout()
    plt.savefig(OUT / "combined_score_distribution.png", dpi=180)
    plt.close()


def write_readme(shortlist: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = {
        "similarity_ranking": "hist_gradient_boosting",
        "pre_outburst_change_detection": "one_class_svm",
        "time_to_event": "cox_ph_penalizer_0_1",
    }
    top100 = shortlist.head(100)
    documented_in_top100 = int(top100["is_documented_glof_lake"].sum())
    readme = f"""# Consolidated GLOF Model Summary

This folder summarizes the three separately trained model outputs. It does not retrain models and does not create GLOF negative labels.

## Selected models

- Similarity ranking: `{selected["similarity_ranking"]}`
- Pre-outburst change detection: `{selected["pre_outburst_change_detection"]}`
- Time-to-event: `{selected["time_to_event"]}`

## Combined review score

`combined_review_score = 0.60 * similarity_percentile + 0.25 * pre_outburst_anomaly_percentile + 0.15 * survival_risk_percentile`

Similarity gets the highest weight because it has the broadest matched positive evidence. The pre-outburst and survival signals are included as secondary review cues because only a few documented events have usable pre-event or in-window timing.

## Top-list summary

- Lakes in shortlist table: `{len(shortlist):,}`
- Documented GLOF lakes in top 100 combined ranks: `{documented_in_top100}`
- Highest combined review score: `{shortlist["combined_review_score"].max():.6f}`

## Outputs

- `combined_glof_review_shortlist.csv`
- `top_100_combined_glof_review_lakes.csv`
- `all_candidate_model_metrics.csv`
- `candidate_proxy_average_precision.png`
- `survival_candidate_concordance.png`
- `top500_score_space.png`
- `combined_score_distribution.png`

## Use guidance

Use `glof_similarity_rankings.csv` as the primary operational ranking from the trained models. Use the combined shortlist for manual review prioritization, not as calibrated probability. The event inventory is incomplete and unlabeled lakes are not confirmed non-events.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "selected_models": selected,
        "combined_review_score_formula": {
            "similarity_percentile": 0.60,
            "pre_outburst_anomaly_percentile": 0.25,
            "survival_risk_percentile": 0.15,
        },
        "shortlist_rows": int(len(shortlist)),
        "documented_glof_lakes_in_top_100": documented_in_top100,
        "max_combined_review_score": float(shortlist["combined_review_score"].max()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics()
    shortlist = create_shortlist()
    metrics.to_csv(OUT / "all_candidate_model_metrics.csv", index=False)
    shortlist.to_csv(OUT / "combined_glof_review_shortlist.csv", index=False)
    shortlist.head(100).to_csv(OUT / "top_100_combined_glof_review_lakes.csv", index=False)
    plot_metrics(metrics)
    plot_shortlist(shortlist)
    write_readme(shortlist, metrics)
    print(f"Wrote consolidated summary to {OUT}")


if __name__ == "__main__":
    main()
