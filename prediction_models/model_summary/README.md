# Consolidated GLOF Model Summary

This folder summarizes the three separately trained model outputs. It does not retrain models and does not create GLOF negative labels.

## Selected models

- Similarity ranking: `hist_gradient_boosting`
- Pre-outburst change detection: `one_class_svm`
- Time-to-event: `cox_ph_penalizer_0_1`

## Combined review score

`combined_review_score = 0.60 * similarity_percentile + 0.25 * pre_outburst_anomaly_percentile + 0.15 * survival_risk_percentile`

Similarity gets the highest weight because it has the broadest matched positive evidence. The pre-outburst and survival signals are included as secondary review cues because only a few documented events have usable pre-event or in-window timing.

## Top-list summary

- Lakes in shortlist table: `49,024`
- Documented GLOF lakes in top 100 combined ranks: `7`
- Highest combined review score: `0.992383`

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
