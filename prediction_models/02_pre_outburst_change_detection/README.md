# Pre-Outburst Change Detection Model

## Purpose

Detects lake-year records whose measured area/shape/change features look anomalous relative to the unlabeled background lake-year population.

## Why this model family

This was chosen because very few documented GLOF events have pre-event polygon observations in the supplied temporal record. A supervised classifier would overfit. Unsupervised anomaly detection can still produce a defensible diagnostic ranking while keeping labels conservative.

## Critical limitation

Only 3 documented matched events have source polygon observations before the event year. This artifact is therefore a feasibility baseline and anomaly-ranker, not a validated warning model.

## Candidate models compared

- Isolation forest: robust tabular anomaly detector and selected default for sparse positive validation.
- One-class SVM: kernel boundary baseline.
- Local outlier factor with novelty mode: local-density anomaly baseline.

## Selected model

`one_class_svm`

## Key metrics

| candidate_model | evaluation_type | pre_event_positive_rows | background_rows_for_eval | proxy_roc_auc | proxy_average_precision | precision_at_positive_count |
| --- | --- | --- | --- | --- | --- | --- |
| one_class_svm | pre_event_documented_positives_vs_unlabeled_background_proxy | 3 | 5000 | 0.8120666666666667 | 0.0027528354394195436 | 0.0 |
| isolation_forest | pre_event_documented_positives_vs_unlabeled_background_proxy | 3 | 5000 | 0.7338 | 0.0017829531799406328 | 0.0 |
| local_outlier_factor | pre_event_documented_positives_vs_unlabeled_background_proxy | 3 | 5000 | 0.5129999999999999 | 0.0010852661468360005 | 0.0 |

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
