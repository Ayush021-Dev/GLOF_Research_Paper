# GLOF Similarity Ranking Model

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

`hist_gradient_boosting`

Selection used proxy average precision first, then proxy ROC-AUC. Metrics are proxy metrics against unlabeled lakes, not true specificity.

## Key metrics

| candidate_model | evaluation_type | test_rows | test_documented_positive_lakes | proxy_roc_auc | proxy_average_precision | precision_at_test_positive_count |
| --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | documented_positive_vs_unlabeled_proxy | 14708 | 16 | 0.9461654301660767 | 0.03947971450384746 | 0.125 |
| balanced_random_forest | documented_positive_vs_unlabeled_proxy | 14708 | 16 | 0.8704928702695344 | 0.029629518893688544 | 0.0625 |
| balanced_logistic_regression | documented_positive_vs_unlabeled_proxy | 14708 | 16 | 0.9714172679008984 | 0.02848958840628784 | 0.0 |
| positive_centroid_similarity | documented_positive_vs_unlabeled_proxy | 14708 | 16 | 0.8779693030220528 | 0.016962776648481143 | 0.0 |

## Curves

- `similarity_ranking_proxy_roc.png`
- `similarity_ranking_proxy_pr.png`

## Outputs

- `model.joblib`
- `glof_similarity_rankings.csv`
- `metrics.csv`
- `metrics.json`
