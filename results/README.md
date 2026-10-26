# Selected results

This folder holds the lightweight, shareable outputs used in the report. Raw
datasets, full generated review shortlists, and the large generated 3D viewer
are deliberately not tracked.

Use the numbered figures in this order in the report.

| File | Report purpose |
| --- | --- |
| `images/01_qa_glof_lake_matches.png` | Shows documented GLOF-to-lake matching quality assurance. |
| `images/02_lake_time_series.png` | Shows example lake evolution through time. |
| `images/03_similarity_proxy_roc.png` | Similarity-ranking proxy ROC result. |
| `images/04_similarity_proxy_pr.png` | Similarity-ranking proxy precision-recall result. |
| `images/05_pre_outburst_proxy_roc.png` | Pre-outburst anomaly proxy ROC result. |
| `images/06_pre_outburst_proxy_pr.png` | Pre-outburst anomaly precision-recall result. |
| `images/07_top500_score_space.png` | Top candidate lakes in combined score space. |
| `images/08_combined_score_distribution.png` | Distribution of the combined manual-review score. |
| `images/09_candidate_average_precision.png` | Candidate-model average-precision comparison. |
| `images/10_survival_concordance.png` | Survival-model concordance comparison. |
| `tables/model_metrics.csv` | Compact table of all selected/candidate model metrics. |
| `tables/top_100_review_lakes.csv` | Top 100 lakes for manual review. |
| `tables/lake_3d_metrics.json` | Metrics for the 3D viewer example. |

## Important reporting caveat

Treat ROC, precision-recall, and concordance values as exploratory proxy evidence. The historical event inventory is sparse and unlabeled lakes are not verified non-events. The scores prioritize manual review; they are not operational GLOF probabilities or warnings.
