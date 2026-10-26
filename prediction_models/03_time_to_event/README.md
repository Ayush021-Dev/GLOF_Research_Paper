# Time-to-Event GLOF Model

## Purpose

Estimates relative event hazard over observed lake lifetimes using source event years and first-observation lake geometry features.

## Why this model family

Time-to-event modeling is more appropriate than simple binary classification when lakes are observed for unequal durations and many lakes have no documented event by the end of the record. Those lakes are treated as censored, not as permanent non-events.

## Critical limitation

Only 3 documented events occur inside an observed lake time span. The model is trained as a structural survival baseline; the apparent concordance is not a stable validation estimate.

## Candidate models compared

- Cox proportional hazards with penalizer 0.1.
- Cox proportional hazards with penalizer 1.0.
- Weibull accelerated failure time with penalizer 0.1.

## Selected model

`cox_ph_penalizer_0_1`

## Key metrics

| candidate_model | evaluation_type | rows | observed_events_inside_observation_window | concordance_index | fit_status |
| --- | --- | --- | --- | --- | --- |
| cox_ph_penalizer_0_1 | apparent_concordance_due_low_event_count | 49024 | 3 | 0.9175402458704522 | fit |
| cox_ph_penalizer_1_0 | apparent_concordance_due_low_event_count | 49024 | 3 | 0.9175140056941182 | fit |
| weibull_aft_penalizer_0_1 | apparent_concordance_due_low_event_count | 49024 | 3 | 0.88483186607014 | fit |

## Curves

- `survival_curves.png`

## Outputs

- `model.joblib`
- `time_to_event_risk_scores.csv`
- `survival_training_matrix.csv`
- `metrics.csv`
- `metrics.json`
