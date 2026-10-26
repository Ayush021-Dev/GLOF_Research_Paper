# GLOF Model Decision Log

- Input folder: `master_lake_dataset`.
- Output folder: `prediction_models`.
- Modeling order followed: similarity ranking, pre-outburst change detection, time-to-event.
- Loaded 216,505 feature observations.
- Loaded 200,542 master-lake/year feature rows.
- Documented matched GLOF events: 65; unique matched event lakes: 53.
- Event-year handling: exact source year is used when numeric; numeric approximate source year is used only when exact year is missing; non-numeric ranges such as `2001-2003` are retained in source metadata but excluded from time-based training matrices.
- No GLOF=0 labels were created. Unmatched or non-event-inventory lakes remain unlabeled/comparison/censored.
- Similarity ranking target: 53 documented-positive master lakes and 48,971 unlabeled comparison lakes.
- Similarity ranking features: 67 lake-level aggregate temporal/shape/area columns.
- Similarity ranking trained. Selected `hist_gradient_boosting` from 4 candidates (proxy AP=0.03947971450384746, proxy ROC-AUC=0.9461654301660767).
- Pre-outburst target construction: for each documented event with a numeric source year, selected the latest real lake-year observation strictly before that event year.
- Pre-outburst anomaly model trained with 3 documented pre-event cases and unlabeled background lake-year rows; selected `one_class_svm` (proxy AP=0.0027528354394195436, proxy ROC-AUC=0.8120666666666667).
- Time-to-event construction: event lakes are events only when the source event year falls inside that lake's observed year span; all other lake histories are right-censored at last observation.
- Time-to-event predictors use first-observation lake geometry/area features to avoid using post-event temporal change as a baseline survival predictor.
- Time-to-event model trained with 3 observed in-window event lakes; selected `cox_ph_penalizer_0_1` (apparent concordance=0.9175402458704522).
