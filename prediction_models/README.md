# GLOF Prediction Models

Three separate model artifacts were trained from the master lake dataset:

1. `01_glof_similarity_ranking`: ranks lakes by similarity to documented GLOF lakes.
2. `02_pre_outburst_change_detection`: anomaly-ranks lake-year records for pre-outburst-like change behavior.
3. `03_time_to_event`: survival/time-to-event baseline using censored lake histories.

No model in this folder uses synthetic imagery, generated masks, segmentation output, fabricated observations, inferred GLOF events, or blanket GLOF=0 labels.

Read each subfolder `README.md` before using scores. The first model is the most usable immediately. The second and third are constrained by the small number of matched documented events with usable pre-event or in-window timing.
