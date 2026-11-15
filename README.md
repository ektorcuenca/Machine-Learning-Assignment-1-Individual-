# Smartphone activity recognition with asymmetric decision costs

**Héctor Cuenca · Atlantic Technological University**\
Machine Learning Assignment 1 · November 2026

This project compares multinomial logistic regression (LR) and random forest (RF) for five smartphone-recorded activities: sitting (`SIT`), standing (`STAND`), level walking (`WALK`), upstairs (`UP`) and downstairs (`DOWN`). It adapts Dr Seán Mullery's supplied starter notebook, with attribution and coloured comparisons retained in the notebook.

The final configuration uses **3-second windows, no overlap and 40 accelerometer/gyroscope features**. Both primary models correctly classify all 117 windows from 13 reserved own recordings. Across the three other pool participants, performance is much weaker: equal-person mean minimum-cost accuracy is 45.60% for LR and 44.51% for RF. Their costs are 1696.83 and 1541.80 assignment units per 1000 windows. Neither result establishes clinical readiness.

## Files and data availability

| Path                                              | Purpose                                                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `ML_Assignment1_Hector_Final.ipynb`               | Submitted notebook, including saved outputs and manual configuration cells                         |
| `data/own/`                                       | The author's 40 original phyphox recording ZIPs                                                    |
| `data/pool/`                                      | Place the real anonymised Moodle class pool here                                                   |
| `artifacts/`                                      | Created by the notebook: models, configuration, predictions, audits, metrics and figures           |
| `paper/ML_Assignment1_Paper_Hector_Final_EN.pdf`  | Four-page paper, including references                                                              |
| `paper/ML_Assignment1_Paper_Hector_Final_EN.docx` | Editable paper; complete the repository URL here                                                   |
| `paper/figures/`                                  | Paper figures and rendered equations                                                               |
| `paper/tables/`                                   | Numeric tables reproduced by the support script                                                    |
| `scripts/reproduce_paper_assets.py`               | Reproduces the two paper figures and four numeric tables                                           |
| `reproduction/notebook_saved_outputs/`            | Labelled snapshots of the displayed notebook results; some values are rounded                      |
| `reproduction/paper_assets_provenance.json`       | Records whether each support-script input came from fresh artifact CSVs or saved notebook displays |
| `requirements.txt`                                | Notebook-reported dependency versions and the additional interactive/plotting tools                |

The support package contains the author's raw recordings and the final notebook. **Classmates' raw files and the complete pool artifacts are not included in this package.** Obtain the actual pool from the module's Moodle source. Saved results document the reported run but do not replace a full rerun on those raw files.

Keep each individual phyphox export compressed as its original ZIP. Extract an outer collection archive so the individual recording ZIPs are discoverable. Supported arrangements include `data/pool/S02/S02_UP_01.zip` and `data/pool/S02_UP_01.zip`. Participant prefixes are required; a folder name alone is not used by `parse_name()` as an identity label. The notebook's preparation cells add a folder's prefix to recognised unprefixed filenames after checking for collisions. They rename files without changing their sensor contents.

Do not place pooled recordings in `data/own`. The own-data names `STAND_2.zip` and `STAND_3.zip` are preserved and accepted with warnings.

## Environment and installation

The final notebook reports this execution environment:

| Component    | Recorded version |
| ------------ | ---------------- |
| Python       | 3.14.6           |
| NumPy        | 2.4.6            |
| pandas       | 3.0.3            |
| SciPy        | 1.18.0           |
| scikit-learn | 1.9.0            |
| joblib       | 1.5.3            |

Visual Studio Code is supported: open the repository folder, select the corresponding Python kernel and open the notebook. The working directory must be the repository root because `data` and `artifacts` are relative paths.

## Reproduce the complete experiment

1. Put the 40 original own-data ZIPs in `data/own/` and the actual four-person Moodle pool in `data/pool/`. The recorded run discovered 160 pool ZIPs before catalogue and quality checks.
2. Restart the kernel and execute the notebook from the top. Coloured HTML comparison cells explain starter adaptations; the subsequent Python cells define the actual functions.
3. In **14 Search settings using development only**, retain lengths `[2.0, 2.5, 3.0]`, overlaps `[0.0, 0.5]` and modes `['axes', 'magnitude', 'both']`. The search evaluates 108 model candidates with three recording-level folds: 324 fits.
4. In the following manual window cell, retain `BEST_LENGTH = 3.0`, `BEST_OVERLAP = 0.0`, `BEST_MODE = 'both'` for the paper results.
5. Review the parameter table, then retain `LR_C = 0.1`, `RF_MAX_DEPTH = 5`, `RF_MIN_SAMPLES_LEAF = 3` in the manual model-parameter cell.
6. Run **15** to refit the six frozen variants on 243 development windows. The selected parameters, feature columns, split identifiers, versions and cost matrix are saved in `artifacts/frozen_config.json` and `artifacts/models/`.
7. Run **16–17** for the 13 reserved own recordings and the feature ablation. Run the pool preparation cells, then **18** and its participant-comparison cell.&#x20;

## Reproduce every paper figure and table

| Paper item                                   | Notebook source                                                                                                                                         | Reproduction output                         |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Figure 1, initial-trim evidence              | **5–6**; raw `data/own/SIT_01.zip` and the exact cleaning functions                                                                                     | `paper/figures/fig1_cleaning.png`           |
| Figure 2, transfer class recalls             | **18**, primary models with `min_cost`                                                                                                                  | `paper/figures/fig2_transfer_recalls.png`   |
| Table 1, selected-model validation           | **14–15**, `artifacts/tables/validation_selected_models.csv`                                                                                            | `paper/tables/table1_validation.csv`        |
| Table 2, reserved-recording confusion counts | **16**, `artifacts/tables/within_person_own_holdout_LR_min_cost_confusion.csv`; the same counts apply to both primary models and both rules in this run | `paper/tables/table2_holdout_confusion.csv` |
| Table 3, feature ablation                    | **17**, `artifacts/tables/within_person_all_variants.csv`, filtered to `min_cost`                                                                       | `paper/tables/table3_feature_ablation.csv`  |
| Table 4, per-person primary results          | **18**, `artifacts/tables/across_person_primary.csv`, filtered to `min_cost`                                                                            | `paper/tables/table4_pool_primary.csv`      |

Figure 2 is a per-class recall summary, not a confusion matrix. The notebook separately exports each person's full primary confusion matrix and per-class report:

- `artifacts/tables/across_person_S02_LR_min_cost_confusion.csv` and the corresponding person/model/rule files;
- `artifacts/figures/across_person/S02_LR_min_cost_confusion.png` and corresponding images;
- `artifacts/tables/across_person_per_class.csv`, containing precision, recall, F1 and support;
- `artifacts/tables/across_person_predictions.csv`, containing window-level labels, predictions and probabilities;
- `artifacts/tables/across_person_all_variants.csv`, containing axes, magnitude and combined-feature results under both decision rules.

Include these actual generated outputs with the repository submission. The saved notebook does not embed all their contents; unreported confusion counts, precision values, pooled exclusion reasons or cross-person ablation scores must not be invented from aggregate accuracy.

## Data checks and validation design

Own data are split by complete recording: 27 development and 13 reserved test recordings. Three stratified validation folds also operate on recordings. Every window inherits its recording's partition. Numeric sensor signatures are asserted disjoint across the own development/test boundary. Label, participant code, recording identifier and window times are excluded from predictors. LR scaling is fitted inside each training fold.

All recordings pass through the same rules: metadata start/pause checks, numeric/finite-row checks, monotonic times, duplicate timestamp handling, a shared sensor interval, a two-second initial trim and nearest-time sensor matching. Window checks reject large gaps, insufficient samples or inadequate temporal coverage. No interpolation is used. No selected own-data windows failed the checks. S03 produced 315 usable windows rather than 360; inspect the actual pool audit to establish the file-specific reasons.

## Costs and interpretation

The class order is `SIT, STAND, WALK, UP, DOWN`. Correct predictions cost 0; a non-stair confusion costs 1; confusing UP and DOWN costs 2; predicting stairs for a non-stair activity costs 3; and predicting a non-stair activity for true stairs costs 10.

Training loss and decision cost are separate. LR minimises regularised multinomial log loss. RF greedily reduces weighted Gini impurity at its tree splits. The second prediction rule chooses the class with minimum expected cost from the model's ordered class probabilities. Observed `cost_per_1000` is `1000 * mean(per-window cost)`; it is not an error percentage, monetary expenditure or an independently measured stair-bout rate.

S01 has 40 sensor-content matches with own files, 27 used for development. Under maximum probability it is perfect; under minimum cost each primary model makes one false stair alarm, giving accuracy 359/360 and cost 8.33 per 1000. S01 is excluded from transfer averages. Averaging S02–S04 equally gives:

| Model and rule         | Mean accuracy | Mean macro F1 | Mean cost / 1000 |
| ---------------------- | ------------: | ------------: | ---------------: |
| LR maximum probability |        45.11% |        0.3712 |          2018.78 |
| LR minimum cost        |        45.60% |        0.3768 |          1696.83 |
| RF maximum probability |        44.33% |        0.3423 |          1604.50 |
| RF minimum cost        |        44.51% |        0.3426 |          1541.80 |

These are equal-person averages, not averages weighted by each person's number of windows. The saved notebook also reports each participant separately. Identity is supported by matching numerical sensor data, not by a high classification score.

## Change settings manually without stale results

- To choose an already evaluated window setting, change the `BEST_...` values, rerun both parameter-review/selection cells, then **15–18**. To introduce a new duration or overlap, add it to `LENGTHS` or `OVERLAPS` and rerun **14** first.
- To choose already evaluated model parameters, edit `LR_C`, `RF_MAX_DEPTH` and `RF_MIN_SAMPLES_LEAF`, rerun that manual cell and **15–18**. New parameter combinations must be added to `PARAMS` and evaluated in **14** first. Existing RF pairs are `(5, 3)`, `(10, 3)` and `(None, 1)`.
- Choose settings from development validation. Reserved own recordings and other participants remain evaluation data. Changes to a reported experiment require regenerating downstream predictions and figures.
- Changing `CLEANING` or replacing raw files requires `clean_recording.cache_clear()` or a kernel restart, followed by rebuilding affected tables. The cache uses paths, so it does not automatically notice changed file contents.

The final submitted notebook ends with the pool analysis; this snippet supplies the live prediction call. Without true labels, it reports predictions rather than accuracy or cost.

## AI use and attribution

OpenAI ChatGPT/Codex assisted with interpreting the assignment, explaining methods, proposing notebook adaptations and implementing manual configuration controls.

Primary references are Anguita et al. (2013), *A Public Domain Dataset for Human Activity Recognition Using Smartphones*; the UCI HAR dataset, DOI `10.24432/C54S4K`; Breiman (2001), *Random Forests*, DOI `10.1023/A:1010933404324`; Elkan (2001), *The Foundations of Cost-Sensitive Learning*; and the official scikit-learn documentation. The assignment brief, starter notebook and anonymised pool are ATU module materials supplied through Moodle.
