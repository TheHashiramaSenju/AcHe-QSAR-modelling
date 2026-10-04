# 1. Executive Summary

This report is a forensic scientific audit of the actual pipeline currently implemented in [src/CDDgem.py](src/CDDgem.py) and the saved dataset at [database/csv/Acetylcholinesterase_Homo_sapiens.csv](database/csv/Acetylcholinesterase_Homo_sapiens.csv). I did not change the modeling pipeline, retrain models, or alter the data. The goal was to establish whether the current reported test metrics are scientifically defensible as scaffold-test estimates.

Overall conclusion: C. NOT YET TRUSTWORTHY.

Why:

- The raw data retrieval is consistent with CHEMBL target CHEMBL220 and IC50-only endpoint selection.
- The scaffold split is properly implemented as a whole-scaffold, size-sorted greedy split and does not show scaffold or molecule overlap between train and test.
- The model optimization is nested inside the training set: [src/CDDgem.py](src/CDDgem.py#L877-L922) uses GroupKFold on the training rows only.
- However, the thresholding logic is physically unit-inconsistent because the code applies `standard_value <= 10000` after unit normalization exists, rather than applying a normalized IC50 threshold consistently across units.
- The duplicate aggregation precedes relation and potency filtering, which means the target is built from a pre-filter cohort, but in this dataset no key had both kept and removed rows, so the empirical target shift is zero in the current snapshot.
- The current design is scientifically defensible as a scaffold extrapolation estimate only with caveats, but it is not fully trustworthy as a clean, units-consistent QSAR benchmark.

---

# 2. Actual Executed Pipeline

The path that actually affects the current experiment is the following:

1. Retrieval and CSV export: [src/CDDgem.py](src/CDDgem.py#L120-L176), method `DuckDBEngine.create_and_load_csv`.
2. Validity and null filtering: [src/CDDgem.py](src/CDDgem.py#L184-L286), `DataCleaning.null_and_columnhandler`.
3. Salt stripping and SMILES cleaning: [src/CDDgem.py](src/CDDgem.py#L188-L286), `DataCleaning.InChIstandardization`.
4. Supported-unit filtering and IC50 normalization: [src/CDDgem.py](src/CDDgem.py#L327-L376), `DataCleaning.IC50_units_standardization`.
5. pIC50 conversion: [src/CDDgem.py](src/CDDgem.py#L378-L396), `DataCleaning.IC50_to_PIC50Conv`.
6. Duplicate/replicate aggregation and MAD filtering: [src/CDDgem.py](src/CDDgem.py#L398-L416), `DataCleaning.InChI_based_duplicate_res`.
7. Relation and potency thresholding: [src/CDDgem.py](src/CDDgem.py#L418-L456), `DataCleaning.relationalvalue`.
8. Structural feature generation and scaffold assignment: [src/CDDgem.py](src/CDDgem.py#L493-L577), `DataEng` methods and `DataEng._split_frame`.
9. Model training and Optuna search: [src/CDDgem.py](src/CDDgem.py#L815-L1069), `Model._optimize_and_evaluate` and `Model.objective`.
10. Final metric calculation: [src/CDDgem.py](src/CDDgem.py#L884-L995), `Model._save_run_outputs`.

Important distinction: The existence of helper methods does not mean they are used. The `Validation.random_split` method in [src/CDDgem.py](src/CDDgem.py#L1179-L1184) is not used in the current model path; it is effectively UNUSED for the reported results.

---

# 3. Raw Data Audit

## 3.1 Provenance

The saved raw CSV file is [database/csv/Acetylcholinesterase_Homo_sapiens.csv](database/csv/Acetylcholinesterase_Homo_sapiens.csv) and its actual counts from the snapshot are:

- Raw rows: 8,793
- Unique target CHEMBL IDs: 1
- Unique molecule CHEMBL IDs: 7,179
- Unique assay CHEMBL IDs: 1,020
- Unique standard_type: IC50 only
- Unique standard_relation: '=', '>', '<', '~', '>>'
- Unique standard_units: nM, ug.mL-1, 10'5pM, 10'6pM, 10^-4microM, 10'3pM
- Unique target pref_name: 1 (Acetylcholinesterase)

This is consistent with a single-target retrieval with multiple assay contexts and mixed unit/relationship records, which is normal in ChEMBL but needs explicit normalization and filtering logic. The raw CSV does include assay-level and relation diversity, plus a small set of non-supported units.

## 3.2 What is actually retrieved

The network code in [src/CDDgem.py](src/CDDgem.py#L122-L176) performs:

- `activity.filter(target_chembl_id=self.target_chembl_id, standard_value__isnull=False, standard_type__in=["IC50"])`

This means the retrieval is target-specific and endpoint-specific, but it does not constrain:

- relation type
- unit type
- assay context
- duplicate activity rows
- salts or stereochemistry differences
- multiple measurement records per molecule

This is expected ChEMBL behavior, not necessarily a bug, but it is a scientific hazard if not handled intentionally.

---

# 4. Unit / IC50 / pIC50 Audit

## 4.1 Supported conversions

The supported-unit mapping is in [src/CDDgem.py](src/CDDgem.py#L23-L25) and [src/CDDgem.py](src/CDDgem.py#L327-L376):

| original_unit | conversion_factor_to_nM | physical_interpretation | correct? |
|---|---:|---|---|
| nM | 1.0 | direct nM concentration | YES |
| 10'3pM | 1.0 | 10^-3 pM = 10^-12 M = 1e-3 nM | YES |
| 10'5pM | 100.0 | 10^5 pM = 100 nM | YES |
| 10'6pM | 1000.0 | 10^6 pM = 1000 nM | YES |

The code uses the same mapping as the audit summary: [output/audit/20260927_112634/config_snapshot.json](output/audit/20260927_112634/config_snapshot.json). The audit script confirms these conversion factors by prefix math.

## 4.2 pIC50 formula

The conversion is:

- `IC50_nM = standard_value * factor_to_nM`
- `pIC50 = -log10(IC50_nM * 10^-9)` for standard_type == IC50 and positive IC50

This is mathematically correct.

## 4.3 Threshold inconsistency

The non-trivial scientific concern is not the formula itself. It is the threshold application:

- The pipeline does `standard_value <= 10000` in [src/CDDgem.py](src/CDDgem.py#L440-L456)
- The normalized concentration is `IC50_nM`, and the intended threshold concept is physically a 10,000 nM activity cutoff.

This is inconsistent across units because raw standard_value has different physical meanings depending on unit.

Examples:

- If unit is nM, 10,000 means 10,000 nM.
- If unit is 10'5pM, 10,000 means 1,000,000 nM.
- If unit is 10'6pM, 10,000 means 10,000,000 nM.
- If unit is 10'3pM, 10,000 means 10,000 nM.

So the implementation is not physically unit-consistent if the intended threshold is 10,000 nM across all records. The audit output confirms the current code reports A and B to be the same at final-cohort level ([output/audit/20260927_112634/filtering_comparison.csv](output/audit/20260927_112634/filtering_comparison.csv)), but that is because the actual rows surviving the supported-unit filter are not strongly unit-sensitive in this snapshot. The concept remains scientifically wrong unless all values are first normalized to the same unit before thresholding.

## 4.4 Current dataset-specific impact

The raw snapshot has:

- nM: 8,702
- ug.mL-1: 83
- 10'5pM: 3
- 10'6pM: 2
- 10^-4microM: 2
- 10'3pM: 1

The code excludes unsupported units: [src/CDDgem.py](src/CDDgem.py#L327-L376). The current dataset therefore has a small but real unit-heterogeneity phenomenon, and the raw thresholding is not physically unified.

---

# 5. Filtering Order Audit

The real order is:

1. raw ChEMBL retrieval
2. validity/null filtering
3. salt removal and SMILES cleaning
4. InChIKey generation
5. potential_duplicate removal
6. supported-unit filtering and IC50 normalization
7. pIC50 conversion
8. replicate median/MAD calculation
9. group MAD filtering
10. exact duplicate removal after MAD
11. relation filtering (`standard_relation == "="`)
12. current potency threshold (`standard_value <= 10000`)
13. feature generation and scaffold split
14. model optimization and evaluation

The dataset-stage counts in [output/audit/20260927_112634/stage_counts.csv](output/audit/20260927_112634/stage_counts.csv) show that the filter chain reduces from 8,793 to 4,571 rows.

The actual scientific risk: in the duplicate aggregation logic the median is computed before the later relation and potency filters. The current implementation in [src/CDDgem.py](src/CDDgem.py#L398-L416) does not use a kept-only cohort when calculating per-InChIKey medians. That is not necessarily/test leakage, because the code is not fitting a model on excluded rows, but it is a cohort-definition issue. The current data snapshot had zero keys with both kept and removed rows after the later filters, so the empirical impact on final target values is zero in this dataset.

Therefore:

- DATA LEAKAGE: NO
- TARGET-CONSTRUCTION / COHORT-DEFINITION ISSUE: PARTIALLY yes, but empirically minimal in this snapshot

---

# 6. Replicate and Duplicate Audit

The duplicate logic is in [src/CDDgem.py](src/CDDgem.py#L398-L416):

- group by InChIKey
- median PIC50 per group
- absolute deviance from median
- median absolute deviation per group
- drop groups with group_MAD > 1.0
- replace PIC50 with median
- drop exact duplicates on InChIKey + cleaned_smiles + PIC50

This is a robust single-target collapse strategy for replicate assays, but it is a statistical assumption: if replicate measurements are noisy and near-identical, the median acts as a robust representative. It does not assume all records are biologically identical; it assumes that within the same molecule and same cleaned representation, a robust center is an acceptable summary for target construction.

From the audit results:

- replicate groups: 6,421
- groups with MAD > 1.0: 34
- rows in those groups: 74
- final molecules affected: not zero, but limited

The statistical assumption is not automatically invalid, but it is a design decision rather than a universal truth.

---

# 7. Target Construction Audit

The raw pIC50 conversion is mathematically consistent with the formula used in [src/CDDgem.py](src/CDDgem.py#L378-L396). The final target distribution is in [output/audit/20260927_112634/target_distribution.csv](output/audit/20260927_112634/target_distribution.csv) and the summary in [output/audit/20260927_112634/audit_summary.json](output/audit/20260927_112634/audit_summary.json).

Observed target stats after current processing:

- min pIC50: 4.92082
- median pIC50: 6.30103
- max pIC50: 10.96059

The final cohort is not degenerate and is in a reasonable pIC50 range for IC50 data.

The key scientific caveat is that the final target is not built on a physically consistent unit-normalized thresholding regime for every record type. This matters for scientific interpretation, even if the current snapshot does not show large differences.

---

# 8. Molecular Structure Audit

The structure cleaning pipeline is in [src/CDDgem.py](src/CDDgem.py#L188-L286). The audit output in [output/audit/20260927_112634/audit_summary.json](output/audit/20260927_112634/audit_summary.json) reports:

- failed RDKit parse rows: 1
- empty or invalid cleaned SMILES: 1
- disconnected molecules after salt removal: 90
- unusual elements: 0

This means the structure pipeline is generally valid but there are real cases of chemically awkward or disconnected molecules that were not removed. Salt stripping is not always scientifically neutral; it can merge distinct salts and charges into a single representation. The current code does not prove that every salt-removal decision is chemically equivalent across the dataset, but this is a normalization decision rather than direct target leakage.

---

# 9. Scaffold Split Audit

The actual scaffold split is implemented in [src/CDDgem.py](src/CDDgem.py#L578-L597) and [src/CDDgem.py](src/CDDgem.py#L730-L767) and is a greedy, size-sorted whole-scaffold allocation algorithm.

Classification: deterministic size-sorted greedy scaffold split.

This means:

- scaffolds are generated by Murcko scaffold from cleaned SMILES
- scaffold groups are sorted by group size descending
- whole scaffold groups are assigned to train until the 80% training target count is reached
- remaining groups go to test

This is not a random scaffold split. It is a structured split that can disproportionately emphasize large scaffolds in train and leave small uncommon scaffolds to test. It is a valid scaffold-extrapolation benchmark, but it is not a random sample of molecules.

From the dataset audit:

- Train molecules: 3,656
- Test molecules: 915
- Train fraction: 0.80
- Test fraction: 0.20
- Train scaffolds: 1,191
- Test scaffolds: 915
- Shared scaffold count: 0
- Shared InChIKey count: 0
- Shared cleaned SMILES count: 0
- `__NO_SCAFFOLD__` handling exists, but is not a major issue in the current snapshot because the split audit reported zero overlap and no giant combined group issue.

The scaffold split is therefore cleanly separated in the snapshot.

---

# 10. Train/Test Leakage Audit

The train/test split is performed after feature generation in the code and uses scaffold groups only; the split itself is separate from model fitting. The core check is in [src/CDDgem.py](src/CDDgem.py#L730-L767).

The audit shows:

- no scaffold overlap
- no InChIKey overlap
- no cleaned SMILES overlap

This is strong evidence against molecule-level leakage and scaffold-level leakage in the actual current snapshot.

Therefore the current split is not leaking by row identity.

---

# 11. Feature Leakage Audit

The final model feature sets are assembled in [src/CDDgem.py](src/CDDgem.py#L892-L922) and the model feature selection logic in [src/CDDgem.py](src/CDDgem.py#L667-L696) exists, but the result reported is from the actual path. The audit script checked the model feature set and found no identifier leakage and no direct pIC50 features. This is consistent with the code path.

The final features used are:

- Morgan fingerprints: 1024 bits, radius 2
- RDKit descriptors: 14 descriptor columns
- combined variant: Morgan + descriptors

The existing helper functions for feature selection and random split exist but are not used in the current reported path; they are effectively unused for the final model.

---

# 12. Cross-Validation Audit

The CV is in [src/CDDgem.py](src/CDDgem.py#L877-L922):

- `cv = GroupKFold(n_splits=5)`
- `groups=scaffolds_train`
- `study.optimize(..., direction="maximize")` with `score.mean()` from `cross_val_score(..., scoring="neg_mean_absolute_error")`

This means:

- the training set is used for nested CV
- the test set is not included in Optuna
- scaffold groups remain in a single fold

This is a valid nested scaffold-group optimization design.

---

# 13. Optuna Audit

Optuna is used in [src/CDDgem.py](src/CDDgem.py#L827-L922) for the three model families. It is properly nested within the training set. The `direction="maximize"` with `neg_mean_absolute_error` is mathematically consistent with maximizing the negative error, but note that the returned objective is the mean of negative MAE across folds, which is a lower-is-better metric being converted to a larger-is-better objective. This is correct.

The LightGBM parameter `subsample` is searched but it is not necessarily a complete guarantee that subsampling is active in the same way across all training configurations if the underlying model defaults differ. The current code does search the parameter; the parameter is consumed when the model is instantiated. This is not an inactive parameter in the current implementation, but the scientific meaning depends on the model and library version.

---

# 14. Model Configuration Audit

The main model families are explicitly:

- RandomForest in [src/CDDgem.py](src/CDDgem.py#L798-L835)
- XGBoost in [src/CDDgem.py](src/CDDgem.py#L836-L856)
- LightGBM in [src/CDDgem.py](src/CDDgem.py#L857-L877)

The code uses:

- random_state=50 for RF and XGB/LGBM
- Optuna TPESampler with seed=50
- `GroupKFold` based on scaffold groups

This is reproducible at the code level, but only if the same library versions and same dataset snapshot are used.

---

# 15. Final Metric Audit

The final metric code is in [src/CDDgem.py](src/CDDgem.py#L955-L995):

- MAE is computed using `mean_absolute_error`
- RMSE uses `np.sqrt(mean_squared_error)`
- R² uses `r2_score`
- Pearson and Spearman use `pearsonr` and `spearmanr`

The saved prediction files are written, and index alignment is kept in the DataFrame. The design is logically coherent.

However, the audit still cannot treat the current MAE as a clean scientific performance estimate without caveats because the thresholding logic is unit-inconsistent and the current pipeline uses a pre-filtered duplicate-aggregation target definition.

---

# 16. Reproducibility Audit

The code has explicit seeds for:

- Optuna: `TPESampler(seed=50)`
- RF: `random_state=50`
- XGB: `random_state=50`
- LGBM: `random_state=50`

The project also includes a dataset snapshot with stable hashing in [output/audit/20260927_112634/config_snapshot.json](output/audit/20260927_112634/config_snapshot.json). That is strong evidence for reproducibility at the dataset level, provided the exact environment is preserved.

Reproducibility is PARTIALLY satisfied: code-level randomization is controlled, but the full end-to-end experiment still depends on the exact ChEMBL snapshot and library stack.

---

# 17. Existing Audit-System Audit

The current audit subsystem in [src/data_audit.py](src/data_audit.py) is useful but it is not a substitute for a scientific verdict. It checks the actual dataset and code path, and it is strong on stage tracking and scaffold split verification.

It does have blind spots:

- it does not fully isolate the unit-threshold inconsistency as a scientific issue in all contexts
- it does not independently recompute every model metric from raw predictions in a separate code path
- it treats the current threshold policy as a warning, not as a definite invalidation

This is why the audit script reports PASS/WARNING/CRITICAL but those do not automatically mean the model results are scientifically validated.

---

# 18. Known Red Flags

## RED FLAG A — unit-inconsistent thresholding
- File: [src/CDDgem.py](src/CDDgem.py#L440-L456)
- What current code does: uses `standard_value <= 10000` after unit normalization exists elsewhere.
- Why it matters: raw values mean different physical concentrations depending on the unit.
- Current effect on MAE: likely limited in this snapshot but scientifically important.
- Severity: HIGH

## RED FLAG B — duplicate/replicate aggregation before later filtering
- File: [src/CDDgem.py](src/CDDgem.py#L398-L416)
- What current code does: computes median PIC50 before later relation and threshold filters.
- Why it matters: target construction is partially cohort-dependent.
- Current effect on MAE: likely none in the current snapshot, but not a general guarantee.
- Severity: MEDIUM

## RED FLAG C — greedy scaffold allocation
- File: [src/CDDgem.py](src/CDDgem.py#L578-L597)
- What current code does: sorts by scaffold size and greedily fills train.
- Why it matters: this is a structured split, not a random scaffold split.
- Current effect on MAE: yes, it changes the distribution of the benchmark.
- Severity: MEDIUM

## RED FLAG D — LightGBM subsample state
- File: [src/CDDgem.py](src/CDDgem.py#L857-L877)
- What current code does: tunes `subsample` but this depends on exact model semantics and version.
- Why it matters: could be active or effectively inactive depending on library behavior.
- Current effect on MAE: unknown.
- Severity: LOW/MEDIUM

## RED FLAG E — helper methods exist but are not used
- File: [src/CDDgem.py](src/CDDgem.py#L1179-L1184)
- What current code does: random split helper exists and is not used in the current reported experiment.
- Why it matters: it should not be taken as evidence the experiment used a random split.
- Current effect on MAE: none, but potential confusion.
- Severity: LOW

---

# 19. Required Corrections

The required corrections are analytical and procedural, not an automatic rewrite:

1. Normalize all concentrations to a single unit before applying any potency threshold.
2. Apply the threshold consistently to IC50_nM rather than raw `standard_value`.
3. Record the exact dataset snapshot and model environment, and tie the final metrics to those immutable artifacts.
4. Treat duplicate aggregation as a target-construction step and document when it happens relative to thresholding.
5. Report scaffold split as deterministic and size-sorted rather than as a generic scaffold split without qualification.
6. Re-run the model only after these scientific issues are explicitly addressed in a controlled, documented benchmark workflow.

---

# 20. Final Scientific Verdict

Verdict: C. NOT YET TRUSTWORTHY.

This is not because the code is obviously broken. It is because the current experiment is scientifically valid only under a narrow interpretation: a scaffold-generalization estimate on a fixed dataset snapshot, under a deterministic scaffold split, with consistent train/test identity separation, but with a unit-inconsistent potency threshold and a target-construction pipeline that is not fully documented as threshold-safe across units.

The current report may still be useful as a benchmark, but it should not be treated as a clean, publication-ready, unit-consistent QSAR estimate without correcting the thresholding and documenting the final dataset-creation pipeline more rigorously.

---

# Final 10 Questions

1. Is the final dataset construction scientifically reproducible? PARTIALLY. The dataset snapshot and code-level seeds are documented, but the exact environment and data version must be preserved for exact reproduction.
2. Is pIC50 mathematically correct? YES. The formula used in the code matches the standard convention for IC50 in nM: `-log10(IC50_nM * 10^-9)`.
3. Are all unit conversions correct? YES. The supported conversions are mathematically correct for the encoded unit mapping.
4. Is the potency threshold physically unit-consistent? NO. `standard_value <= 10000` is not physically equivalent across units; a true 10,000 nM threshold should be applied after normalization to nM.
5. Can excluded measurements influence final target values? NO in the current snapshot. The audit showed zero keys with both kept and removed rows whose median changed after later filtering.
6. Is there molecule-level leakage? NO. The scaffold split has zero InChIKey overlap and zero cleaned SMILES overlap.
7. Is there scaffold-level leakage? NO. The audit reports zero scaffold overlap.
8. Is Optuna completely isolated from the test set? YES. Objective evaluation is done only on training data using scaffold-aware GroupKFold.
9. Are the final metrics independently recomputable? PARTIALLY. The formulas are standard and the predictions can be recomputed from saved outputs, but the pipeline was not independently rerun from raw data in a separate implementation here.
10. Can the current ~0.63 MAE be treated as a defensible scaffold-test estimate? PARTIALLY. It is a defensible scaffold-estimation number under the current split, but it is not yet a fully trustworthy scientific benchmark because of the threshold inconsistency and mixed-unit target-definition details.

---

## Scientific note

This audit was designed to answer whether the reported metrics are valid on their own terms. The answer is: the scaffold split and nested CV are reasonably well designed, and no train/test identity leakage is evident in the current snapshot. However, the current potency threshold is not physically unit-consistent, and the target-construction cohort is not fully documented with respect to the exact threshold semantics. Therefore, the metrics are not yet fully trustworthy as a clean scientific estimate.
