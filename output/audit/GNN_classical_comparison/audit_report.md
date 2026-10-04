# GNN and Classical Baseline Audit

## Data and partitions

- Source table: `database/csv/Acetylcholinesterase_Homo_sapiens_cleaned_pipeline`
- Valid graph records: 4571; rejected records: 0
- Train / validation / test rows: 3108 / 548 / 915
- Every partition exactly matches the saved GNN source-row IDs; scaffold groups are disjoint.
- The test partition is held out from both classical hyperparameter search and model fitting.
- Classical tuning uses 8 Optuna trials per model/representation and 3-fold `GroupKFold` over training scaffolds only.
- RF/XGBoost fit on the same training rows as the GNN. The saved GNN validation rows are not used for classical fitting or tuning.

## Feature verification

- Node features (7): atomic_number, degree, formal_charge, hybridization, aromaticity, hydrogen_count, ring_membership. Categorical inputs use embedding indices; formal charge is shifted by 5 after clipping to [-5, 5].
- Global descriptors (14): MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, MaxPartialCharge, MinPartialCharge, NumHeteroatoms, NumRotatableBonds, FractionCSP3, NumAromaticRings, RingCount, NumAliphaticNitrogens, NumFormalCharge. Missing partial-charge values are imputed using means from training rows only.
- Edge channels: 13, not 3. Encoded as single_bond, double_bond, triple_bond, aromatic_bond, conjugated, aromatic, ring_membership, stereo_none, stereo_any, stereo_z, stereo_e, stereo_cis, stereo_trans. This is four bond-type one-hot channels, three bond properties, and six stereo one-hot channels; each chemical bond is represented in both directions.
- The code currently implements 13 edge channels. If the intended specification is exactly 3 edge inputs, the current model does not match that specification and should be changed before interpreting its result.

## Leakage audit

- RF/XGBoost predictors are restricted to the explicit Morgan and/or RDKit descriptor matrices. PIC50, IC50, SMILES, molecule/InChIKey identifiers, and scaffold labels are excluded from model inputs.
- Morgan fingerprints and molecular descriptors depend only on molecular structure. Mean imputation is inside each CV pipeline, so each fold estimates missing-value means from that fold's training rows; final imputation is fitted on the full model-training partition.
- Hyperparameter search uses training rows and scaffold-grouped folds only; the test set is evaluated after final fitting.
- The scaffold split keeps each scaffold in exactly one of train, validation, or test.
- The current baseline feature-selection path has no direct target-column leakage. The earlier metric artifacts were not reused for this clean comparison.

## Results

| model        | features          |   n_features |   n_train |   n_test |   cv_mae |   test_mae |   test_rmse |   test_r2 |   test_pearson_r |   test_spearman_r |
|:-------------|:------------------|-------------:|----------:|---------:|---------:|-----------:|------------:|----------:|-----------------:|------------------:|
| XGBoost      | combined          |         1038 |      3108 |      915 |   0.6479 |     0.6352 |      0.8347 |    0.4553 |           0.6795 |            0.6650 |
| XGBoost      | morgan            |         1024 |      3108 |      915 |   0.6569 |     0.6510 |      0.8519 |    0.4326 |           0.6648 |            0.6586 |
| RandomForest | combined          |         1038 |      3108 |      915 |   0.6875 |     0.6764 |      0.8507 |    0.4341 |           0.6644 |            0.6524 |
| RandomForest | morgan            |         1024 |      3108 |      915 |   0.6948 |     0.6852 |      0.8631 |    0.4175 |           0.6506 |            0.6523 |
| GNN          | graph+descriptors |           34 |      3108 |      915 | nan      |     0.6908 |      0.9093 |    0.3535 |           0.6225 |          nan      |
| XGBoost      | descriptors       |           14 |      3108 |      915 |   0.7276 |     0.6969 |      0.8892 |    0.3818 |           0.6188 |            0.5903 |
| RandomForest | descriptors       |           14 |      3108 |      915 |   0.7366 |     0.7169 |      0.8978 |    0.3698 |           0.6140 |            0.5894 |

## Assessment

The best classical baseline (XGBoost with combined features) has lower held-out MAE than the GNN by 0.0556 pIC50. This run does not support added predictive value from the GNN.

This is one fixed scaffold split. Treat the comparison as evidence for this partition, not a stable estimate across chemical space; repeated scaffold splits or an external test set are needed for a stronger claim.
