# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2026-10-06

### Initial documented research release

This release represents the documented state of the AChE molecular machine-learning project at the time of the v1.0.0 release.

### Included

- ChEMBL AChE activity-data retrieval
- CHEMBL220 target definition
- IC50 activity processing
- Molecular structure cleaning
- Salt removal
- InChIKey generation
- Duplicate and replicate handling
- IC50 normalization
- pIC50 transformation
- Morgan fingerprints
- RDKit molecular descriptors
- Combined molecular representations
- Bemis–Murcko scaffold grouping
- Scaffold-aware train/test partitioning
- Grouped cross-validation
- Random Forest modelling
- XGBoost modelling
- LightGBM modelling
- Optuna hyperparameter optimization
- Multi-metric model evaluation
- Residual and diagnostic analysis
- Graph Neural Network evaluation
- Forensic data and methodology audit
- Complete scientific report
- Reproducibility artifacts and saved experimental outputs

### Scientific scope

The release documents an AChE pIC50 regression workflow and its evaluated results.

The repository explicitly records methodological limitations identified during the forensic audit. In particular, the audit identified a unit-consistency issue concerning the potency threshold and recommends applying the threshold to normalized IC50 values.

The release therefore should be interpreted as a documented research snapshot rather than as a claim of a publication-ready or state-of-the-art QSAR benchmark.

### Attribution

Please cite the corresponding software release and associated scientific report when using this repository, methodology, code, or derived results.