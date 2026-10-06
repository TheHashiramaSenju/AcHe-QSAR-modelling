# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2026-10-06

### Initial documented research release

This release establishes the first versioned and archived research state of the
AChE molecular machine-learning project.

### Included

- ChEMBL AChE activity-data retrieval
- CHEMBL220 target definition
- IC50 activity processing and normalization
- Molecular structure cleaning
- Salt removal
- InChIKey generation
- Duplicate and replicate handling
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
- Classical ML vs GNN comparison
- Forensic data and methodology audit
- Reproducibility artifacts and saved experimental outputs
- Complete scientific report
- Citation metadata and research provenance through `CITATION.cff`

### Research scope

The release documents an end-to-end workflow for predicting
acetylcholinesterase (AChE) inhibitory potency as pIC50 from molecular
structure.

The study evaluates both engineered molecular representations
(Morgan fingerprints and RDKit descriptors) with classical ensemble models
and a graph-based molecular representation with a Graph Neural Network.

The repository also preserves the associated diagnostics, audit findings,
experimental outputs, and methodological limitations identified during the
study.

### Versioning and archival

This release is associated with the versioned research record and its
archival DOI:

**DOI:** https://doi.org/10.5281/zenodo.23182018

The repository contains `CITATION.cff`, `RESEARCH_RECORD.md`,
`REPORT.pdf`, saved experimental outputs, and audit artifacts to preserve
the provenance and reproducibility of this research snapshot.

### Attribution

Please cite the corresponding software release and scientific report when
using this repository, its methodology, code, or derived results.