# AChE Data Audit

## 1. Run Information

- Timestamp: 20260927_112634
- Target: `CHEMBL220`
- Input snapshot: `/media/notshadow/d5dd988b-c393-4302-aa45-32bcfc8463c2/WorkFolder/DrugDiscovery-BioInformatics/database/csv/Acetylcholinesterase_Homo_sapiens.csv`
- Input SHA-256: `492f42efacc8b49bfa607c0e6ffa66f4f40f5822094f68a54509e58e61ee4418`
- Stable final molecular data SHA-256: `b6ad18c926faf5f0221f33342bdc2a105a4bc84471d46ff7c9ed4f496ca20a62`
- Git branch / commit: `Correction-Branch` / `bc86f0411a8cd99b408a5a88ae25607cdedee0ad`
- This mode reads the saved raw CSV snapshot; it does not query ChEMBL or modify datasets.

## 2. Dataset Flow

|   stage_order | stage                                           |   before_rows |   rows_after |   rows_removed |   percent_removed |   unique_InChIKeys |   unique_cleaned_SMILES |
|--------------:|:------------------------------------------------|--------------:|-------------:|---------------:|------------------:|-------------------:|------------------------:|
|             1 | Raw saved ChEMBL retrieval snapshot             |          8793 |         8793 |              0 |          0        |                  0 |                    7175 |
|             2 | Initial validity/null handling                  |          8793 |         8072 |            721 |          8.1997   |                  0 |                    6571 |
|             3 | Salt removal and SMILES cleaning                |          8072 |         8072 |              0 |          0        |                  0 |                    6471 |
|             4 | InChIKey generation                             |          8072 |         8072 |              0 |          0        |               6471 |                    6471 |
|             5 | potential_duplicate removal                     |          8072 |         7248 |            824 |         10.2081   |               6471 |                    6471 |
|             6 | Supported-unit filter and IC50 normalization    |          7248 |         7185 |             63 |          0.869205 |               6421 |                    6421 |
|             7 | pIC50 conversion                                |          7185 |         7185 |              0 |          0        |               6421 |                    6421 |
|             8 | InChIKey replicate median/MAD calculation       |          7185 |         7185 |              0 |          0        |               6421 |                    6421 |
|             9 | Group MAD <= 1.0 filter                         |          7185 |         7110 |             75 |          1.04384  |               6387 |                    6387 |
|            10 | Exact duplicate removal after MAD               |          7110 |         6387 |            723 |         10.1688   |               6387 |                    6387 |
|            11 | standard_relation == "=" filter                 |          6387 |         5695 |            692 |         10.8345   |               5695 |                    5695 |
|            12 | Current standard_value <= 10000 activity filter |          5695 |         4571 |           1124 |         19.7366   |               4571 |                    4571 |

The existing validity filter runs before salt cleaning. Unit support filtering and unit normalization occur in one current method; the audit reports them together. Median aggregation is a transform and does not itself reduce row count. The current code then applies group MAD filtering, exact duplicate removal, relation filtering, and the raw `standard_value` activity cutoff.

## 3. Unit Audit

- Original units before normalization: `{"nM": 8702, "ug.mL-1": 83, "10'5pM": 3, "10'6pM": 2, "10^-4microM": 2, "10'3pM": 1}`
- Unsupported or null raw unit records: 85
- Conversion examples and the code's conversion factors are in `unit_conversion_examples.csv`.
- Unit distributions are in `unit_distribution.csv`.

### Raw-value versus normalized-concentration filter

- Current A (`standard_value <= 10000`) retains 4571 records.
- B (`IC50 in nM <= 10000`) retains 4571 records.
- Differently classified: 0 (0.0000% of post-relation candidates); 0 unique InChIKeys affected.
- No threshold behavior was changed. Detailed rows are in `filtering_differences.csv` and `filtering_differences_by_unit.csv`.

## 4. Target Audit

- Final pIC50 stats: `{"count": 4571, "missing": 0, "nonfinite": 0, "min": 4.92082, "max": 10.96059, "mean": 6.515185706628747, "median": 6.30103, "std": 1.1266659397617451, "quantiles": {"1%": 5.00877, "5%": 5.106795, "25%": 5.59007, "50%": 6.30103, "75%": 7.20761, "95%": 8.6202425, "99%": 9.630444000000002}, "zero_IC50": 0, "negative_IC50": 0, "zero_pIC50": 0, "negative_pIC50": 0, "extreme_pIC50_below_0_or_above_20": 0, "normalized_IC50_stats_nM": {"count": 4571, "missing": 0, "nonfinite": 0, "min": 0.01095, "max": 10000.0, "mean": 1787.8618176547802, "median": 500.0, "std": 2547.595871865583}, "threshold_boundary_counts": {"within_1_percent_of_10000_nM": 32, "within_5_percent_of_10000_nM": 67, "within_10_percent_of_10000_nM": 114}, "raw_standard_value_sanity": {"null_or_non_numeric": 0, "nonfinite": 0, "zero": 1, "negative": 0}}`
- Full target and split distributions are in `target_distribution.csv` and `split_target_distribution.csv`.

## 5. Duplicate / Replicate Audit

- Replicate group summary: `{"InChIKey_groups": 6421, "groups_MAD_over_1": 34, "percent_groups_MAD_over_1": 0.5295125369880082, "measurements_in_groups_MAD_over_1": 74, "percent_measurements_in_groups_MAD_over_1": 1.0299234516353515, "maximum_measurements_per_InChIKey": 55, "MAD_distribution": {"count": 6421, "missing": 0, "nonfinite": 0, "min": 0.0, "max": 2.625045, "mean": 0.01733527020713285, "median": 0.0, "std": 0.13652050748121808}, "within_group_pIC50_range_distribution": {"count": 6421, "missing": 0, "nonfinite": 0, "min": 0.0, "max": 5.712879999999999, "mean": 0.04813544307740227, "median": 0.0, "std": 0.3553983413799034}}`
- Group-level statistics and replicate counts are in `duplicate_analysis.csv` and `replicate_count_distribution.csv`.
- The 20 largest within-InChIKey pIC50 ranges are in `high_variability_molecules.csv`.
- Exact duplicate signature counts are in `exact_duplicate_checks.csv`.

## 6. Molecular Identity Audit

- Identity summary: `{"unique_cleaned_SMILES": 6471, "unique_InChIKeys": 6471, "InChIKeys_with_multiple_cleaned_SMILES": 0, "cleaned_SMILES_with_multiple_InChIKeys": 0}`
- Suspicious multi-mappings, if present, are in `identity_mappings.csv`.

## 7. Structure Sanity

- Structure checks: `{"identity_check_stage": "after salt/SMILES cleaning and InChIKey generation, before downstream filters", "unique_cleaned_SMILES": 6471, "unique_InChIKeys": 6471, "InChIKeys_with_multiple_cleaned_SMILES": 0, "cleaned_SMILES_with_multiple_InChIKeys": 0, "final_unique_InChIKeys_after_current_filters": 4571, "failed_RDKit_parse": 1, "empty_or_invalid_cleaned_SMILES": 1, "zero_atom_molecules": 0, "disconnected_molecules_after_salt_removal": 90, "unusual_element_rows": 0, "atom_count_min": 4, "atom_count_median": 28.0, "atom_count_max": 77, "atom_count_over_100": 0}`
- Unusual elements are reported in `structure_warnings.csv`; no structures were removed by the audit.

## 8. Feature Sanity

- Morgan: `{"molecules": 4571, "columns": 1024, "expected_columns": 1024, "dtype": "uint8", "missing": 0, "nonfinite": 0, "all_zero_fingerprints": 0, "duplicate_fingerprints_excess": 479, "unique_fingerprints": 4092, "average_active_bits": 50.461387005031725, "min_active_bits": 11, "max_active_bits": 102, "configuration_matches_radius_2_size_1024": true}`
- Descriptor count: 14; descriptors: `MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, MaxPartialCharge, MinPartialCharge, NumHeteroatoms, NumRotatableBonds, FractionCSP3, NumAromaticRings, RingCount, NumAliphaticNitrogens, NumFormalCharge`
- Per-feature ranges and missing/nonfinite counts are in `feature_sanity.csv`.
- Exact model columns are in `model_feature_columns.csv`.

## 9. Leakage Checks

- Target leakage: **PASS**; no target-named or directly target-equal selected features found.
- Identifier leakage: **PASS**; none.

## 10. Scaffold Split Audit

- Allocation: Scaffolds are currently ordered by group size before allocation.
- Scaffolds train/test: 1191 / 915
- Molecules train/test: 3656 / 915
- Scaffold overlap: 0 []
- Molecule overlap: 0 InChIKeys, 0 cleaned SMILES.
- Scaffold size summary: `{"largest_scaffold_size": 73, "median_scaffold_size": 1.0, "mean_scaffold_size": 2.1704653371320037, "train_fraction": 0.7998249835922118, "test_fraction": 0.20017501640778823, "train_scaffold_size_mean": 10.24562363238512, "train_scaffold_size_median": 6.0, "test_scaffold_size_mean": 1.0, "test_scaffold_size_median": 1.0, "largest_train_scaffold": 73, "largest_test_scaffold": 1, "largest_1_percent_scaffolds_molecule_fraction": 0.12513673156858454, "largest_5_percent_scaffolds_molecule_fraction": 0.3299059286808138, "largest_10_percent_scaffolds_molecule_fraction": 0.45766790636622184}`
- Per-scaffold sizes and per-molecule assignments are in `scaffold_analysis.csv` and `scaffold_assignments.csv`.

## 11. Train/Test Distribution

Train and test pIC50 statistics and quantiles are in `split_target_distribution.csv`.

## 12. Findings

### PASS

- **standard relation filter recorded**: Current pipeline retains only relation '='; 892 raw records have a different relation or null.
- **finite model target**: Final pIC50 has no missing or nonfinite values.
- **scaffold_split_overlap**: No scaffold overlap between train and test.
- **molecule_split_overlap**: No InChIKey or cleaned SMILES overlap between train and test.
- **Morgan fingerprint dimension**: Generated Morgan matrix has 1024 columns using radius 2 and fpSize 1024.
- **identifier leakage**: Selected model features exclude molecular, assay, target, record, and scaffold identifiers.
- **target leakage**: Selected numeric feature names and values show no direct pIC50 feature match.

### WARNING

- **chemical structure sanity**: Found parse failures=1, disconnected structures=90, zero-atom structures=0, unusual-element rows=0.

### CRITICAL

None.

## 13. Recommended Next Actions

- Review the exported structure_warnings.csv and source structures; audit did not remove rows.
