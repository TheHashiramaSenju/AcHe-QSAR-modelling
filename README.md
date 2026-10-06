
<div align="center">

### **ஓம் சரவணபவ**

</div>

<div align="center">
  <img src="assets/archbannertop.png" width="900"/>
</div>

<div align="center">

# 🧬 AChE Molecular Machine Learning

### Structure-aware prediction of acetylcholinesterase inhibitory potency

<br>

![Domain](https://img.shields.io/badge/Domain-Computational%20Chemistry-243B53?style=for-the-badge)
![Cheminformatics](https://img.shields.io/badge/Cheminformatics-RDKit-2E7D32?style=for-the-badge)
![Target](https://img.shields.io/badge/Target-AChE%20%7C%20CHEMBL220-6A4C93?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-pIC50%20Regression-1565C0?style=for-the-badge)
![GNN](https://img.shields.io/badge/GNN-Evaluated-D97706?style=for-the-badge)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23191332.svg)](https://doi.org/10.5281/zenodo.23191332)

<br><br>

</div>

---

## 📑 Contents

- [🔬 Project Overview](#-project-overview)
- [🎯 Objective](#-objective)
- [🧪 Dataset](#-dataset)
- [🧹 Data Preparation](#-data-preparation)
- [🧬 Molecular Representations](#-molecular-representations)
- [🧩 Scaffold-Aware Partitioning](#-scaffold-aware-partitioning)
- [🤖 Classical Machine Learning](#-classical-machine-learning)
- [⚙️ Hyperparameter Optimization](#️-hyperparameter-optimization)
- [📏 Evaluation](#-evaluation)
- [📊 Experimental Design](#-experimental-design)
- [🧠 Graph Neural Network](#-graph-neural-network)
- [📈 GNN Architecture](#-gnn-architecture)
- [🔬 Matched Classical vs GNN Evaluation](#-matched-classical-vs-gnn-evaluation)
- [⚠️ Methodological Notes](#️-methodological-notes)
- [🛠️ Technology Stack](#️-technology-stack)
- [🚧 Current Status](#-current-status)
- [🔭 Future Direction](#-future-direction)
- [📚 Technical Documentation](#-technical-documentation)
- [📖 Citation](#-citation)
- [👤 Author](#-author)

---

## 🔬 Project Overview

This project develops a **structure-aware molecular machine-learning pipeline**
for predicting the inhibitory potency of compounds against
**acetylcholinesterase (AChE)**.

The workflow begins with ChEMBL activity data and progresses from engineered
molecular representations to explicit molecular graph representations.

The classical benchmark includes:

- ChEMBL bioactivity data
- Molecular standardization and quality control
- IC50 normalization and pIC50 transformation
- Morgan circular fingerprints
- RDKit molecular descriptors
- Bemis–Murcko scaffold grouping
- Scaffold-aware train/test partitioning
- Grouped cross-validation
- Random Forest
- XGBoost
- LightGBM
- Optuna hyperparameter optimization
- Multi-metric evaluation
- Residual and diagnostic analysis

The project is subsequently extended with a **Graph Neural Network (GNN)**
evaluation using explicit molecular graph representations.

The GNN is evaluated under the same fixed scaffold-held-out test partition
used in the matched classical comparison.

> **Research direction:** move from fixed molecular feature representations
> toward increasingly structure-aware and scientifically constrained molecular
> learning.

---

## 🎯 Objective

### Primary Task

Predict continuous **pIC50** values from molecular structure.

$$
pIC_{50} = -\log_{10}(IC_{50}[M])
$$

The learning problem is formulated as:

```text
Molecular Structure
        ↓
Molecular Representation
        ↓
Machine Learning Model
        ↓
Predicted pIC50
```

### Core Questions

| Question | Purpose |
|---|---|
| How informative are circular molecular fingerprints? | Evaluate local structural representation |
| What additional information do molecular descriptors provide? | Evaluate physicochemical representation |
| Does combining both representations improve prediction? | Compare complementary feature spaces |
| How does structural separation affect performance? | Test molecular generalization |
| How do classical models compare under the same representation? | Establish reliable baselines |
| Can graph representations improve molecular learning? | Evaluate an explicit molecular graph representation |

---

## 🧪 Dataset

| Property | Configuration |
|---|---|
| Data source | ChEMBL |
| Target | Acetylcholinesterase |
| ChEMBL Target ID | `CHEMBL220` |
| Activity endpoint | `IC50` |
| Learning target | `pIC50` |
| Task | Continuous regression |

The raw activity dataset contains **8,793 records** associated with
CHEMBL220.

Following molecular and activity curation, the final modelling dataset contains
**4,571 unique molecular identities by InChIKey**.

The final target distribution is:

| Property | Value |
|---|---:|
| Minimum pIC50 | 4.9208 |
| Maximum pIC50 | 10.9606 |
| Mean pIC50 | 6.5152 |
| Median pIC50 | 6.3010 |
| Standard deviation | 1.1267 |

---

## 🧹 Data Preparation

The preprocessing pipeline converts raw bioactivity records into a consistent
molecular-level regression dataset.

### Processing Flow

```text
ChEMBL Activity Records
          ↓
Data Validation
          ↓
Salt / Structure Processing
          ↓
Molecular Identity / InChIKey
          ↓
Duplicate and Replicate Handling
          ↓
IC50 Unit Normalization
          ↓
pIC50 Transformation
          ↓
Activity Quality Control
          ↓
Relation Filtering
          ↓
IC50 Threshold
          ↓
Final Molecular Dataset
```

### Molecular Cleaning

The pipeline performs:

- SMILES validation
- molecular structure processing
- salt removal
- canonical molecular representation
- InChIKey generation
- duplicate handling
- activity-value validation

### IC50 Normalization

Supported source units are converted into a common concentration scale before
pIC50 calculation.

The normalized activity is represented in nanomolar concentration:

$$
IC50_{nM}
$$

and pIC50 is calculated as:

$$
pIC_{50}=-\log_{10}(IC_{50}[M])
$$

### Replicate Measurements

Multiple measurements corresponding to the same molecular identity are grouped
using `InChIKey`.

The implementation calculates the median pIC50 and uses group-level median
absolute deviation as a quality-control criterion.

This step is treated as part of the **dataset definition**, rather than as a
modelling operation.

### Curation Audit

The documented curation stages are:

| Stage | Rows |
|---|---:|
| Raw ChEMBL records | 8,793 |
| Initial validity / null handling | 8,072 |
| InChIKey generation | 8,072 |
| Potential duplicate removal | 7,248 |
| Supported-unit filtering | 7,185 |
| Group MAD filtering | 7,110 |
| Exact duplicate removal | 6,387 |
| Equality-relation filtering | 5,695 |
| Final IC50 threshold | 4,571 |

The repository preserves the associated audit artifacts and intermediate
research record.

---

## 🧬 Molecular Representations

Two engineered representations form the classical molecular feature space.

### 1. Morgan Fingerprints

Morgan fingerprints encode circular atom environments around each atom.

| Parameter | Value |
|---|---:|
| Representation | Morgan circular fingerprint |
| Radius | `2` |
| Fingerprint size | `1024 bits` |

```text
Molecular Structure
       ↓
Circular Atom Environments
       ↓
Morgan Fingerprint
       ↓
1024-bit Vector
```

The final dataset contains **4,092 unique fingerprint vectors**, with a mean
of approximately **50.5 active bits per molecule**.

### 2. RDKit Molecular Descriptors

The descriptor representation contains **14 molecular descriptors**:

| Descriptor |
|---|
| Molecular weight |
| MolLogP |
| Number of H-bond donors |
| Number of H-bond acceptors |
| TPSA |
| Maximum partial charge |
| Minimum partial charge |
| Number of heteroatoms |
| Number of rotatable bonds |
| Fraction Csp3 |
| Number of aromatic rings |
| Ring count |
| Number of aliphatic nitrogens |
| Formal charge |

### Feature Configurations

| Configuration | Input |
|---|---|
| Morgan | 1024-bit Morgan fingerprint |
| RDKit | 14 molecular descriptors |
| Combined | 1024-bit Morgan + 14 RDKit descriptors |

Therefore, the combined classical representation contains:

$$
1024 + 14 = 1038
$$

input features.

---

## 🧩 Scaffold-Aware Partitioning

A central component of the project is **structure-aware evaluation**.

Random molecular splits can distribute structurally related compounds across
training and test sets. This can make generalization appear stronger when
closely related molecules are present in both partitions.

The pipeline therefore groups molecules according to their
**Bemis–Murcko scaffold** and assigns complete scaffold groups to partitions.

```text
Molecules
    ↓
Bemis–Murcko Scaffold
    ↓
Scaffold Groups
    ↓
Training / Test Partition
```

> Molecules belonging to the same scaffold group are kept within the same
> partition.

### Original Classical Split

The original benchmark uses:

| Partition | Molecules |
|---|---:|
| Training | 3,656 |
| Held-out test | 915 |

The partition contains:

- **0 scaffold overlap**
- **0 InChIKey overlap**
- **0 cleaned-SMILES overlap**

<div align="center">

<img src="assets/splits.png" width="900"/>

</div>

### Important Structural Note

The scaffold distribution is highly fragmented.

The held-out test set contains **915 scaffolds for 915 molecules**, meaning the
test partition is dominated by singleton scaffolds.

This is retained as part of the documented experimental design and should be
considered when interpreting generalization performance.

---

## 🤖 Classical Machine Learning

The classical benchmark consists of three tree-based ensemble models.

| Model | Role |
|---|---|
| **Random Forest** | Non-parametric ensemble baseline |
| **XGBoost** | Gradient-boosted tree model |
| **LightGBM** | Gradient-boosted tree model |

Each model is evaluated across the same molecular representation
configurations:

```text
                 Morgan     RDKit     Combined
                    │          │          │
Random Forest       ✓          ✓          ✓
XGBoost             ✓          ✓          ✓
LightGBM            ✓          ✓          ✓
```

This provides a controlled comparison between **model family** and
**molecular representation**.

### Original Classical Benchmark

The original classical implementation uses **5-fold GroupKFold**, with
scaffold identity used as the grouping variable.

The strongest saved original benchmark result was obtained using XGBoost with
the combined Morgan + RDKit representation:

| Model | Representation | Test MAE | RMSE | R² | Pearson r |
|---|---|---:|---:|---:|---:|
| **XGBoost** | Morgan + RDKit | **0.6045** | 0.7981 | 0.5020 | 0.7109 |
| LightGBM | Morgan + RDKit | 0.6312 | 0.8223 | 0.4712 | 0.6885 |
| Random Forest | Morgan + RDKit | 0.6985 | 0.8718 | 0.4058 | 0.6476 |

These values belong to the **original classical benchmark**.

A separate matched audit was subsequently performed to ensure that the
classical and GNN models were compared under a common controlled evaluation
protocol.

---

## ⚙️ Hyperparameter Optimization

Hyperparameter selection is performed using **Optuna**.

Conceptually:

```text
Training Data
      ↓
Scaffold-Grouped Cross-Validation
      ↓
Model Evaluation
      ↓
Validation MAE
      ↓
Optuna Trials
      ↓
Selected Hyperparameters
      ↓
Final Model Training
      ↓
Held-Out Evaluation
```

The original classical benchmark uses **5-fold GroupKFold**, with scaffold
identity as the grouping variable.

For the later matched classical-vs-GNN audit, a separate controlled
comparison uses:

- **3-fold GroupKFold**
- scaffold identity as the grouping variable
- **8 Optuna trials per model / representation configuration**
- the fixed held-out test partition excluded from tuning

These are two distinct experimental stages and should not be conflated.

---

## 📏 Evaluation

The models are evaluated using multiple complementary metrics.

| Metric | Interpretation |
|---|---|
| **MAE** | Mean absolute prediction error |
| **RMSE** | Error with stronger penalty for large deviations |
| **R²** | Variance explained relative to the target mean |
| **Pearson r** | Linear association |
| **Spearman ρ** | Rank-order association |

Residuals are calculated as:

$$
e_i = y_i - \hat{y}_i
$$

This enables analysis of:

- systematic prediction bias
- high-error compounds
- activity-range dependence
- residual distributions
- prediction calibration
- model failure regions

The repository also contains diagnostic plots and saved evaluation outputs.

---

## 📊 Experimental Design

### Classical Representation Comparison

```text
Morgan
   │
   ├── Random Forest
   ├── XGBoost
   └── LightGBM

RDKit Descriptors
   │
   ├── Random Forest
   ├── XGBoost
   └── LightGBM

Morgan + RDKit
   │
   ├── Random Forest
   ├── XGBoost
   └── LightGBM
```

### Original Classical Validation

```text
                Full Dataset
                     │
                     ▼
          Scaffold-Based Split
              ┌──────┴──────┐
              ▼             ▼
           Training        Test
              │
              ▼
        5-Fold GroupKFold
              │
              ▼
       Optuna Optimization
              │
              ▼
       Final Model Training
              │
              ▼
        Held-Out Evaluation
```

### Matched Classical-vs-GNN Audit

The later controlled comparison uses the same fixed scaffold-held-out test
partition while reserving a validation subset from the training partition.

```text
                 Curated Dataset
                       │
                       ▼
             Fixed Scaffold Split
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Training     Validation      Test
        3108           548           915
          │             │             │
          ▼             │             │
     3-Fold GroupKFold  │             │
          │             │             │
          ▼             │             │
     Optuna / Model     │             │
       Selection        │             │
          │             │             │
          └──────┬──────┘             │
                 ▼                    │
        Final Model Evaluation ───────┘
```

The held-out test partition is excluded from hyperparameter tuning and model
selection.

---

# 🧠 Graph Neural Network

The classical models represent molecules using fixed engineered features such
as Morgan fingerprints and RDKit descriptors.

The GNN stage instead represents each molecule explicitly as a
**chemical graph**, allowing the model to learn structure-dependent
representations through message passing.

The GNN was evaluated as a separate model family under the same
**fixed scaffold-held-out test partition** used in the matched classical
comparison.

---

## Molecular Graph

Each molecule is represented as:

$$
G=(V,E)
$$

where:

- $V$ represents atoms
- $E$ represents chemical bonds

The graph representation contains:

- **7 atom-level feature channels**
- **13 bond-level feature channels**
- **14 global RDKit molecular descriptors**

### Node Features

Each atom is represented using:

| Feature | Description |
|---|---|
| Atomic number | Element identity |
| Degree | Number of directly connected atoms |
| Formal charge | Charge state |
| Hybridization | Hybridization state |
| Aromaticity | Aromatic atom indicator |
| Hydrogen count | Number of attached hydrogens |
| Ring membership | Whether the atom belongs to a ring |

### Edge Features

Bond representations contain **13 channels**, including:

- single bond
- double bond
- triple bond
- aromatic bond
- conjugation
- aromaticity
- ring membership
- stereochemical configuration

The stereochemical channels distinguish the supported stereochemical states.

Each molecular bond is represented in both directions for message passing.

---

## Message Passing

The graph network projects the categorical node and edge representations into a
shared hidden representation and performs **three message-passing steps**.

Conceptually:

```text
Atom Features + Bond Features
             ↓
      Feature Projection
             ↓
      Message Passing × 3
             ↓
       GRU-based Updates
             ↓
     Residual Connections
             ↓
     Dropout + LayerNorm
             ↓
       Graph Representation
```

At each message-passing step, neighboring atoms contribute information through
their connecting bond features.

Messages are aggregated at the destination atom and normalized by destination
degree before the node representation is updated.

A GRU-based update mechanism is used together with residual connections,
dropout, and layer normalization.

---

## Graph Readout

After message passing, node representations are converted into a
molecule-level representation using:

- **mean pooling**
- **max pooling**

The two pooled representations are combined and concatenated with the
**14 RDKit molecular descriptors**.

Therefore, the final GNN is a:

> **Graph + descriptor model**

rather than a graph-only model.

```text
                 Molecular Graph
                       │
                       ▼
               Message Passing
                       │
                       ▼
                Node Embeddings
                       │
                 ┌─────┴─────┐
                 ▼           ▼
             Mean Pooling  Max Pooling
                 │           │
                 └─────┬─────┘
                       │
                       ▼
                Graph Features
                       │
                       ├───────────────┐
                       │               │
                       ▼               ▼
               Graph Features    14 RDKit Descriptors
                       │               │
                       └───────┬───────┘
                               ▼
                         Regression Head
                               │
                               ▼
                         Predicted pIC50
```

---

# 📈 GNN Architecture

<div align="center">

<img src="assets/archclean.png" width="900"/>

</div>

### Architecture Configuration

| Component | Configuration |
|---|---|
| Atom feature channels | 7 |
| Bond feature channels | 13 |
| Global descriptors | 14 |
| Hidden representation | 128 |
| Message-passing steps | 3 |
| Graph pooling | Mean + Max |
| Node update | GRU-based |
| Normalization | LayerNorm |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | Smooth L1 |
| Gradient clipping | 5.0 |
| Early stopping patience | 20 epochs |
| Random seed | 50 |
| Training hardware | CPU |

Descriptor standardization and missing-value imputation are fitted using the
training partition to avoid using validation or test information during
preprocessing.

The GNN training used:

| Partition | Molecules |
|---|---:|
| Training | 3,108 |
| Validation | 548 |
| Held-out test | 915 |

The best validation MAE was **0.6507**, reached at epoch 48.

Training stopped after **68 epochs** under the configured early-stopping
criterion.

---

# 🔬 Matched Classical vs GNN Evaluation

The most controlled comparison in the repository uses the same fixed
scaffold-held-out test partition for both classical and graph-based models.

The matched audit uses:

- the same training/test molecular partition
- the same held-out test set
- scaffold-aware grouping
- training-only preprocessing
- no test-set hyperparameter selection
- 3-fold GroupKFold for classical tuning
- 8 Optuna trials per classical model / representation

### Matched Results

| Model | Representation | Test MAE | RMSE | R² | Pearson r |
|---|---|---:|---:|---:|---:|
| **XGBoost** | Morgan + RDKit | **0.6352** | 0.8347 | 0.4553 | 0.6795 |
| XGBoost | Morgan | 0.6510 | 0.8519 | 0.4326 | 0.6648 |
| Random Forest | Morgan + RDKit | 0.6764 | 0.8507 | 0.4341 | 0.6644 |
| Random Forest | Morgan | 0.6852 | 0.8631 | 0.4175 | 0.6506 |
| **GNN** | Graph + RDKit | **0.6908** | 0.9093 | 0.3535 | 0.6225 |
| XGBoost | RDKit | 0.6969 | 0.8892 | 0.3818 | 0.6188 |
| Random Forest | RDKit | 0.7169 | 0.8978 | 0.3698 | 0.6140 |

Under this matched evaluation, the best classical model is **XGBoost using
Morgan fingerprints + RDKit descriptors**.

Its held-out MAE is:

$$
MAE = 0.6352
$$

The GNN achieves:

$$
MAE = 0.6908
$$

The difference is approximately:

$$
0.6908 - 0.6352 = 0.0556
$$

pIC50 units.

### Interpretation

The GNN therefore **did not outperform the strongest classical baseline under
this experimental design**.

This result should be interpreted as a finding specific to:

- the curated AChE dataset
- the scaffold partition
- the molecular graph construction
- the descriptor augmentation
- the GNN architecture
- the training configuration
- the optimization budget

It should **not** be interpreted as evidence that graph neural networks are
intrinsically inferior to classical molecular representations.

> **Key result:** the GNN provides an explicit graph-based molecular
> representation, but under the present experimental design it did not improve
> held-out predictive performance over the strongest classical baseline.

---

## ⚠️ Methodological Notes

The repository contains a separate forensic audit documenting methodological
decisions, validation design, data-processing checks, and identified
limitations.

### IC50 Threshold Audit

The current implementation applies the final activity threshold to the
normalized concentration:

```text
IC50_nM <= 10,000
```

An audit compared this with applying the same numerical threshold directly to
the raw `standard_value`.

For the audited dataset snapshot:

- **0 rows** were classified differently
- **0 unique InChIKeys** were affected

Therefore, although thresholding before normalization would be methodologically
ambiguous across mixed units, the two approaches produced the same retained
cohort for this dataset snapshot.

### Feature-Count Audit

The actual estimator matrices in the audited classical experiments contain:

| Representation | Features |
|---|---:|
| Morgan | 1,024 |
| RDKit descriptors | 14 |
| Combined | 1,038 |

Earlier saved metric artifacts contained inconsistent feature-count metadata.
The audit determined that this logging discrepancy did **not** alter the model
predictions or the underlying feature matrices.

### GNN Scope

The GNN is a **graph + descriptor** model.

It should therefore not be described as a purely graph-only model.

### PiGNN Scope

A Physics-Informed GNN (PiGNN) was considered as a future research direction.

It was **not included as a completed model in the final experimental study**.

Accordingly, no PiGNN performance result is reported in this repository.

### Scientific Scope

The reported models predict **AChE inhibitory potency as pIC50**.

The project does not claim to predict:

- therapeutic efficacy
- clinical response
- toxicity
- pharmacokinetics
- disease outcome
- mechanism of action

The repository is an evidence-bound molecular machine-learning research record
rather than a claim of state-of-the-art QSAR performance.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Bioactivity | ChEMBL |
| Cheminformatics | RDKit |
| Data | NumPy · Pandas · DuckDB |
| Classical ML | scikit-learn · XGBoost · LightGBM |
| Optimization | Optuna |
| Statistics | SciPy |
| Visualization | Matplotlib · Seaborn |
| Experiment Tracking | MLflow |
| Graph ML | PyTorch · PyTorch Geometric |

---

## 🚧 Current Status

| Component | Status |
|---|:---:|
| ChEMBL data retrieval | ✅ |
| AChE / CHEMBL220 dataset | ✅ |
| Molecular cleaning | ✅ |
| IC50 normalization | ✅ |
| pIC50 transformation | ✅ |
| Replicate handling | ✅ |
| Morgan fingerprints | ✅ |
| RDKit descriptors | ✅ |
| Scaffold grouping | ✅ |
| Scaffold-aware split | ✅ |
| Random Forest | ✅ |
| XGBoost | ✅ |
| LightGBM | ✅ |
| Optuna optimization | ✅ |
| Grouped cross-validation | ✅ |
| Multi-metric evaluation | ✅ |
| Residual analysis | ✅ |
| Dataset / validation audit | ✅ |
| GNN evaluation | ✅ |
| Matched classical-vs-GNN audit | ✅ |
| PiGNN | 🔭 Future direction |

---

## 🔭 Future Direction

The project progresses through increasingly structured molecular
representations:

```text
Engineered Molecular Features
            ↓
Classical Molecular ML
            ↓
Explicit Molecular Graphs
            ↓
Graph Neural Networks
            ↓
Physics-Informed Molecular Learning
            ↓
Structure-Aware Molecular Machine Learning
```

A longer-term research direction is to investigate how additional
scientifically motivated constraints and molecular information can be
incorporated into graph-based learning.

The PiGNN direction remains **future work** and is not represented as a
completed experimental result in the current release.

---

## 📚 Technical Documentation

The repository contains additional technical documentation covering:

- data processing
- molecular representations
- model construction
- hyperparameter optimization
- scaffold-aware evaluation
- GNN methodology
- diagnostic analysis
- audit findings
- reproducibility artifacts
- research provenance

The complete scientific report is available as:

**`REPORT.pdf`**

The research record is maintained in:

**`RESEARCH_RECORD.md`**

The methodological audit artifacts are maintained under:

**`output/audit/`**

---

## 📖 Citation

If you use this repository, its methodology, implementation, or derived
results in academic or scientific work, please cite the archived software
release.

**Venkataramanan, D. (2026).**  
*AChE Molecular Machine Learning: Structure-Aware Prediction of
Acetylcholinesterase Inhibitory Potency.*  
Zenodo.

**DOI:** https://doi.org/10.5281/zenodo.23191332

### BibTeX

```bibtex
@software{venkataramanan_ache_qsar_2026,
  author       = {Venkataramanan, Darshan},
  title        = {AChE Molecular Machine Learning: Structure-Aware Prediction of Acetylcholinesterase Inhibitory Potency},
  year         = {2026},
  version      = {1.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23191332},
  url          = {https://doi.org/10.5281/zenodo.23191332}
}
```

---

## 👤 Author

<div align="center">

### **தர்ஷன் வெங்கட்டரமணன்**

**கணக்கீட்டு உயிரியல் · வேதியியல் தகவல் புள்ளியியல் · மூலக்கூறு இயந்திரக்
கற்றல் · கணக்கீட்டு மருந்து கண்டுபிடிப்பு**

<br>

### **Darshan Venkataramanan**

**Computational Biology · Cheminformatics · Molecular Machine Learning ·
Computational Drug Discovery**

<br>

*Building from molecular representations toward scientifically grounded
learning.*

</div>

---

## 📜 Research Record

This repository is maintained as a versioned research record.

- **v1.0.0** — original archived research snapshot
- **v1.0.1** — current documented repository state

The historical v1.0.0 record remains preserved, while v1.0.1 contains
subsequent documentation, provenance, and research-record updates.

**Current archival DOI:**  
https://doi.org/10.5281/zenodo.23191332
```
