<div align="center">

### **ஓம் சரவணபவ**

</div>


<div align="center">
      <img src="assets/archbannertop.png" width="900"/>
</div>

<div align="center">

## 🧬 AChE Molecular Machine Learning

### Structure-aware prediction of acetylcholinesterase inhibitory potency


<br>

![Domain](https://img.shields.io/badge/Domain-Computational%20Chemistry-243B53?style=for-the-badge)
![Cheminformatics](https://img.shields.io/badge/Cheminformatics-RDKit-2E7D32?style=for-the-badge)
![Target](https://img.shields.io/badge/Target-AChE%20%7C%20CHEMBL220-6A4C93?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-pIC50%20Regression-1565C0?style=for-the-badge)
![GNN](https://img.shields.io/badge/GNN-Implemented-D97706?style=for-the-badge)
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
- [Graph Neural Network](#graph-neural-network)
        - [Molecular Graph](#molecular-graph)
        - [Message Passing](#message-passing)
        - [Graph Readout and Prediction](#graph-readout-and-prediction)
        - [Architecture and Training](#architecture-and-training)
        - [Matched Results](#matched-results)
        - [Reproducibility Artifacts](#reproducibility-artifacts)
- [🛠️ Technology Stack](#️-technology-stack)
- [🚧 Current Status](#-current-status)
- [🔭 Future Direction](#-future-direction)
- [📚 Technical Documentation](#-technical-documentation)
- [📖 Citation](#-citations)
- [👤 Author](#-author)

---

## 🔬 Project Overview

This project develops a **structure-aware molecular machine-learning pipeline** for predicting the inhibitory potency of compounds against **acetylcholinesterase (AChE)**.

The current implementation establishes a classical QSAR/ML benchmark using:

- ChEMBL bioactivity data
- Molecular standardization and quality control
- IC50 normalization and pIC50 transformation
- Morgan circular fingerprints
- RDKit molecular descriptors
- Scaffold-based molecular partitioning
- Grouped cross-validation
- Random Forest
- XGBoost
- LightGBM
- Optuna hyperparameter optimization
- Multi-metric evaluation and residual analysis

The project is subsequently already extended to engineering molecular representations toward **Graph Neural Networks (GNNs)**..

> **Research direction:** move from fixed molecular feature representations toward increasingly structure-aware and scientifically constrained molecular learning.

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
| Can graph representations improve molecular learning? | GNN extension |


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

Activity records are retrieved for the AChE target with available IC50 measurements.

Raw activity data are processed before being passed to the learning algorithms.

---

## 🧹 Data Preparation

The preprocessing pipeline converts raw bioactivity records into a consistent molecular-level regression dataset.

### Processing flow

```text
ChEMBL Activity Records
          ↓
Data Validation
          ↓
Salt Removal
          ↓
Molecular Identity / InChIKey
          ↓
Duplicate Handling
          ↓
IC50 Unit Normalization
          ↓
pIC50 Transformation
          ↓
Replicate Aggregation
          ↓
Activity Filtering
          ↓
Final Molecular Dataset
```

### Molecular cleaning

The current pipeline performs:

- SMILES validation
- salt removal
- canonical molecular representation
- InChIKey generation
- duplicate handling
- activity-value validation

### IC50 normalization

Supported source representations are normalized to a common concentration representation before pIC50 calculation.

$$
pIC_{50}=-\log_{10}(IC_{50}[M])
$$

### Replicate measurements

Multiple measurements corresponding to the same molecular identity are grouped using `InChIKey`.

The current implementation calculates the median pIC50 and uses group-level median absolute deviation as a quality-control criterion.

This step is treated as part of the **dataset definition**, rather than as a modeling operation.

---

## 🧬 Molecular Representations

Two complementary engineered representations form the current classical ML feature space.

### 1. Morgan Fingerprints

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

### 2. RDKit Molecular Descriptors

The current descriptor representation captures molecular physicochemical and structural properties including:

- Molecular weight
- MolLogP
- H-bond donors
- H-bond acceptors
- TPSA
- Partial-charge descriptors
- Heteroatom count
- Rotatable bonds
- Fraction Csp3
- Aromatic ring count
- Ring count
- Aliphatic nitrogen count
- Formal charge

### Feature configurations

| Configuration | Input |
|---|---|
| Morgan | 1024-bit Morgan fingerprint |
| RDKit | Molecular descriptors |
| Combined | Morgan + RDKit descriptors |

---

## 🧩 Scaffold-Aware Partitioning

A central component of the project is **structure-aware evaluation**.

Random molecular splits can distribute closely related compounds between training and test sets. This can make a model appear to generalize when it is benefiting from structurally similar compounds during training.

The pipeline therefore groups molecules according to their **Bemis–Murcko scaffold** and assigns scaffold groups to dataset partitions.

```text
Molecules
    ↓
Murcko Scaffold
    ↓
Scaffold Groups
    ↓
Training / Test Partition
```

> Molecules sharing a scaffold are kept within the same partition.

<div align="center">

<img src="assets/splits.png" width="900"/>

</div>

---

## 🤖 Classical Machine Learning

The current benchmark consists of three tree-based ensemble models.

| Model | Role |
|---|---|
| **Random Forest** | Non-parametric ensemble baseline |
| **XGBoost** | Gradient-boosted tree model |
| **LightGBM** | Gradient-boosted tree model |

Each model is evaluated across the same molecular representation configurations:

```text
                 Morgan     RDKit     Combined
                    │          │          │
Random Forest       ✓          ✓          ✓
XGBoost             ✓          ✓          ✓
LightGBM            ✓          ✓          ✓
```

This provides a controlled comparison between **model family** and **molecular representation**.

---

## ⚙️ Hyperparameter Optimization

Hyperparameter selection is performed using **Optuna**.

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
Best Hyperparameters
      ↓
Final Training
```

The current implementation uses **3-fold GroupKFold**, with scaffold identity used as the grouping variable.

The optimization objective is based on mean absolute error.

---

## 📏 Evaluation

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

---

## 📊 Experimental Design

### Representation comparison

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

### Validation hierarchy

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
       Grouped 3-Fold CV
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

The held-out test partition is reserved for final evaluation rather than hyperparameter selection.

---


## Graph Neural Network

Classical models receive fixed fingerprints or descriptor vectors. The GNN
represents each molecule as a **chemical graph** and learns atom embeddings by
passing bond-conditioned messages between neighboring atoms. In this study the
learned graph representation is combined with molecular descriptors, so the
evaluated model is a **graph-plus-descriptor model**, not a graph-only model.

The GNN uses the same scaffold-held-out test partition as the matched classical
comparison described in [Experimental Design](#-experimental-design). Its
implementation and saved outputs are linked under
[Reproducibility Artifacts](#reproducibility-artifacts).

### Molecular Graph

For a molecule, the graph is $G=(V,E)$, where $V$ is the set of atoms and $E$
is the set of bonds. The implementation encodes each atom with seven categorical
features:

| Atom feature | Encoded information |
|---|---|
| Atomic number | Element identity |
| Degree | Number of directly bonded atoms |
| Formal charge | Charge, clipped to a supported range and shifted to a nonnegative category |
| Hybridization | Supported RDKit hybridization states |
| Aromaticity | Whether the atom is aromatic |
| Hydrogen count | Number of attached hydrogens |
| Ring membership | Whether the atom belongs to a ring |

Each bond has **13 edge channels**: four one-hot bond types (single, double,
triple, aromatic), three bond properties (conjugated, aromatic, ring), and six
one-hot stereochemical states. Every bond is added in both directions so
messages can travel between either pair of neighboring atoms.

### Message Passing

Each of the seven atom categories is mapped through its own embedding layer.
Those embeddings are concatenated and projected into a **128-dimensional**
hidden state. At each of three message-passing steps, the source atom state is
combined with the 13 bond channels and transformed into a message. Incoming
messages are summed and divided by the destination atom's degree, then passed
to a GRU-based update. A residual connection, dropout, and layer normalization
produce the next atom state.

```text
Atom categories ──> embeddings ──> node projection
                                           │
Bond channels ──> edge-conditioned messages
                                           │
                   degree-normalized aggregation
                                           │
                      GRU update + residual
                                           │
                    dropout + layer normalization
                                           │
                          repeat 3 times
```

### Graph Readout and Prediction

After message passing, node states are pooled per molecule using both **mean**
and **max** pooling. The two graph-level vectors are concatenated with **14
RDKit descriptors**. Descriptor imputation and standardization are fitted on
the training partition only; target values are also standardized for training
and predictions are returned to the original pIC50 scale.

The combined vector is passed through a multilayer regression head with ReLU
activations and dropout to predict pIC50. This makes the role of the two input
sources explicit:

```text
Molecular graph ──> message passing ──> mean + max pooled graph vector ──┐
                                                                                  ├─> regression head ──> pIC50
14 RDKit descriptors ──> train-fitted imputation / scaling ────────────┘
```

### Architecture and Training

<p align="center">
  <img src="assets/archclean.png" alt="Graph neural network architecture showing atom and bond features, message passing, graph pooling, descriptor fusion, and pIC50 prediction" width="100%">
</p>

| Component | Configuration |
|---|---|
| Atom features | 7 categorical channels |
| Bond features | 13 channels; each bond is bidirectional |
| Global molecular descriptors | 14 RDKit descriptors |
| Hidden size | 128 |
| Message-passing steps | 3 |
| Graph pooling | Mean + max |
| Node update | GRU cell with residual connection |
| Regularization | Dropout 0.15 + LayerNorm |
| Regression loss | Smooth L1 |
| Optimizer | AdamW; learning rate 0.001; weight decay 0.0001 |
| Batch size | 64 |
| Gradient clipping | 5.0 |
| Early stopping | Validation MAE; patience 20 epochs |
| Random seed | 50 |
| Hardware | CPU |

The scaffold-disjoint partitions contain 3,108 training, 548 validation, and
915 test molecules. The best validation MAE was **0.6507** at epoch **48**;
early stopping ended training after **68 epochs**. No molecules were rejected
during graph construction for this run.

### Matched Results

The table reports held-out metrics from the matched audit. Every model is
evaluated on the same 915-molecule test partition; the test set is excluded
from hyperparameter selection and training.

| Model | Representation | Test MAE | RMSE | R² | Pearson r |
|---|---|---:|---:|---:|---:|
| **XGBoost** | Morgan + RDKit | **0.6352** | 0.8347 | 0.4553 | 0.6795 |
| XGBoost | Morgan | 0.6510 | 0.8519 | 0.4326 | 0.6648 |
| Random Forest | Morgan + RDKit | 0.6764 | 0.8507 | 0.4341 | 0.6644 |
| Random Forest | Morgan | 0.6852 | 0.8631 | 0.4175 | 0.6506 |
| **GNN** | Graph + RDKit | **0.6908** | 0.9093 | 0.3535 | 0.6225 |
| XGBoost | RDKit | 0.6969 | 0.8892 | 0.3818 | 0.6188 |
| Random Forest | RDKit | 0.7169 | 0.8978 | 0.3698 | 0.6140 |

On this split, the best classical model's MAE is **0.0556 pIC50** lower than
the GNN's. This is a result for this dataset, scaffold split, graph encoding,
descriptor augmentation, and training configuration; it is not evidence that
GNNs are generally inferior. The audit recommends repeated scaffold splits or
an external test set for a more stable estimate.

### Reproducibility Artifacts

- [GNN implementation](src/GNNmodel.py)
- [Architecture figure](assets/archclean.png)
- [Matched classical/GNN audit report](output/audit/GNN_classical_comparison/audit_report.md)
- [GNN run metrics](output/logs/GNN_metrics.csv)
- [Training history](output/logs/GNN_training_history.csv)
- [Held-out test predictions](output/logs/GNN_test.csv)
- [All GNN predictions](output/logs/GNN_predictions.csv)

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
| Graph ML  | PyTorch · PyTorch Geometric |

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
| Grouped CV | ✅ |
| Multi-metric evaluation | ✅ |
| Residual analysis | ✅ |
| Dataset / validation audit |  ✅ |
| GNN |  ✅ |
---

## 🔭 Future Direction

The project progresses through increasingly structured molecular representations:

```text
Engineered Features
        ↓
Classical Molecular ML
        ↓
Molecular Graphs
        ↓
Graph Neural Networks
        ↓
Physics-Informed GNNs
        ↓
Structure-aware Molecular Machine Learning
```

The longer-term direction is to connect:

**molecular structure → biological activity → mechanistic understanding**

rather than optimizing predictive performance in isolation.

---

## 📚 Technical Documentation

A separate technical report contains the detailed methodology, data-processing decisions, feature definitions, model configurations, hyperparameter spaces, validation analysis, GNN methodology, and experimental results.

The complete research record and archived software release are preserved through Zenodo.

---

## 📖 Citation

If you use this repository, its methodology, implementation, or derived results in academic or scientific work, please cite the archived software release:

**Venkataramanan, D. (2026).**  
*AChE Molecular Machine Learning: Structure-Aware Prediction of Acetylcholinesterase Inhibitory Potency.*  
Zenodo.  
https://doi.org/10.5281/zenodo.23191332

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

**கணக்கீட்டு உயிரியல் • வேதியியல் தகவல் புள்ளியியல் • மூலக்கூறு இயந்திரக் கற்றல் • கணக்கீட்டு மருந்து கண்டுபிடிப்பு**

a.k.a 

### **Darshan Venkataramanan**

**Computational Biology · Cheminformatics · Molecular Machine Learning · Computational Drug Discovery**

<br>

*Building from molecular representations toward scientifically grounded learning.*

</div>

---
