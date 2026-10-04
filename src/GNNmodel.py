import argparse
import copy
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

if __package__:
    from .data_cleaning import calculate_molecular_descriptors
    from .data_cleaning import InChIKeyConversion
else:
    from data_cleaning import calculate_molecular_descriptors
    from data_cleaning import InChIKeyConversion

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "FractionCSP3",
    "NumAromaticRings",
    "RingCount",
    "NumAliphaticNitrogens",
    "NumFormalCharge",
]
ATOM_FEATURE_COLUMNS = [
    "atomic_number",
    "degree",
    "formal_charge",
    "hybridization",
    "aromaticity",
    "hydrogen_count",
    "ring_membership",
]
EDGE_FEATURE_COLUMNS = [
    "single_bond",
    "double_bond",
    "triple_bond",
    "aromatic_bond",
    "conjugated",
    "aromatic",
    "ring_membership",
    "stereo_none",
    "stereo_any",
    "stereo_z",
    "stereo_e",
    "stereo_cis",
    "stereo_trans",
]
HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}


@dataclass
class GraphSample:
    source_index: int
    smiles: str
    scaffold: str
    node_categories: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    target: float


def extract_atomic_features(mol):
    features = []
    for atom in mol.GetAtoms():
        features.append(
            [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0),
                float(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                float(atom.IsInRing()),
            ]
        )
    return np.asarray(features, dtype=np.float32).reshape(-1, 7)


def extract_atom_categories(mol):
    features = extract_atomic_features(mol).astype(np.int64)
    if not len(features):
        return features.reshape(0, 7)
    features[:, 0] = np.clip(features[:, 0], 0, 118)
    features[:, 1] = np.clip(features[:, 1], 0, 6)
    features[:, 2] = np.clip(features[:, 2], -5, 5) + 5
    features[:, 3] = np.clip(features[:, 3], 0, 5)
    features[:, 4] = np.clip(features[:, 4], 0, 1)
    features[:, 5] = np.clip(features[:, 5], 0, 6)
    features[:, 6] = np.clip(features[:, 6], 0, 1)
    return features


def one_hot_encode(value, choices):
    return [float(value == choice) for choice in choices]


def extract_edge_features(mol):
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    features = []
    for bond in mol.GetBonds():
        features.append(
            one_hot_encode(bond.GetBondType(), bond_types)
            + [
                float(bond.GetIsConjugated()),
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
            ]
            + one_hot_encode(bond.GetStereo(), stereo_types)
        )
    return np.asarray(features, dtype=np.float32).reshape(-1, 13)


def _resolve_project_root():
    return Path(__file__).resolve().parent.parent


def _resolve_default_dataset():
    csv_dir = _resolve_project_root() / "database" / "csv"
    candidates = [
        csv_dir / "Acetylcholinesterase_Homo_sapiens_cleaned_pipeline",
        csv_dir / "Acetylcholinesterase_Homo_sapiens_cleaned_pipeline.csv",
        csv_dir / "Acetylcholinesterase_Homo_sapiens_cleaned.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _scaffold_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "__INVALID__"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "__NO_SCAFFOLD__"
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    key = InChIKeyConversion(scaffold_smiles)
    return key if key else "__NO_SCAFFOLD__"


def _create_graph_sample(source_index, smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SMILES could not be parsed into a non-empty molecule")

    descriptor_map = calculate_molecular_descriptors(smiles)
    global_features = np.asarray(
        [descriptor_map.get(name, np.nan) for name in DESCRIPTOR_COLUMNS],
        dtype=np.float32,
    )
    finite_mask = np.isfinite(global_features)
    charge_indices = {
        DESCRIPTOR_COLUMNS.index("MaxPartialCharge"),
        DESCRIPTOR_COLUMNS.index("MinPartialCharge"),
    }
    missing_indices = set(np.flatnonzero(~finite_mask))
    if missing_indices.difference(charge_indices) or np.isinf(global_features).any():
        raise ValueError("Molecular descriptors contain unsupported non-finite values")

    bond_features = extract_edge_features(mol)
    edge_pairs = []
    directed_features = []
    for bond_index, bond in enumerate(mol.GetBonds()):
        atom_start = bond.GetBeginAtomIdx()
        atom_end = bond.GetEndAtomIdx()
        edge_pairs.extend([[atom_start, atom_end], [atom_end, atom_start]])
        directed_features.extend([bond_features[bond_index], bond_features[bond_index]])

    edge_index = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2).T
    edge_features = np.asarray(directed_features, dtype=np.float32).reshape(-1, 13)
    return GraphSample(
        source_index=int(source_index),
        smiles=smiles,
        scaffold=_scaffold_key(smiles),
        node_categories=extract_atom_categories(mol),
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
        target=float(target),
    )


def load_graph_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    required_columns = {"cleaned_smiles", "PIC50"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    samples = []
    rejected = []
    for source_index, row in dataset.iterrows():
        smiles = row["cleaned_smiles"]
        target = pd.to_numeric(row["PIC50"], errors="coerce")
        if not isinstance(smiles, str) or not smiles.strip():
            rejected.append({"source_index": source_index, "reason": "missing_smiles"})
            continue
        if not np.isfinite(target):
            rejected.append({"source_index": source_index, "reason": "invalid_PIC50"})
            continue
        try:
            samples.append(_create_graph_sample(source_index, smiles, target))
        except (ValueError, RuntimeError, TypeError) as error:
            rejected.append(
                {"source_index": source_index, "reason": str(error)}
            )

    if len(samples) < 10:
        raise ValueError("Fewer than 10 usable molecular records were found")
    return samples, pd.DataFrame(rejected, columns=["source_index", "reason"])


def _baseline_scaffold_split(samples):
    frame = pd.DataFrame(
        {"sample_index": np.arange(len(samples)), "scaffold": [s.scaffold for s in samples]}
    )
    groups = frame.groupby("scaffold").groups
    ordered_scaffolds = sorted(
        groups, key=lambda scaffold: len(groups[scaffold]), reverse=True
    )
    target_train_size = int(0.8 * len(samples))
    train_indices = []
    test_indices = []
    for scaffold in ordered_scaffolds:
        row_indices = list(groups[scaffold])
        if len(train_indices) < target_train_size:
            train_indices.extend(row_indices)
        else:
            test_indices.extend(row_indices)
    if not train_indices or not test_indices:
        raise ValueError("Scaffold partitioning did not produce train and test sets")
    return train_indices, test_indices


def _validation_split(samples, train_indices, fraction, seed):
    if not 0.0 < fraction < 0.5:
        raise ValueError("Validation fraction must be between 0 and 0.5")
    frame = pd.DataFrame(
        {"scaffold": [samples[index].scaffold for index in train_indices]},
        index=train_indices,
    )
    groups = frame.groupby("scaffold").groups
    if len(groups) < 2:
        raise ValueError("At least two training scaffolds are required for validation")
    scaffold_keys = list(groups)
    random.Random(seed).shuffle(scaffold_keys)
    validation_target = max(1, int(len(train_indices) * fraction))
    validation_indices = []
    remaining_indices = []
    for position, scaffold in enumerate(scaffold_keys):
        row_indices = list(groups[scaffold])
        if (
            len(validation_indices) < validation_target
            and position < len(scaffold_keys) - 1
        ):
            validation_indices.extend(row_indices)
        else:
            remaining_indices.extend(row_indices)
    if not validation_indices or not remaining_indices:
        raise ValueError("Scaffold partitioning did not produce train and validation sets")
    return remaining_indices, validation_indices


if torch is not None:
    class GraphRegressor(nn.Module):
        def __init__(self, hidden_size=128, message_passing_steps=3, dropout=0.15):
            super().__init__()
            embedding_sizes = [119, 7, 11, 6, 2, 7, 2]
            embedding_width = 16
            self.atom_embeddings = nn.ModuleList(
                nn.Embedding(size, embedding_width) for size in embedding_sizes
            )
            self.node_projection = nn.Linear(embedding_width * len(embedding_sizes), hidden_size)
            self.message_layers = nn.ModuleList(
                nn.Linear(hidden_size + 13, hidden_size)
                for _ in range(message_passing_steps)
            )
            self.update_layers = nn.ModuleList(
                nn.GRUCell(hidden_size, hidden_size)
                for _ in range(message_passing_steps)
            )
            self.norm_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(message_passing_steps)
            )
            self.dropout = nn.Dropout(dropout)
            self.readout = nn.Sequential(
                nn.Linear(hidden_size * 2 + len(DESCRIPTOR_COLUMNS), hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(
            self,
            node_categories,
            edge_index,
            edge_features,
            graph_index,
            global_features,
        ):
            node_hidden = self.node_projection(
                torch.cat(
                    [
                        embedding(node_categories[:, feature_index])
                        for feature_index, embedding in enumerate(self.atom_embeddings)
                    ],
                    dim=1,
                )
            )
            node_hidden = torch.relu(node_hidden)
            for message_layer, update_layer, norm_layer in zip(
                self.message_layers, self.update_layers, self.norm_layers
            ):
                aggregated = torch.zeros_like(node_hidden)
                if edge_index.shape[1]:
                    source, destination = edge_index
                    messages = torch.relu(
                        message_layer(
                            torch.cat(
                                [node_hidden[source], edge_features], dim=1
                            )
                        )
                    )
                    aggregated.index_add_(0, destination, messages)
                    degree = node_hidden.new_zeros((node_hidden.shape[0], 1))
                    degree.index_add_(
                        0,
                        destination,
                        node_hidden.new_ones((destination.shape[0], 1)),
                    )
                    aggregated = aggregated / degree.clamp_min(1.0)
                updated = update_layer(aggregated, node_hidden)
                node_hidden = norm_layer(node_hidden + self.dropout(updated))

            graph_count = global_features.shape[0]
            graph_sum = node_hidden.new_zeros((graph_count, node_hidden.shape[1]))
            graph_sum.index_add_(0, graph_index, node_hidden)
            graph_sizes = node_hidden.new_zeros((graph_count, 1))
            graph_sizes.index_add_(
                0, graph_index, node_hidden.new_ones((graph_index.shape[0], 1))
            )
            graph_mean = graph_sum / graph_sizes.clamp_min(1.0)
            graph_max = torch.stack(
                [
                    node_hidden[graph_index == graph_number].max(dim=0).values
                    for graph_number in range(graph_count)
                ],
                dim=0,
            )
            graph_features = torch.cat(
                [graph_mean, graph_max, global_features], dim=1
            )
            return self.readout(graph_features).squeeze(1)
else:
    GraphRegressor = None


def _make_batch(samples, sample_indices, descriptor_mean, descriptor_scale, target_mean, target_scale, device):
    node_categories = []
    edge_indices = []
    edge_features = []
    graph_indices = []
    global_features = []
    targets = []
    node_offset = 0
    for graph_number, sample_index in enumerate(sample_indices):
        sample = samples[int(sample_index)]
        node_count = len(sample.node_categories)
        node_categories.append(sample.node_categories)
        if sample.edge_index.shape[1]:
            edge_indices.append(sample.edge_index + node_offset)
            edge_features.append(sample.edge_features)
        graph_indices.append(np.full(node_count, graph_number, dtype=np.int64))
        global_features.append(sample.global_features)
        targets.append(sample.target)
        node_offset += node_count

    if edge_indices:
        edge_index_array = np.concatenate(edge_indices, axis=1)
        edge_feature_array = np.concatenate(edge_features, axis=0)
    else:
        edge_index_array = np.empty((2, 0), dtype=np.int64)
        edge_feature_array = np.empty((0, 13), dtype=np.float32)
    descriptor_array = np.asarray(global_features, dtype=np.float32)
    descriptor_array = (descriptor_array - descriptor_mean) / descriptor_scale
    target_array = (np.asarray(targets, dtype=np.float32) - target_mean) / target_scale
    return (
        torch.as_tensor(np.concatenate(node_categories), dtype=torch.long, device=device),
        torch.as_tensor(edge_index_array, dtype=torch.long, device=device),
        torch.as_tensor(edge_feature_array, dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(graph_indices), dtype=torch.long, device=device),
        torch.as_tensor(descriptor_array, dtype=torch.float32, device=device),
        torch.as_tensor(target_array, dtype=torch.float32, device=device),
    )


def _predict(model, samples, indices, batch_size, descriptor_mean, descriptor_scale, target_mean, target_scale, device):
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = _make_batch(
                samples,
                batch_indices,
                descriptor_mean,
                descriptor_scale,
                target_mean,
                target_scale,
                device,
            )
            output = model(*batch[:5])
            predictions.extend((output.cpu().numpy() * target_scale + target_mean).tolist())
            actual.extend([samples[index].target for index in batch_indices])
    return np.asarray(actual), np.asarray(predictions)


def _metric_values(actual, predicted):
    residual = actual - predicted
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    total_variance = float(np.sum(np.square(actual - np.mean(actual))))
    r_squared = (
        float(1.0 - np.sum(np.square(residual)) / total_variance)
        if total_variance > 0
        else float("nan")
    )
    pearson = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return mae, rmse, r_squared, pearson


def _save_diagnostics(actual, predicted, output_path):
    residuals = actual - predicted
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(actual, predicted, alpha=0.55, s=18, color="#176B87")
    limits = [
        min(float(actual.min()), float(predicted.min())),
        max(float(actual.max()), float(predicted.max())),
    ]
    axes[0].plot(limits, limits, linestyle="--", color="#C0392B", linewidth=1)
    axes[0].set_xlabel("Actual pIC50")
    axes[0].set_ylabel("Predicted pIC50")
    axes[0].set_title("Scaffold-held-out predictions")
    axes[1].hist(residuals, bins=30, color="#2E8B57", edgecolor="white")
    axes[1].axvline(0, color="#C0392B", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Actual - predicted pIC50")
    axes[1].set_ylabel("Molecules")
    axes[1].set_title("Held-out residuals")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _resolve_device(device_name):
    if device_name in {"auto", "cpu"}:
        return torch.device("cpu"), "cpu"
    if device_name != "cuda":
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    if not torch.backends.cuda.is_built():
        raise RuntimeError("This PyTorch installation was not built with CUDA support")

    supported_architectures = set(torch.cuda.get_arch_list())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            available_devices = [
                (index, torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            ]
    except (RuntimeError, torch.AcceleratorError) as error:
        raise RuntimeError("Could not inspect CUDA device compatibility") from error

    for index, (major, minor) in available_devices:
        architecture = f"sm_{major}{minor}"
        if architecture in supported_architectures:
            return torch.device(f"cuda:{index}"), f"cuda:{index}"

    detected = ", ".join(
        f"sm_{major}{minor}" for _, (major, minor) in available_devices
    ) or "no CUDA device detected"
    supported = ", ".join(sorted(supported_architectures)) or "none"
    raise RuntimeError(
        f"CUDA device architecture ({detected}) is not supported by this PyTorch "
        f"build ({supported}). Run with --device cpu."
    )


def train_gnn(
    dataset_path=None,
    epochs=150,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.0001,
    patience=20,
    validation_fraction=0.15,
    hidden_size=128,
    message_passing_steps=3,
    seed=50,
    device_name="cpu",
):
    if torch is None:
        raise ImportError("PyTorch is required. Install it in the Python environment used to run this script.")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")

    root = _resolve_project_root()
    dataset_path = Path(dataset_path) if dataset_path else _resolve_default_dataset()
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_root = root / "output"
    log_path = output_root / "logs"
    model_path = output_root / "models"
    figure_path = output_root / "figures"
    for directory in (log_path, model_path, figure_path):
        directory.mkdir(parents=True, exist_ok=True)

    device, selected_device = _resolve_device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if selected_device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    samples, rejected = load_graph_dataset(dataset_path)
    rejected.to_csv(log_path / "GNN_rejected_rows.csv", index=False)
    train_and_validation_indices, test_indices = _baseline_scaffold_split(samples)
    train_indices, validation_indices = _validation_split(
        samples, train_and_validation_indices, validation_fraction, seed
    )
    train_scaffolds = {samples[index].scaffold for index in train_indices}
    validation_scaffolds = {samples[index].scaffold for index in validation_indices}
    test_scaffolds = {samples[index].scaffold for index in test_indices}
    if (
        train_scaffolds & validation_scaffolds
        or train_scaffolds & test_scaffolds
        or validation_scaffolds & test_scaffolds
    ):
        raise RuntimeError("Scaffold leakage detected across data partitions")

    train_descriptors = np.stack(
        [samples[index].global_features for index in train_indices]
    ).astype(np.float32)
    descriptor_mean = np.nanmean(train_descriptors, axis=0)
    descriptor_mean = np.nan_to_num(descriptor_mean, nan=0.0, posinf=0.0, neginf=0.0)
    imputed_train_descriptors = np.where(
        np.isfinite(train_descriptors), train_descriptors, descriptor_mean
    )
    descriptor_scale = imputed_train_descriptors.std(axis=0)
    descriptor_scale[descriptor_scale < 1e-8] = 1.0
    imputed_descriptor_values = int(np.isnan(train_descriptors).sum())
    for sample in samples:
        sample.global_features = np.where(
            np.isfinite(sample.global_features),
            sample.global_features,
            descriptor_mean,
        ).astype(np.float32)
    train_targets = np.asarray(
        [samples[index].target for index in train_indices], dtype=np.float32
    )
    target_mean = float(train_targets.mean())
    target_scale = float(train_targets.std())
    if target_scale < 1e-8:
        target_scale = 1.0

    model = GraphRegressor(
        hidden_size=hidden_size,
        message_passing_steps=message_passing_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.SmoothL1Loss()
    best_validation_mae = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        shuffled_indices = np.random.permutation(train_indices)
        for start in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[start : start + batch_size].tolist()
            batch = _make_batch(
                samples,
                batch_indices,
                descriptor_mean,
                descriptor_scale,
                target_mean,
                target_scale,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(*batch[:5])
            loss = loss_function(predictions, batch[5])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_actual, validation_predicted = _predict(
            model,
            samples,
            validation_indices,
            batch_size,
            descriptor_mean,
            descriptor_scale,
            target_mean,
            target_scale,
            device,
        )
        validation_mae = float(np.mean(np.abs(validation_actual - validation_predicted)))
        history.append(
            {
                "epoch": epoch,
                "training_smooth_l1": float(np.mean(epoch_losses)),
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validated model checkpoint")
    model.load_state_dict(best_state)
    test_actual, test_predicted = _predict(
        model,
        samples,
        test_indices,
        batch_size,
        descriptor_mean,
        descriptor_scale,
        target_mean,
        target_scale,
        device,
    )
    test_mae, test_rmse, test_r2, test_pearson = _metric_values(
        test_actual, test_predicted
    )

    run_name = "GNN"
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(log_path / f"{run_name}_training_history.csv", index=False)
    train_records = [samples[index] for index in train_indices]
    validation_records = [samples[index] for index in validation_indices]
    test_records = [samples[index] for index in test_indices]
    for split_name, records in (
        ("train", train_records),
        ("validation", validation_records),
        ("test", test_records),
    ):
        pd.DataFrame(
            [
                {
                    "source_index": sample.source_index,
                    "cleaned_smiles": sample.smiles,
                    "PIC50": sample.target,
                    "Scaffold_InChI": sample.scaffold,
                }
                for sample in records
            ]
        ).to_csv(log_path / f"{run_name}_{split_name}.csv", index=False)

    pd.DataFrame(
        [
            {
                "source_index": samples[index].source_index,
                "actual": float(actual),
                "predicted": float(predicted),
                "residual": float(actual - predicted),
                "Scaffold_InChI": samples[index].scaffold,
            }
            for index, actual, predicted in zip(
                test_indices, test_actual, test_predicted
            )
        ]
    ).to_csv(log_path / f"{run_name}_predictions.csv", index=False)

    metrics = {
        "model": "GraphRegressor",
        "dataset": str(dataset_path),
        "target": "PIC50",
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_pearson_r": test_pearson,
        "best_validation_mae": best_validation_mae,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "n_test": len(test_indices),
        "n_train_scaffolds": len(train_scaffolds),
        "n_validation_scaffolds": len(validation_scaffolds),
        "n_test_scaffolds": len(test_scaffolds),
        "n_rejected": len(rejected),
        "n_train_descriptor_values_imputed": imputed_descriptor_values,
        "n_node_features": len(ATOM_FEATURE_COLUMNS),
        "n_edge_features": len(EDGE_FEATURE_COLUMNS),
        "n_global_features": len(DESCRIPTOR_COLUMNS),
        "seed": seed,
        "device": selected_device,
    }
    pd.DataFrame([metrics]).to_csv(log_path / f"{run_name}_metrics.csv", index=False)
    _save_diagnostics(
        test_actual,
        test_predicted,
        figure_path / f"{run_name}_diagnostics.png",
    )
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": {
            "hidden_size": hidden_size,
            "message_passing_steps": message_passing_steps,
            "dropout": 0.15,
        },
        "descriptor_columns": DESCRIPTOR_COLUMNS,
        "descriptor_mean": descriptor_mean.tolist(),
        "descriptor_scale": descriptor_scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "atom_feature_columns": ATOM_FEATURE_COLUMNS,
        "edge_feature_columns": EDGE_FEATURE_COLUMNS,
        "dataset_path": str(dataset_path),
        "seed": seed,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }
    torch.save(checkpoint, model_path / f"{run_name}_model.pt")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    arguments = parser.parse_args()
    train_gnn(
        dataset_path=arguments.data,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        patience=arguments.patience,
        validation_fraction=arguments.validation_fraction,
        hidden_size=arguments.hidden_size,
        message_passing_steps=arguments.message_passing_steps,
        seed=arguments.seed,
        device_name=arguments.device,
    )


if __name__ == "__main__":
    main()
