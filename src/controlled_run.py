from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdBase

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_audit import stable_hash
from data_cleaning import InChIKeyConversion, calculate_molecular_descriptors
from data_engineering import DataEng

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
MORGAN_COLUMNS = [f"morgan_{index}" for index in range(1024)]
MODEL_VARIANTS = ("morgan", "descriptors", "combined")
CLASSICAL_MODELS = ("XGBoost", "RandomForest")
MODEL_FEATURE_COUNTS = {"morgan": 1024, "descriptors": 14, "combined": 1038}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    packages = ["numpy", "pandas", "rdkit", "scikit-learn", "xgboost", "lightgbm", "optuna", "torch"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _scaffold_key(smiles: str) -> str:
    scaffold_smiles = DataEng.mscl(smiles)
    if not scaffold_smiles:
        return "__NO_SCAFFOLD__"
    return InChIKeyConversion(scaffold_smiles) or "__NO_SCAFFOLD__"


def _load_verified_cohort(audit_dir: Path):
    summary_path = audit_dir / "audit_summary.json"
    assignment_path = audit_dir / "scaffold_assignments.csv"
    if not summary_path.is_file() or not assignment_path.is_file():
        raise FileNotFoundError(f"Expected corrected audit artifacts under {audit_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_path = Path(summary["dataset"]["raw_path"]).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Audit source snapshot is missing: {raw_path}")
    raw_hash = _sha256(raw_path)
    if raw_hash != summary["dataset"]["raw_sha256"]:
        raise ValueError("Raw snapshot SHA-256 does not match the corrected audit")

    cohort = pd.read_csv(assignment_path)
    required = {"source_index", "InChIkey", "cleaned_smiles", "PIC50", "scaffold_InChI", "split"}
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError(f"Audit assignment is missing required columns: {sorted(missing)}")
    expected_count = int(summary["dataset"]["final_records_current_behavior"])
    expected_hash = summary["dataset"]["stable_final_sha256"]
    if len(cohort) != expected_count or cohort["InChIkey"].nunique(dropna=True) != expected_count:
        raise ValueError("Audit assignment row count or unique InChIKey count is invalid")
    if cohort["InChIkey"].isna().any() or cohort["InChIkey"].duplicated().any():
        raise ValueError("Audit cohort has null or duplicate InChIKeys")
    if cohort["cleaned_smiles"].isna().any() or cohort["PIC50"].isna().any():
        raise ValueError("Audit cohort has missing cleaned_smiles or PIC50")
    if not np.isfinite(pd.to_numeric(cohort["PIC50"], errors="coerce")).all():
        raise ValueError("Audit cohort contains non-finite PIC50 values")
    actual_hash = stable_hash(cohort, ["InChIkey", "cleaned_smiles", "PIC50"])
    if actual_hash != expected_hash:
        raise ValueError("Audit cohort stable hash does not match audit_summary.json")

    recalculated_scaffolds = cohort["cleaned_smiles"].map(_scaffold_key)
    if not recalculated_scaffolds.equals(cohort["scaffold_InChI"].astype(str)):
        raise ValueError("Recalculated Bemis-Murcko scaffold keys differ from the audit")
    if not cohort["source_index"].is_unique:
        raise ValueError("Audit cohort source_index values are not unique")
    return summary, raw_path, raw_hash, cohort, actual_hash


def _calculate_features(cohort: pd.DataFrame) -> pd.DataFrame:
    fingerprints = []
    descriptor_rows = []
    for smiles in cohort["cleaned_smiles"]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("A curated cleaned_smiles could not be parsed")
        fingerprint = DataEng.morgan_fingerprinting_s_method(smiles, radius=2, nBits=1024)
        if fingerprint.shape != (1024,):
            raise ValueError(f"Unexpected Morgan vector shape: {fingerprint.shape}")
        fingerprints.append(fingerprint)
        descriptor_rows.append(calculate_molecular_descriptors(smiles))

    fingerprint_frame = pd.DataFrame(
        np.vstack(fingerprints), index=cohort.index, columns=MORGAN_COLUMNS
    )
    descriptor_frame = pd.DataFrame(
        descriptor_rows, index=cohort.index, columns=DESCRIPTOR_COLUMNS
    )
    descriptor_frame = descriptor_frame.apply(pd.to_numeric, errors="coerce")
    if descriptor_frame.drop(columns=["MaxPartialCharge", "MinPartialCharge"]).isna().any().any():
        raise ValueError("Unexpected missing values outside partial-charge descriptors")
    return pd.concat([fingerprint_frame, descriptor_frame], axis=1)


def _calculate_split(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    split_input = pd.DataFrame(
        {"PIC50": cohort["PIC50"], "Scaffold_InChI": cohort["scaffold_InChI"]},
        index=np.arange(len(cohort)),
    )
    x_train, _, x_test, _ = DataEng._split_frame(split_input)
    train_positions = set(map(int, x_train.index))
    test_positions = set(map(int, x_test.index))
    if train_positions & test_positions or train_positions | test_positions != set(range(len(cohort))):
        raise ValueError("Scaffold split does not uniquely cover the curated cohort")

    assignments = cohort[["source_index", "InChIkey", "cleaned_smiles", "scaffold_InChI"]].copy()
    assignments.insert(0, "row_position", np.arange(len(cohort)))
    assignments["split"] = [
        "train" if position in train_positions else "test"
        for position in range(len(cohort))
    ]
    if not assignments["InChIkey"].is_unique:
        raise ValueError("Split assignments are not unique by InChIKey")

    train = assignments.loc[assignments["split"] == "train"]
    test = assignments.loc[assignments["split"] == "test"]
    train_keys, test_keys = set(train["InChIkey"]), set(test["InChIkey"])
    train_smiles, test_smiles = set(train["cleaned_smiles"]), set(test["cleaned_smiles"])
    train_scaffolds, test_scaffolds = set(train["scaffold_InChI"]), set(test["scaffold_InChI"])
    stats = {
        "train_molecules": len(train),
        "test_molecules": len(test),
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "scaffold_overlap": len(train_scaffolds & test_scaffolds),
        "InChIKey_overlap": len(train_keys & test_keys),
        "cleaned_SMILES_overlap": len(train_smiles & test_smiles),
        "test_singleton_fraction": float(
            (test.groupby("scaffold_InChI")["InChIkey"].transform("size") == 1).mean()
        ),
    }
    audited_split = cohort.set_index("source_index")["split"]
    recalculated_split = assignments.set_index("source_index")["split"]
    if not audited_split.equals(recalculated_split):
        raise ValueError("Recalculated train/test assignments differ from the corrected audit")
    if stats["scaffold_overlap"] or stats["InChIKey_overlap"] or stats["cleaned_SMILES_overlap"]:
        raise ValueError("Train/test leakage detected in the recalculated split")
    return assignments, stats


def _new_run_directory() -> Path:
    runs_root = ROOT / "output" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{stamp}_{suffix:02d}"
        suffix += 1
    return run_dir


def prepare(audit_argument: str) -> Path:
    audit_dir = (ROOT / audit_argument).resolve()
    summary, raw_path, raw_hash, assignments, cohort_hash = _load_verified_cohort(audit_dir)
    features = _calculate_features(assignments)
    if features.shape != (len(assignments), 1038):
        raise ValueError(f"Unexpected combined feature matrix shape: {features.shape}")
    split_assignments, split_stats = _calculate_split(assignments)

    audit_split = assignments.set_index("InChIkey")["split"]
    new_split = split_assignments.set_index("InChIkey")["split"]
    if not audit_split.equals(new_split):
        raise ValueError("Prepared split differs from the verified audit split")
    expected_split = summary["scaffold_split"]
    for field, actual in [
        ("molecules_train", split_stats["train_molecules"]),
        ("molecules_test", split_stats["test_molecules"]),
        ("scaffolds_train", split_stats["train_scaffolds"]),
        ("scaffolds_test", split_stats["test_scaffolds"]),
    ]:
        if actual != int(expected_split[field]):
            raise ValueError(f"Recalculated split {field} does not match the audit")

    run_dir = _new_run_directory()
    input_dir = run_dir / "input"
    feature_dir = run_dir / "features"
    split_dir = run_dir / "splits"
    for directory in (input_dir, feature_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_input = assignments[["source_index", "InChIkey", "cleaned_smiles", "PIC50"]].copy()
    model_input.to_csv(input_dir / "curated_molecules.csv", index=False)
    features.to_csv(feature_dir / "combined_features.csv", index=False)
    split_assignments.to_csv(split_dir / "split_assignments.csv", index=False)
    split_assignments.loc[split_assignments["split"] == "train", "InChIkey"].to_csv(
        split_dir / "train_inchikeys.txt", index=False, header=False
    )
    split_assignments.loc[split_assignments["split"] == "test", "InChIkey"].to_csv(
        split_dir / "test_inchikeys.txt", index=False, header=False
    )

    package_versions = _package_versions()
    metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_directory": str(run_dir),
        "source_raw_snapshot": str(raw_path),
        "raw_snapshot_sha256": raw_hash,
        "audit_directory": str(audit_dir),
        "audit_summary": str(audit_dir / "audit_summary.json"),
        "audit_assignment": str(audit_dir / "scaffold_assignments.csv"),
        "audit_cohort_stable_sha256": cohort_hash,
        "molecule_count": len(model_input),
        "unique_InChIKey_count": int(model_input["InChIkey"].nunique()),
        "feature_file": str(feature_dir / "combined_features.csv"),
        "feature_dimensions": {"Morgan": 1024, "RDKit_descriptors": 14, "combined": 1038},
        "morgan_radius": 2,
        "morgan_bits": 1024,
        "descriptor_columns": DESCRIPTOR_COLUMNS,
        "feature_columns": list(features.columns),
        "descriptor_missing_values": {
            column: int(features[column].isna().sum())
            for column in ("MaxPartialCharge", "MinPartialCharge")
        },
        "split_method": "Bemis-Murcko scaffold grouping, descending scaffold size, whole groups allocated toward 80% train",
        "split_counts": split_stats,
        "split_assignment_file": str(split_dir / "split_assignments.csv"),
        "train_InChIKey_file": str(split_dir / "train_inchikeys.txt"),
        "test_InChIKey_file": str(split_dir / "test_inchikeys.txt"),
        "code_git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "rdkit_version": rdBase.rdkitVersion,
        "package_versions": package_versions,
        "classical_phase_models": ["XGBoost", "RandomForest"],
        "lightgbm_invoked": False,
        "optuna_invoked": False,
        "gnn_invoked": False,
        "training_started": False,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_directory": str(run_dir), **metadata["split_counts"], "feature_dimensions": metadata["feature_dimensions"]}, indent=2))
    return run_dir


def _load_prepared_run(run_dir: Path):
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_path = Path(metadata["source_raw_snapshot"])
    if _sha256(raw_path) != metadata["raw_snapshot_sha256"]:
        raise ValueError("Raw snapshot changed since this run was prepared")
    cohort = pd.read_csv(run_dir / "input" / "curated_molecules.csv")
    split = pd.read_csv(run_dir / "splits" / "split_assignments.csv")
    if len(cohort) != 4597 or cohort["InChIkey"].nunique() != 4597:
        raise ValueError("Prepared cohort no longer matches the verified 4,597-molecule set")
    if cohort["InChIkey"].isna().any() or not np.isfinite(cohort["PIC50"]).all():
        raise ValueError("Prepared cohort has invalid identifiers or target values")
    if stable_hash(cohort, ["InChIkey", "cleaned_smiles", "PIC50"]) != metadata["audit_cohort_stable_sha256"]:
        raise ValueError("Prepared cohort no longer matches the verified audit hash")
    return run_dir, metadata, cohort, split


def _build_persisted_split_inputs(run_dir: Path, cohort: pd.DataFrame, split: pd.DataFrame, model_class):
    feature_frame = pd.read_csv(run_dir / "features" / "combined_features.csv")
    train_keys = (run_dir / "splits" / "train_inchikeys.txt").read_text(encoding="utf-8").splitlines()
    test_keys = (run_dir / "splits" / "test_inchikeys.txt").read_text(encoding="utf-8").splitlines()
    split_train_keys = split.loc[split["split"] == "train", "InChIkey"].tolist()
    split_test_keys = split.loc[split["split"] == "test", "InChIkey"].tolist()

    if len(cohort) != 4597 or len(train_keys) != 3677 or len(test_keys) != 920:
        raise ValueError("Persisted cohort or train/test key-list counts are incorrect")
    if len(set(train_keys)) != len(train_keys) or len(set(test_keys)) != len(test_keys):
        raise ValueError("Persisted train/test key lists contain duplicates")
    if train_keys != split_train_keys or test_keys != split_test_keys:
        raise ValueError("Persisted key lists differ from split_assignments.csv")
    if set(train_keys) & set(test_keys):
        raise ValueError("Persisted train/test InChIKey overlap is nonzero")
    if len(set(train_keys) | set(test_keys)) != len(cohort):
        raise ValueError("Persisted train/test keys do not cover the curated cohort")
    if list(cohort["InChIkey"]) != list(split["InChIkey"]):
        raise ValueError("Curated cohort and split assignment row order do not match")
    if feature_frame.shape != (len(cohort), 1038):
        raise ValueError(f"Prepared feature matrix has unexpected shape: {feature_frame.shape}")

    feature_frame.insert(0, "InChIkey", cohort["InChIkey"].to_numpy())
    feature_frame.insert(1, "cleaned_smiles", cohort["cleaned_smiles"].to_numpy())
    feature_frame["Scaffold_InChI"] = split["scaffold_InChI"].to_numpy()
    feature_columns = set(feature_frame.columns)
    target = cohort.set_index("InChIkey")["PIC50"]
    indexed_features = feature_frame.set_index("InChIkey", drop=False)
    split_inputs = {}
    for variant in MODEL_VARIANTS:
        columns = model_class._model_columns(feature_frame, variant)
        expected_columns = (
            MORGAN_COLUMNS if variant == "morgan"
            else DESCRIPTOR_COLUMNS if variant == "descriptors"
            else MORGAN_COLUMNS + DESCRIPTOR_COLUMNS
        )
        if columns != expected_columns or len(columns) != MODEL_FEATURE_COUNTS[variant]:
            raise ValueError(f"{variant} does not select its exact expected predictors")
        if "PIC50" in feature_columns or "PIC50" in columns:
            raise ValueError("PIC50 must remain separate from the feature matrix")

        x_train = indexed_features.loc[train_keys].reset_index(drop=True)
        x_test = indexed_features.loc[test_keys].reset_index(drop=True)
        y_train = pd.Series(target.loc[train_keys].to_numpy(), name="PIC50")
        y_test = pd.Series(target.loc[test_keys].to_numpy(), name="PIC50")
        if set(x_train["InChIkey"]) != set(train_keys) or set(x_test["InChIkey"]) != set(test_keys):
            raise ValueError(f"{variant} matrices do not preserve persisted train/test identities")
        if len(x_train) != 3677 or len(x_test) != 920 or len(x_train) + len(x_test) != 4597:
            raise ValueError(f"{variant} matrices have invalid train/test row counts")
        if set(x_train["Scaffold_InChI"]) & set(x_test["Scaffold_InChI"]):
            raise ValueError(f"{variant} matrices have scaffold overlap")
        split_inputs[variant] = (x_train, y_train, x_test, y_test)
    return split_inputs


def validate_classical_inputs(run_dir: Path) -> None:
    run_dir, _, cohort, split = _load_prepared_run(run_dir)
    from model_building import Model

    if "lightgbm" in sys.modules:
        raise RuntimeError("LightGBM was imported by the controlled XGBoost/RF path")
    if "GNNmodel" in sys.modules or "src.GNNmodel" in sys.modules:
        raise RuntimeError("GNN code was imported by the controlled XGBoost/RF path")
    split_inputs = _build_persisted_split_inputs(run_dir, cohort, split, Model)
    validation = {
        variant: {
            "train_rows": len(data[0]),
            "test_rows": len(data[2]),
            "train_keys": data[0]["InChIkey"].nunique(),
            "test_keys": data[2]["InChIkey"].nunique(),
            "predictor_columns": len(Model._model_columns(data[0], variant)),
        }
        for variant, data in split_inputs.items()
    }
    configurations = [
        {"model": model, "features": variant, **validation[variant]}
        for model in CLASSICAL_MODELS
        for variant in MODEL_VARIANTS
    ]
    print(json.dumps({
        "persisted_split_validated": True,
        "lightgbm_imported": False,
        "gnn_imported": False,
        "configurations": configurations,
    }, indent=2))


def run_classical(run_dir: Path, trials: int) -> None:
    run_dir, metadata, cohort, split = _load_prepared_run(run_dir)
    if trials < 1:
        raise ValueError("Trial count must be positive")

    from model_building import Model

    if "lightgbm" in sys.modules:
        raise RuntimeError("LightGBM was imported by the controlled XGBoost/RF path")
    if "GNNmodel" in sys.modules or "src.GNNmodel" in sys.modules:
        raise RuntimeError("GNN code was imported by the controlled XGBoost/RF path")
    prepared_inputs = _build_persisted_split_inputs(run_dir, cohort, split, Model)
    model_runner = Model.__new__(Model)
    model_runner.output_path = run_dir
    model_runner.log_path = run_dir / "logs"
    model_runner.model_path = run_dir / "models"
    model_runner.figure_path = run_dir / "figures"
    for directory in (model_runner.log_path, model_runner.model_path, model_runner.figure_path):
        directory.mkdir(parents=True, exist_ok=True)

    variant_metadata = {}
    for variant, data in prepared_inputs.items():
        x_train, _, x_test, _ = data
        columns = model_runner._model_columns(x_train, variant)
        if len(columns) != MODEL_FEATURE_COUNTS[variant]:
            raise ValueError(f"{variant} predictor width differs from the prepared metadata")
        variant_metadata[variant] = {
            "n_train": len(x_train),
            "n_test": len(x_test),
            "n_features": len(columns),
        }

    results = []
    for model_type in CLASSICAL_MODELS:
        for variant in MODEL_VARIANTS:
            file_name = f"{model_type}_{variant}.joblib"
            study_name = f"{run_dir.name}_{model_type}_{variant}"
            _, study, _ = model_runner._optimize_and_evaluate(
                model_type,
                variant,
                study_name,
                file_name,
                n_trials=trials,
                prepared_data=prepared_inputs[variant],
            )
            metrics = dict(model_runner.last_metrics)
            metrics["n_features"] = variant_metadata[variant]["n_features"]
            metrics["optuna_trials_requested"] = trials
            metrics["optuna_trials_complete"] = sum(
                trial.state.name == "COMPLETE" for trial in study.trials
            )
            pd.DataFrame([metrics]).to_csv(
                model_runner.log_path / f"{model_type}_{variant}_metrics.csv", index=False
            )
            results.append({
                "model": model_type,
                "features": variant,
                "cv_mae": -float(study.best_value),
                "n_features": variant_metadata[variant]["n_features"],
                "n_train": variant_metadata[variant]["n_train"],
                "n_test": variant_metadata[variant]["n_test"],
                "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
                "test_r2": metrics["test_r2"],
                "pearson_r": metrics["pearson_r"],
                "spearman_r": metrics["spearman_r"],
                "optuna_trials_requested": trials,
                "optuna_trials_complete": len(study.trials),
            })

    pd.DataFrame(results).to_csv(run_dir / "logs" / "classical_comparison_metrics.csv", index=False)
    metadata["training_started"] = True
    metadata["optuna_invoked"] = True
    metadata["models_completed"] = list(CLASSICAL_MODELS)
    metadata["optuna_trials_per_configuration"] = trials
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and run isolated AChE QSAR experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="verify cohort, generate features, and freeze a scaffold split")
    prepare_parser.add_argument("--audit-dir", default="audit/20261007_110522")
    classical_parser = subparsers.add_parser("classical", help="run only XGBoost and Random Forest on a prepared run")
    classical_parser.add_argument("--run-dir", type=Path, required=True)
    classical_parser.add_argument("--trials", type=int, default=5)
    validate_parser = subparsers.add_parser("validate-classical", help="validate persisted model matrices without training")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        prepare(arguments.audit_dir)
    elif arguments.command == "classical":
        run_classical(arguments.run_dir, arguments.trials)
    else:
        validate_classical_inputs(arguments.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())