import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

if __package__:
    from .GNNmodel import (
        ATOM_FEATURE_COLUMNS,
        DESCRIPTOR_COLUMNS,
        EDGE_FEATURE_COLUMNS,
        _baseline_scaffold_split,
        _resolve_default_dataset,
        _resolve_project_root,
        _validation_split,
        load_graph_dataset,
    )
else:
    from GNNmodel import (
        ATOM_FEATURE_COLUMNS,
        DESCRIPTOR_COLUMNS,
        EDGE_FEATURE_COLUMNS,
        _baseline_scaffold_split,
        _resolve_default_dataset,
        _resolve_project_root,
        _validation_split,
        load_graph_dataset,
    )


SEED = 50
MORGAN_COLUMNS = [f"morgan_{index}" for index in range(1024)]
VARIANTS = {
    "morgan": MORGAN_COLUMNS,
    "descriptors": DESCRIPTOR_COLUMNS,
    "combined": MORGAN_COLUMNS + DESCRIPTOR_COLUMNS,
}


def _resolve_output_directory():
    return _resolve_project_root() / "output" / "audit" / "GNN_classical_comparison"


def _build_feature_matrices(samples, train_indices):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprint_rows = []
    for sample in samples:
        mol = Chem.MolFromSmiles(sample.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at source row {sample.source_index}")
        fingerprint_rows.append(
            np.asarray(generator.GetFingerprint(mol), dtype=np.float32)
        )
    fingerprints = np.vstack(fingerprint_rows)

    descriptors = np.vstack([sample.global_features for sample in samples]).astype(
        np.float32
    )
    missing_rows, missing_columns = np.where(~np.isfinite(descriptors))
    allowed_missing_columns = {
        DESCRIPTOR_COLUMNS.index("MaxPartialCharge"),
        DESCRIPTOR_COLUMNS.index("MinPartialCharge"),
    }
    if np.isinf(descriptors).any() or not set(missing_columns).issubset(
        allowed_missing_columns
    ):
        raise ValueError("Unexpected non-finite molecular descriptor values")

    combined = np.concatenate([fingerprints, descriptors], axis=1)
    matrices = {
        "morgan": fingerprints,
        "descriptors": descriptors,
        "combined": combined,
    }
    expected_widths = {"morgan": 1024, "descriptors": 14, "combined": 1038}
    for variant, matrix in matrices.items():
        if matrix.shape != (len(samples), expected_widths[variant]):
            raise ValueError(f"Unexpected {variant} matrix shape: {matrix.shape}")
        if np.isinf(matrix).any() or (
            variant == "morgan" and not np.isfinite(matrix).all()
        ):
            raise ValueError(f"Non-finite values remain in {variant} features")
    return matrices


def _create_model(trial, model_type):
    if model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 150, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 8, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            ccp_alpha=trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True),
            criterion="squared_error",
            n_jobs=-1,
            random_state=SEED,
        )
    return xgb.XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 150, 700, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.55, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.55, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 12.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
    )


def _validate_saved_partitions(samples, train_indices, validation_indices, test_indices):
    root = _resolve_project_root()
    log_directory = root / "output" / "logs"
    saved_splits = {
        "train": pd.read_csv(log_directory / "GNN_train.csv"),
        "validation": pd.read_csv(log_directory / "GNN_validation.csv"),
        "test": pd.read_csv(log_directory / "GNN_test.csv"),
    }
    indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    for split_name, sample_indices in indices.items():
        expected = {samples[index].source_index for index in sample_indices}
        observed = set(saved_splits[split_name]["source_index"].astype(int))
        if observed != expected:
            raise ValueError(
                f"Current {split_name} rows do not match the saved GNN partition"
            )
    scaffold_groups = [
        {samples[index].scaffold for index in sample_indices}
        for sample_indices in indices.values()
    ]
    if any(
        scaffold_groups[left] & scaffold_groups[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("Scaffold leakage detected between saved partitions")
    return saved_splits


def _metrics(actual, predicted):
    return {
        "test_mae": float(mean_absolute_error(actual, predicted)),
        "test_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "test_r2": float(r2_score(actual, predicted)),
        "test_pearson_r": float(pearsonr(actual, predicted).statistic),
        "test_spearman_r": float(spearmanr(actual, predicted).statistic),
    }


def _train_variant(model_type, variant, matrix, samples, train_indices, test_indices, folds, trials, output_directory):
    X_train = matrix[train_indices]
    y_train = np.asarray([samples[index].target for index in train_indices])
    groups = np.asarray([samples[index].scaffold for index in train_indices])
    X_test = matrix[test_indices]
    y_test = np.asarray([samples[index].target for index in test_indices])

    splitter = GroupKFold(n_splits=folds)

    def objective(trial):
        model = make_pipeline(
            SimpleImputer(strategy="mean"), _create_model(trial, model_type)
        )
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            groups=groups,
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            error_score="raise",
        )
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"clean_{model_type}_{variant}",
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    final_model = make_pipeline(
        SimpleImputer(strategy="mean"),
        _create_model(optuna.trial.FixedTrial(study.best_params), model_type),
    )
    final_model.fit(X_train, y_train)
    predicted = final_model.predict(X_test)

    model_name = f"{model_type}_{variant}"
    joblib.dump(final_model, output_directory / f"{model_name}.joblib")
    study.trials_dataframe().to_csv(
        output_directory / f"{model_name}_trials.csv", index=False
    )
    predictions = pd.DataFrame(
        {
            "source_index": [samples[index].source_index for index in test_indices],
            "actual": y_test,
            "predicted": predicted,
            "residual": y_test - predicted,
        }
    )
    predictions.to_csv(
        output_directory / f"{model_name}_predictions.csv", index=False
    )
    result = {
        "model": model_type,
        "features": variant,
        "n_features": matrix.shape[1],
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "cv_folds": folds,
        "optuna_trials": trials,
        "cv_mae": float(-study.best_value),
        "best_parameters": json.dumps(study.best_params, sort_keys=True),
    }
    result.update(_metrics(y_test, predicted))
    return result, predictions


def _write_report(report_path, results, samples, rejected, train_indices, validation_indices, test_indices, trials, folds):
    result_frame = pd.DataFrame(results).sort_values("test_mae")
    gnn_row = result_frame[result_frame["model"] == "GNN"].iloc[0]
    baseline_rows = result_frame[result_frame["model"] != "GNN"]
    best_baseline = baseline_rows.iloc[0]
    difference = float(gnn_row["test_mae"] - best_baseline["test_mae"])
    if difference < -1e-9:
        conclusion = (
            f"The GNN has the lowest held-out MAE, improving over {best_baseline['model']} "
            f"with {best_baseline['features']} features by {-difference:.4f} pIC50. "
            "This supports added value on this split, but does not establish general "
            "superiority without repeated scaffold splits or an external set."
        )
    elif difference > 1e-9:
        conclusion = (
            f"The best classical baseline ({best_baseline['model']} with "
            f"{best_baseline['features']} features) has lower held-out MAE than the GNN "
            f"by {difference:.4f} pIC50. This run does not support added predictive value "
            "from the GNN."
        )
    else:
        conclusion = (
            "The GNN and best classical baseline have effectively tied held-out MAE; "
            "this run does not establish added value from the GNN."
        )

    table_columns = [
        "model",
        "features",
        "n_features",
        "n_train",
        "n_test",
        "cv_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_pearson_r",
        "test_spearman_r",
    ]
    table = result_frame[table_columns].to_markdown(index=False, floatfmt=".4f")
    node_text = ", ".join(ATOM_FEATURE_COLUMNS)
    descriptor_text = ", ".join(DESCRIPTOR_COLUMNS)
    edge_text = ", ".join(EDGE_FEATURE_COLUMNS)
    report = f"""# GNN and Classical Baseline Audit

## Data and partitions

- Source table: `{_resolve_default_dataset().relative_to(_resolve_project_root())}`
- Valid graph records: {len(samples)}; rejected records: {len(rejected)}
- Train / validation / test rows: {len(train_indices)} / {len(validation_indices)} / {len(test_indices)}
- Every partition exactly matches the saved GNN source-row IDs; scaffold groups are disjoint.
- The test partition is held out from both classical hyperparameter search and model fitting.
- Classical tuning uses {trials} Optuna trials per model/representation and {folds}-fold `GroupKFold` over training scaffolds only.
- RF/XGBoost fit on the same training rows as the GNN. The saved GNN validation rows are not used for classical fitting or tuning.

## Feature verification

- Node features (7): {node_text}. Categorical inputs use embedding indices; formal charge is shifted by 5 after clipping to [-5, 5].
- Global descriptors (14): {descriptor_text}. Missing partial-charge values are imputed using means from training rows only.
- Edge channels: {len(EDGE_FEATURE_COLUMNS)}, not 3. Encoded as {edge_text}. This is four bond-type one-hot channels, three bond properties, and six stereo one-hot channels; each chemical bond is represented in both directions.
- The code currently implements 13 edge channels. If the intended specification is exactly 3 edge inputs, the current model does not match that specification and should be changed before interpreting its result.

## Leakage audit

- RF/XGBoost predictors are restricted to the explicit Morgan and/or RDKit descriptor matrices. PIC50, IC50, SMILES, molecule/InChIKey identifiers, and scaffold labels are excluded from model inputs.
- Morgan fingerprints and molecular descriptors depend only on molecular structure. Mean imputation is inside each CV pipeline, so each fold estimates missing-value means from that fold's training rows; final imputation is fitted on the full model-training partition.
- Hyperparameter search uses training rows and scaffold-grouped folds only; the test set is evaluated after final fitting.
- The scaffold split keeps each scaffold in exactly one of train, validation, or test.
- The current baseline feature-selection path has no direct target-column leakage. The earlier metric artifacts were not reused for this clean comparison.

## Results

{table}

## Assessment

{conclusion}

This is one fixed scaffold split. Treat the comparison as evidence for this partition, not a stable estimate across chemical space; repeated scaffold splits or an external test set are needed for a stronger claim.
"""
    report_path.write_text(report, encoding="utf-8")


def run_audit(trials=8, folds=3):
    if trials < 1 or folds < 2:
        raise ValueError("trials must be positive and folds must be at least 2")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dataset_path = _resolve_default_dataset()
    samples, rejected = load_graph_dataset(dataset_path)
    train_and_validation, test_indices = _baseline_scaffold_split(samples)
    train_indices, validation_indices = _validation_split(
        samples, train_and_validation, 0.15, SEED
    )
    saved_splits = _validate_saved_partitions(
        samples, train_indices, validation_indices, test_indices
    )
    matrices = _build_feature_matrices(samples, train_indices)

    output_directory = _resolve_output_directory()
    model_directory = output_directory / "models"
    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    results = []
    prediction_frames = []
    for model_type in ("RandomForest", "XGBoost"):
        for variant, column_names in VARIANTS.items():
            if len(column_names) != matrices[variant].shape[1]:
                raise ValueError(f"Feature whitelist mismatch for {variant}")
            if any(
                column in {"PIC50", "IC50", "IC50_nM", "cleaned_smiles", "InChIkey", "Scaffold_InChI"}
                for column in column_names
            ):
                raise ValueError(f"Leakage column found in {variant} features")
            result, predictions = _train_variant(
                model_type,
                variant,
                matrices[variant],
                samples,
                train_indices,
                test_indices,
                folds,
                trials,
                model_directory,
            )
            results.append(result)
            predictions = predictions.rename(
                columns={"predicted": f"{model_type}_{variant}"}
            ).drop(columns="residual")
            prediction_frames.append(predictions)

    gnn_metrics = pd.read_csv(_resolve_project_root() / "output" / "logs" / "GNN_metrics.csv").iloc[0]
    gnn_predictions = pd.read_csv(
        _resolve_project_root() / "output" / "logs" / "GNN_predictions.csv"
    )
    expected_test_sources = {
        samples[index].source_index for index in test_indices
    }
    if set(gnn_predictions["source_index"].astype(int)) != expected_test_sources:
        raise ValueError("Saved GNN predictions do not match the clean test partition")
    actual_by_source = {
        samples[index].source_index: samples[index].target for index in test_indices
    }
    if any(
        not np.isclose(actual_by_source[int(row.source_index)], row.actual)
        for row in gnn_predictions.itertuples()
    ):
        raise ValueError("Saved GNN prediction targets do not match the clean dataset")

    results.append(
        {
            "model": "GNN",
            "features": "graph+descriptors",
            "n_features": len(ATOM_FEATURE_COLUMNS) + len(EDGE_FEATURE_COLUMNS) + len(DESCRIPTOR_COLUMNS),
            "n_train": int(gnn_metrics["n_train"]),
            "n_test": int(gnn_metrics["n_test"]),
            "cv_folds": np.nan,
            "optuna_trials": np.nan,
            "cv_mae": np.nan,
            "test_mae": float(gnn_metrics["test_mae"]),
            "test_rmse": float(gnn_metrics["test_rmse"]),
            "test_r2": float(gnn_metrics["test_r2"]),
            "test_pearson_r": float(gnn_metrics["test_pearson_r"]),
            "test_spearman_r": float("nan"),
            "best_parameters": "saved GNN checkpoint",
        }
    )
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_directory / "comparison_metrics.csv", index=False)

    combined_predictions = pd.DataFrame(
        {
            "source_index": [samples[index].source_index for index in test_indices],
            "actual": [samples[index].target for index in test_indices],
        }
    )
    for frame in prediction_frames:
        combined_predictions = combined_predictions.merge(
            frame, on=["source_index", "actual"], how="left", validate="one_to_one"
        )
    gnn_prediction_column = gnn_predictions[
        ["source_index", "predicted"]
    ].rename(columns={"predicted": "GNN"})
    combined_predictions = combined_predictions.merge(
        gnn_prediction_column, on="source_index", how="left", validate="one_to_one"
    )
    combined_predictions.to_csv(
        output_directory / "heldout_predictions.csv", index=False
    )

    ordered = results_frame.sort_values("test_mae")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(
        ordered["model"] + " / " + ordered["features"],
        ordered["test_mae"],
        color=["#176B87" if model == "GNN" else "#648C79" for model in ordered["model"]],
    )
    axis.set_xlabel("Scaffold-held-out test MAE (pIC50)")
    axis.set_title("GNN and clean classical baselines")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output_directory / "comparison_mae.png", dpi=180)
    plt.close(figure)

    _write_report(
        output_directory / "audit_report.md",
        results,
        samples,
        rejected,
        train_indices,
        validation_indices,
        test_indices,
        trials,
        folds,
    )
    saved_splits["test"].to_csv(output_directory / "matched_gnn_test_rows.csv", index=False)
    return output_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    arguments = parser.parse_args()
    run_audit(trials=arguments.trials, folds=arguments.folds)


if __name__ == "__main__":
    main()