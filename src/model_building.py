
import argparse
import data_engineering as data_engineering_module
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
import joblib 
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import optuna as optuna
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent



OUTPUT_DIR = _resolve_project_root() / "output"


class Model:

    def __init__(self, path: Path):
        
        self.de = data_engineering_module.DataEng(path=path)
        self.df_morgan, self.df_rdkit = self.de.molecular_desc()
        self.output_path = Path(OUTPUT_DIR)
        self.figure_path = self.output_path / "figures"
        self.log_path = self.output_path / "logs"
        self.model_path = self.output_path / "models"
        for directory in [self.figure_path, self.log_path, self.model_path]:
            directory.mkdir(parents=True, exist_ok=True)
        
    def _model_from_trial(self, trial, model_type):
        if model_type == "RandomForest":
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])
            params = {
                'n_estimators' : trial.suggest_int('n_estimators', 100, 1000), # 2000 is excessively slow for CV
                'max_depth' : trial.suggest_int('max_depth', 20, 50),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20), 
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', None]), # None replaces 1.0 safely
                'bootstrap' : bootstrap,
                'criterion' : 'squared_error',
                'ccp_alpha' : trial.suggest_float('ccp_alpha', 1e-8, 1e-2, log=True), 
                'n_jobs' : -1, 
                'random_state' : 50
            }
            model = RandomForestRegressor(**params)
        
        if model_type == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1500),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 12), 
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),  
                "reg_lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True), 
                "n_jobs": -1,
                "random_state": 50,
            }
            model = xgb.XGBRegressor(**params)
        
        if model_type == "LightGBM":
            import lightgbm as lgb

            params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1500),
                "max_depth": trial.suggest_int("lgb_max_depth", 3, 12),
                "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 255),  
                "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("lgb_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgb_lambda", 1e-8, 10.0, log=True),
                "n_jobs": -1,
                "random_state": 50,
                "verbose": -1,  
            }
            model = lgb.LGBMRegressor(**params)
        
        return model

    def objective(self, trial, X_train, y_train, scaffolds_train, model_type):
        model = self._model_from_trial(trial, model_type)
        cv = GroupKFold(n_splits=5)
        score = cross_val_score(
            model, 
            X_train,
            y_train,
            groups=scaffolds_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        
        return score.mean()

    def _prepare_data(self, variant):
        if variant == "morgan":
            return self.de.scaffold_based_split()
        if variant == "descriptors":
            return self.de.scaffold_RDKit()
        if variant == "combined":
            return self.de.scaffold_combined()
        raise ValueError(f"Unknown feature variant: {variant}")

    @staticmethod
    def _model_columns(frame, variant):
        descriptor_columns = {
            "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
            "MaxPartialCharge", "MinPartialCharge", "NumHeteroatoms",
            "NumRotatableBonds", "FractionCSP3", "NumAromaticRings",
            "RingCount", "NumAliphaticNitrogens", "NumFormalCharge",
        }
        if variant == "morgan":
            return [column for column in frame if column.startswith("morgan_")]
        if variant == "descriptors":
            return [column for column in frame if column in descriptor_columns]
        return [
            column for column in frame
            if column.startswith("morgan_") or column in descriptor_columns
        ]

    def _optimize_and_evaluate(
        self,
        model_type,
        variant,
        study_name,
        file_name,
        n_trials=100,
        prepared_data=None,
    ):
        if prepared_data is None:
            X_train, y_train, X_test, y_test = self._prepare_data(variant)
        else:
            X_train, y_train, X_test, y_test = prepared_data
        model_columns = self._model_columns(X_train, variant)
        X_train_model = X_train[model_columns]
        X_test_model = X_test[model_columns]
        scaffold_values = X_train["Scaffold_InChI"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=50),
            study_name=study_name,
        )
        study.optimize(
            lambda trial: self.objective(
                trial,
                X_train_model,
                y_train,
                scaffold_values,
                model_type,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        final_model = self._model_from_trial(
            optuna.trial.FixedTrial(study.best_params),
            model_type,
        )
        final_model.fit(X_train_model, y_train)
        y_pred = final_model.predict(X_test_model)
        model_name = f"{model_type}_{variant}"
        metrics = self._save_run_outputs(
            model_name,
            variant,
            model_type,
            final_model,
            study,
            X_train,
            y_train,
            X_test,
            y_test,
            y_pred,
            file_name,
        )
        self.last_metrics = metrics
        print(f"{model_name} test MAE: {metrics['test_mae']:.4f}")
        return final_model, study, metrics["test_mae"]

    def _save_run_outputs(
        self,
        model_name,
        variant,
        model_type,
        model,
        study,
        X_train,
        y_train,
        X_test,
        y_test,
        y_pred,
        file_name,
    ):
        residuals = y_test - y_pred
        pearson_value, pearson_p = pearsonr(y_test, y_pred)
        spearman_value, spearman_p = spearmanr(y_test, y_pred)
        metrics = {
            "model": model_type,
            "features": variant,
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_r2": r2_score(y_test, y_pred),
            "pearson_r": pearson_value,
            "pearson_p": pearson_p,
            "spearman_r": spearman_value,
            "spearman_p": spearman_p,
            "cv_neg_mae": study.best_value,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
        }

        pd.DataFrame([metrics]).to_csv(
            self.log_path / f"{model_name}_metrics.csv", index=False
        )
        pd.DataFrame({
            "index": y_test.index,
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": residuals.values,
        }).to_csv(self.log_path / f"{model_name}_predictions.csv", index=False)
        pd.DataFrame(study.trials_dataframe()).to_csv(
            self.log_path / f"{model_name}_optuna_trials.csv", index=False
        )
        X_train.to_csv(self.log_path / f"{model_name}_train.csv", index=True)
        X_test.to_csv(self.log_path / f"{model_name}_test.csv", index=True)
        joblib.dump(model, self.model_path / file_name)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_test, y_pred, alpha=0.65, s=18)
        axes[0].set_title(f"{model_name}: predictions")
        axes[0].set_xlabel("Actual pIC50")
        axes[0].set_ylabel("Predicted pIC50")
        axes[1].hist(residuals, bins=30, alpha=0.8)
        axes[1].set_title(f"{model_name}: residuals")
        axes[1].set_xlabel("Actual - predicted")
        fig.tight_layout()
        fig.savefig(self.figure_path / f"{model_name}_diagnostics.png", dpi=200)
        plt.close(fig)

        return metrics



    def RFmodela(self):
        return self._optimize_and_evaluate(
            "RandomForest", "morgan", "RF_Scaffold_Split", "RFModelA-MORGAN-ONLY.joblib"
        )
    
    
    def RFmodelb(self):
        return self._optimize_and_evaluate(
            "RandomForest", "descriptors", "RF_RDKit_Scaffold", "RFModelB-DESCRIPTORS-ONLY.joblib"
        )


    def RFmodelc(self):
        return self._optimize_and_evaluate(
            "RandomForest", "combined", "RF_RDKitMorgan_Split", "RFModelC-COMBINED.joblib"
        )
    
    def XGBoostModelA(self):
        return self._optimize_and_evaluate(
            "XGBoost", "morgan", "XGB_Scaffold_Split", "XGBoostModelA-MORGAN-ONLY.joblib"
        )
    
    def XGBoostModelB(self):
        return self._optimize_and_evaluate(
            "XGBoost", "descriptors", "XGB_RDKit_Scaffold", "XGBoostModelB-DESCRIPTORS-ONLY.joblib"
        )

    def XGBooostModelB(self):
        return self.XGBoostModelB()
    
    def XGBoostModelC(self):
        return self._optimize_and_evaluate(
            "XGBoost", "combined", "XGB_RDKitMorgan_Split", "XGBoostModelC-COMBINED.joblib"
        )
    
    def LGBMModelA(self):
        return self._optimize_and_evaluate(
            "LightGBM", "morgan", "LGBM_Scaffold_Split", "LGBMModelA-MORGAN-ONLY.joblib"
        )
    
    def LGBMModelB(self):
        return self._optimize_and_evaluate(
            "LightGBM", "descriptors", "LGBM_RDKit_Scaffold", "LGBMModelB-DESCRIPTORS-ONLY.joblib"
        )
    
    def LGBMModelC(self):
        return self._optimize_and_evaluate(
            "LightGBM", "combined", "LGBM_RDKitMorgan_Split", "LGBMModelC-COMBINED.joblib"
        )

    def compare_models(self, n_trials=20):
        results = []
        configurations = [
            ("RandomForest", "morgan", "RFModelA-MORGAN-ONLY.joblib"),
            ("RandomForest", "descriptors", "RFModelB-DESCRIPTORS-ONLY.joblib"),
            ("RandomForest", "combined", "RFModelC-COMBINED.joblib"),
            ("XGBoost", "morgan", "XGBoostModelA-MORGAN-ONLY.joblib"),
            ("XGBoost", "descriptors", "XGBoostModelB-DESCRIPTORS-ONLY.joblib"),
            ("XGBoost", "combined", "XGBoostModelC-COMBINED.joblib"),
            ("LightGBM", "morgan", "LGBMModelA-MORGAN-ONLY.joblib"),
            ("LightGBM", "descriptors", "LGBMModelB-DESCRIPTORS-ONLY.joblib"),
            ("LightGBM", "combined", "LGBMModelC-COMBINED.joblib"),
        ]
        for model_type, variant, file_name in configurations:
            _, study, test_mae = self._optimize_and_evaluate(
                model_type,
                variant,
                f"{model_type}_{variant}_comparison",
                file_name,
                n_trials=n_trials,
            )
            results.append({
                "model": model_type,
                "features": variant,
                "cv_neg_mae": study.best_value,
                "test_mae": test_mae,
                "test_rmse": self.last_metrics["test_rmse"],
                "test_r2": self.last_metrics["test_r2"],
                "pearson_r": self.last_metrics["pearson_r"],
                "spearman_r": self.last_metrics["spearman_r"],
            })
        comparison = pd.DataFrame(results).sort_values("test_mae").reset_index(drop=True)
        comparison.to_csv(self.log_path / "model_comparison_metrics.csv", index=False)

        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=comparison, x="test_mae", y="model", hue="features", ax=axes[0])
        axes[0].set_title("Model comparison by test MAE")
        axes[0].set_xlabel("Test MAE")
        axes[0].set_ylabel("")
        sns.barplot(data=comparison, x="test_r2", y="model", hue="features", ax=axes[1])
        axes[1].set_title("Model comparison by test R2")
        axes[1].set_xlabel("Test R2")
        axes[1].set_ylabel("")
        fig.tight_layout()
        fig.savefig(self.figure_path / "model_comparison_metrics.png", dpi=200)
        plt.close(fig)
        self._save_statistical_visualizations(comparison)

        return comparison

    def _save_statistical_visualizations(self, comparison):
        prediction_frames = []
        for _, row in comparison.iterrows():
            prediction_path = self.log_path / (
                f"{row['model']}_{row['features']}_predictions.csv"
            )
            if prediction_path.exists():
                predictions = pd.read_csv(prediction_path)
                predictions["model"] = f"{row['model']}-{row['features']}"
                prediction_frames.append(predictions)

        if not prediction_frames:
            return

        import seaborn as sns

        prediction_table = pd.concat(prediction_frames, ignore_index=True)
        prediction_table.to_csv(
            self.log_path / "all_model_predictions.csv", index=False
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(data=prediction_table, x="model", y="residual", ax=axes[0])
        axes[0].tick_params(axis="x", rotation=75)
        axes[0].set_title("Residual distributions")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Actual - predicted")

        correlation_table = prediction_table.pivot_table(
            index="index", columns="model", values="predicted"
        )
        sns.heatmap(correlation_table.corr(), annot=True, cmap="vlag", center=0, ax=axes[1])
        axes[1].set_title("Prediction correlation")
        fig.tight_layout()
        fig.savefig(self.figure_path / "statistical_model_diagnostics.png", dpi=200)
        plt.close(fig)

    def _pipeline_dataset_path(self):
        pipeline_path = self.de.path.parent / f"{self.de.path.stem}_pipeline"
        if not pipeline_path.exists():
            self.de.finaldrop()
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline dataset was not generated: {pipeline_path}")
        return pipeline_path

    def GNNModel(self, **training_options):
        if __package__:
            from .GNNmodel import train_gnn
        else:
            from GNNmodel import train_gnn
        return train_gnn(
            dataset_path=self._pipeline_dataset_path(),
            **training_options,
        )

    def run_gnn_classical_audit(self, trials=8, folds=3):
        if __package__:
            from .baseline_gnn_audit import run_audit
        else:
            from baseline_gnn_audit import run_audit
        return run_audit(
            trials=trials,
            folds=folds,
            dataset_path=self._pipeline_dataset_path(),
        )

    def output_locations(self):
        return {
            "models": self.model_path,
            "figures": self.figure_path,
            "logs": self.log_path,
            "audit": self.output_path / "audit" / "GNN_classical_comparison",
        }

    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--trials", type=int, default=5)
        parser.add_argument("--gnn-audit", action="store_true")
        parser.add_argument("--gnn-trials", type=int, default=8)
        parser.add_argument("--gnn-folds", type=int, default=3)
        parser.add_argument("--gnn-epochs", type=int, default=150)
        parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        arguments = parser.parse_args(argv)

        root = _resolve_project_root()
        default_csv = root / "database" / "csv" / "Acetylcholinesterase_Homo_sapiens_cleaned.csv"
        if not default_csv.exists():
            raise FileNotFoundError(f"Expected cleaned dataset not found: {default_csv}")

        runner = Model(default_csv)
        if arguments.gnn_audit:
            runner.GNNModel(
                epochs=arguments.gnn_epochs,
                device_name=arguments.device,
            )
            audit_path = runner.run_gnn_classical_audit(
                trials=arguments.gnn_trials,
                folds=arguments.gnn_folds,
            )
            comparison = pd.read_csv(audit_path / "comparison_metrics.csv")
            print(comparison.to_string(index=False))
            print(f"GNN comparison audit: {audit_path}")
            return comparison

        comparison = runner.compare_models(n_trials=arguments.trials)
        print(comparison.to_string(index=False))
        return comparison