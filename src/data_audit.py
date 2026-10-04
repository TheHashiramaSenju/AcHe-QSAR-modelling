from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

ALLOWED_UNITS = ["10'5pM", "10'6pM", "10'3pM", "nM"]
UNIT_TO_NM = {"10'5pM": 100.0, "10'6pM": 1000.0, "10'3pM": 1.0, "nM": 1.0}
VALIDITY_COMMENTS = [
    "Values appear to be an order of magnitude different from previously reported, so units may be incorrect",
    "Potential transcription error",
]
VALIDITY_DESCRIPTIONS = [
    "Values for this activity type are unusually large/small, so may not be accurate",
    "Values appear to be an order of magnitude different from previously reported, so units may be incorrect",
]
MAD_THRESHOLD = 1.0
RELATION_FILTER = "="
ACTIVITY_THRESHOLD = 10000.0
SCAFFOLD_TRAIN_FRACTION = 0.8
MORGAN_RADIUS = 2
MORGAN_SIZE = 1024
DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "FractionCSP3": Lipinski.FractionCSP3,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "RingCount": Lipinski.RingCount,
    "NumAliphaticNitrogens": None,
    "NumFormalCharge": Chem.GetFormalCharge,
}


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def molecule_key_count(frame: pd.DataFrame) -> int:
    return int(frame["InChIkey"].nunique(dropna=True)) if "InChIkey" in frame else 0


def smiles_key_count(frame: pd.DataFrame) -> int:
    column = "cleaned_smiles" if "cleaned_smiles" in frame else "canonical_smiles"
    return int(frame[column].nunique(dropna=True)) if column in frame else 0


def number_stats(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {"count": 0, "missing": int(numeric.isna().sum()), "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(finite.size),
        "missing": int(numeric.isna().sum()),
        "nonfinite": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
    }


def target_stats(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    quantiles = finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if not finite.empty else pd.Series(dtype=float)
    return {
        "count": int(len(values)),
        "missing": int(numeric.isna().sum()),
        "nonfinite": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
        "min": float(finite.min()) if not finite.empty else None,
        "max": float(finite.max()) if not finite.empty else None,
        "mean": float(finite.mean()) if not finite.empty else None,
        "median": float(finite.median()) if not finite.empty else None,
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "quantiles": {f"{int(q * 100)}%": float(value) for q, value in quantiles.items()},
    }


def stage_record(name: str, frame: pd.DataFrame, before: pd.DataFrame | None, order: int) -> dict:
    before_count = len(frame) if before is None else len(before)
    removed = max(0, before_count - len(frame))
    record = {
        "stage_order": order,
        "stage": name,
        "before_rows": before_count,
        "rows_after": int(len(frame)),
        "rows_removed": removed,
        "percent_removed": 0.0 if before_count == 0 else 100.0 * removed / before_count,
        "unique_InChIKeys": molecule_key_count(frame),
        "unique_cleaned_SMILES": smiles_key_count(frame),
    }
    for column in ["canonical_smiles", "cleaned_smiles", "InChIkey", "standard_units", "standard_value", "IC50", "PIC50"]:
        if column in frame:
            record[f"missing_{column}"] = int(frame[column].isna().sum())
    for column in ["standard_value", "IC50", "PIC50"]:
        if column in frame:
            for metric, value in number_stats(frame[column]).items():
                record[f"{column}_{metric}"] = value
    return record


def safe_smiles(value, remover: SaltRemover):
    if not isinstance(value, str) or not value.strip():
        return None
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None
    return Chem.MolToSmiles(remover.StripMol(molecule))


def safe_inchikey(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToInchiKey(molecule)
    except Exception:
        return None


def num_aliphatic_nitrogens(molecule):
    return sum(atom.GetAtomicNum() == 7 and not atom.GetIsAromatic() for atom in molecule.GetAtoms())


def descriptor_values(smiles):
    empty = {name: None for name in DESCRIPTOR_FUNCTIONS}
    if not isinstance(smiles, str) or not smiles.strip():
        return empty
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return empty
    molecule = SaltRemover().StripMol(molecule)
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)
        results = {}
        for name, function in DESCRIPTOR_FUNCTIONS.items():
            value = num_aliphatic_nitrogens(molecule) if function is None else function(molecule)
            results[name] = value if np.isfinite(value) else None
        return results
    except Exception:
        return empty


def scaffold_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold)


def stable_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    stable = frame[available].copy()
    stable = stable.sort_values(available, kind="mergesort", na_position="last")
    return hashlib.sha256(stable.to_csv(index=False, float_format="%.12g").encode("utf-8")).hexdigest()


def git_value(root: Path, command: list[str]):
    try:
        return subprocess.check_output(command, cwd=root, stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


class DataAudit:
    def __init__(self, raw_csv: Path, target_id: str, root: Path, output_root: Path | None = None):
        self.raw_csv = raw_csv.resolve()
        self.target_id = target_id
        self.root = root.resolve()
        self.audit_root = (output_root or self.root / "audit").resolve()
        self.run_dir = self._new_run_directory()
        self.stages = []
        self.findings = []
        self.snapshots = {}

    def _new_run_directory(self) -> Path:
        self.audit_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.audit_root / stamp
        suffix = 1
        while candidate.exists():
            candidate = self.audit_root / f"{stamp}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True)
        return candidate

    def finding(self, severity: str, check: str, message: str, affected_records=None, recommendation=None):
        self.findings.append({
            "severity": severity,
            "check": check,
            "message": message,
            "affected_records": affected_records,
            "recommendation": recommendation,
        })

    def record_stage(self, name, current, previous=None):
        row = stage_record(name, current, previous, len(self.stages) + 1)
        self.stages.append(row)
        self.snapshots[name] = current.copy()

    def _cleaning_stages(self):
        raw = pd.read_csv(self.raw_csv)
        self.record_stage("Raw saved ChEMBL retrieval snapshot", raw)

        data = raw.dropna(axis=1, how="all").copy()
        validity_comment = data.get("data_validity_comment", pd.Series(index=data.index, dtype=object))
        validity_description = data.get("data_validity_description", pd.Series(index=data.index, dtype=object))
        invalid_mask = validity_comment.isin(VALIDITY_COMMENTS) | validity_description.isin(VALIDITY_DESCRIPTIONS)
        data = data.loc[~invalid_mask].copy()
        self.record_stage("Initial validity/null handling", data, raw)

        remover = SaltRemover()
        if "canonical_smiles" not in data:
            raise KeyError("Raw CSV has no canonical_smiles column required by current pipeline.")
        data["cleaned_smiles"] = data["canonical_smiles"].apply(lambda value: safe_smiles(value, remover))
        data = data.drop(columns=["canonical_smiles"])
        self.record_stage("Salt removal and SMILES cleaning", data, self.snapshots["Initial validity/null handling"])

        data["InChIkey"] = data["cleaned_smiles"].apply(safe_inchikey)
        self.record_stage("InChIKey generation", data, self.snapshots["Salt removal and SMILES cleaning"])

        before = data.copy()
        if "potential_duplicate" in data:
            data = data.loc[data["potential_duplicate"] != 1].copy()
        self.record_stage("potential_duplicate removal", data, before)

        if "standard_units" not in data:
            raise KeyError("Raw data has no standard_units column.")
        data["source_standard_units"] = data["standard_units"]
        data["standard_value"] = pd.to_numeric(data["standard_value"], errors="coerce")
        data = data.loc[data["standard_units"].isin(ALLOWED_UNITS)].copy()
        factors = data["standard_units"].map(UNIT_TO_NM)
        data["IC50"] = data["standard_value"] * factors
        data["standard_units"] = "nM"
        self.record_stage("Supported-unit filter and IC50 normalization", data, self.snapshots["potential_duplicate removal"])

        standard_type = data["standard_type"]
        conditions = [
            (standard_type == "IC50") & (data["IC50"] > 0),
            standard_type == "Log IC50",
            standard_type == "pIC50",
            standard_type == "Log IC50(nM)",
        ]
        choices = [
            -np.log10(data["IC50"] * 10 ** -9),
            -data["standard_value"],
            data["standard_value"],
            9 - data["standard_value"],
        ]
        data["PIC50"] = np.select(conditions, choices)
        data["PIC50"] = data["PIC50"].round(5)
        self.record_stage("pIC50 conversion", data, self.snapshots["Supported-unit filter and IC50 normalization"])

        before = data.copy()
        data["median"] = data.groupby("InChIkey")["PIC50"].transform("median")
        data["compared_median"] = abs(data["median"] - data["PIC50"])
        data["group_MAD"] = data.groupby("InChIkey")["compared_median"].transform("median")
        self.record_stage("InChIKey replicate median/MAD calculation", data, before)

        mad_before = data.copy()
        data = data.loc[data["group_MAD"] <= MAD_THRESHOLD].copy()
        data["PIC50"] = data["median"]
        self.record_stage("Group MAD <= 1.0 filter", data, mad_before)

        before = data.copy()
        data = data.drop_duplicates(subset=["InChIkey", "cleaned_smiles", "PIC50"]).copy()
        data = data.drop(columns=["median", "compared_median", "group_MAD"])
        self.record_stage("Exact duplicate removal after MAD", data, before)

        before = data.copy()
        data = data.loc[data["standard_relation"] == RELATION_FILTER].copy()
        self.record_stage('standard_relation == "=" filter', data, before)

        self.interpretation_cohort = data.copy()
        before = data.copy()
        current_final = data.loc[data["standard_value"] <= ACTIVITY_THRESHOLD].copy()
        self.record_stage("Current standard_value <= 10000 activity filter", current_final, before)
        return raw, current_final

    def _unit_audit(self, raw, after_duplicates, supported):
        records = []
        for scope, frame in [
            ("raw retrieval snapshot", raw),
            ("pre-normalization after duplicate flag removal", after_duplicates),
        ]:
            if "standard_units" not in frame:
                continue
            for unit, group in frame.groupby("standard_units", dropna=False, sort=False):
                values = pd.to_numeric(group.get("standard_value", pd.Series(dtype=float)), errors="coerce")
                records.append({
                    "scope": scope,
                    "original_standard_units": "<NULL>" if pd.isna(unit) else str(unit),
                    "records": len(group),
                    "percent": 0.0 if len(frame) == 0 else 100 * len(group) / len(frame),
                    "standard_value_min": values.min(),
                    "standard_value_max": values.max(),
                    "standard_value_median": values.median(),
                    "supported_by_current_code": unit in ALLOWED_UNITS if not pd.isna(unit) else False,
                })
        pd.DataFrame(records).to_csv(self.run_dir / "unit_distribution.csv", index=False)

        converted = self.snapshots.get("pIC50 conversion")
        representatives = []
        if not supported.empty:
            for unit in ALLOWED_UNITS:
                group = supported.loc[supported["source_standard_units"] == unit].copy()
                if converted is not None and not group.empty:
                    group = converted.loc[group.index].copy()
                if group.empty:
                    continue
                sample = group.loc[pd.to_numeric(group["standard_value"], errors="coerce").gt(0)].head(1)
                if sample.empty:
                    sample = group.head(1)
                row = sample.iloc[0]
                expected = pd.to_numeric(row["standard_value"], errors="coerce") * UNIT_TO_NM[unit]
                calculated_pic50 = -np.log10(expected * 1e-9) if expected > 0 else np.nan
                normalized_pic50 = pd.to_numeric(row.get("PIC50", np.nan), errors="coerce")
                representatives.append({
                    "original_standard_units": unit,
                    "standard_value": row["standard_value"],
                    "code_conversion_factor_to_nM": UNIT_TO_NM[unit],
                    "normalized_IC50_nM": row["IC50"],
                    "expected_IC50_nM_from_code_factor": expected,
                    "normalized_pIC50": normalized_pic50 if pd.notna(normalized_pic50) else None,
                    "pIC50_recomputed_from_normalized_IC50": round(float(calculated_pic50), 5) if np.isfinite(calculated_pic50) else None,
                    "conversion_matches_code": bool(np.isclose(row["IC50"], expected, equal_nan=True)),
                    "pIC50_matches_code": bool(np.isclose(normalized_pic50, round(calculated_pic50, 5), equal_nan=True)) if pd.notna(normalized_pic50) and np.isfinite(calculated_pic50) else False,
                })
        pd.DataFrame(representatives).to_csv(self.run_dir / "unit_conversion_examples.csv", index=False)

    def _target_distribution(self, final, split_assignments):
        rows = []
        scopes = [("final_current_filter", final["PIC50"])]
        if split_assignments is not None:
            for split in ["train", "test"]:
                indices = split_assignments.loc[split_assignments["split"] == split, "source_index"]
                scopes.append((split, final.loc[indices, "PIC50"]))
        for scope, values in scopes:
            stats = target_stats(values)
            rows.append({"scope": scope, **{key: value for key, value in stats.items() if key != "quantiles"}})
            for quantile, value in stats["quantiles"].items():
                rows.append({"scope": scope, "statistic": f"quantile_{quantile}", "value": value})
        target_table = pd.DataFrame(rows)
        target_table.to_csv(self.run_dir / "target_distribution.csv", index=False)
        return {scope: target_stats(values) for scope, values in scopes}

    def _duplicate_audit(self, converted, after_mad, final, identity_frame):
        keyed = converted.loc[converted["InChIkey"].notna()].copy()
        group = keyed.groupby("InChIkey", sort=False)["PIC50"]
        measurements = group.agg(
            number_of_measurements="size",
            median_pIC50="median",
            mean_pIC50="mean",
            std_pIC50="std",
            min_pIC50="min",
            max_pIC50="max",
        )
        deviations = keyed.assign(
            _median=keyed.groupby("InChIkey")["PIC50"].transform("median")
        )
        deviations["_abs_dev"] = abs(deviations["PIC50"] - deviations["_median"])
        mad_values = deviations.groupby("InChIkey")["_abs_dev"].median().rename("MAD")
        measurements = measurements.join(mad_values)
        measurements["pIC50_range"] = measurements["max_pIC50"] - measurements["min_pIC50"]
        smiles_by_key = keyed.groupby("InChIkey")["cleaned_smiles"].first()
        measurements = measurements.join(smiles_by_key.rename("cleaned_smiles"))
        measurements = measurements.reset_index()
        measurements["MAD_over_1"] = measurements["MAD"] > MAD_THRESHOLD
        measurements.to_csv(self.run_dir / "duplicate_analysis.csv", index=False)
        measurements.sort_values("pIC50_range", ascending=False).head(20).rename(columns={
            "number_of_measurements": "number_of_measurements",
            "median_pIC50": "median_pIC50",
            "pIC50_range": "range",
        })[["InChIkey", "cleaned_smiles", "number_of_measurements", "min_pIC50", "max_pIC50", "median_pIC50", "MAD", "range"]].to_csv(
            self.run_dir / "high_variability_molecules.csv", index=False
        )

        replicate_counts = measurements["number_of_measurements"].value_counts().sort_index()
        self.replicate_distribution = pd.DataFrame({
            "measurements_per_InChIKey": replicate_counts.index,
            "InChIKey_groups": replicate_counts.values,
        })
        self.replicate_distribution.to_csv(self.run_dir / "replicate_count_distribution.csv", index=False)

        exact_sets = [
            ("cleaned_smiles", ["cleaned_smiles"]),
            ("InChIkey", ["InChIkey"]),
            ("PIC50", ["PIC50"]),
            ("cleaned_smiles + PIC50", ["cleaned_smiles", "PIC50"]),
            ("InChIkey + PIC50", ["InChIkey", "PIC50"]),
        ]
        duplicate_rows = []
        for phase, frame in [("after_pIC50_before_current_resolution", converted), ("after_current_MAD_and_exact_dedup", after_mad), ("final_current_filter", final)]:
            for label, columns in exact_sets:
                duplicate_rows.append({
                    "phase": phase,
                    "signature": label,
                    "rows": len(frame),
                    "distinct_signatures": int(frame.drop_duplicates(columns).shape[0]),
                    "duplicate_excess_rows": int(len(frame) - frame.drop_duplicates(columns).shape[0]),
                })
        pd.DataFrame(duplicate_rows).to_csv(self.run_dir / "exact_duplicate_checks.csv", index=False)

        mappings = []
        identity_keyed = identity_frame.loc[identity_frame["InChIkey"].notna()].copy()
        for key, subframe in identity_keyed.groupby("InChIkey", sort=False):
            smiles = subframe["cleaned_smiles"].dropna().unique()
            if len(smiles) > 1:
                for value in smiles:
                    mappings.append({"mapping_direction": "InChIKey_to_multiple_SMILES", "InChIkey": key, "cleaned_smiles": value})
        for smiles, subframe in identity_keyed.groupby("cleaned_smiles", sort=False, dropna=False):
            keys = subframe["InChIkey"].dropna().unique()
            if len(keys) > 1:
                for key in keys:
                    mappings.append({"mapping_direction": "SMILES_to_multiple_InChIKeys", "InChIkey": key, "cleaned_smiles": smiles})
        pd.DataFrame(mappings, columns=["mapping_direction", "InChIkey", "cleaned_smiles"]).to_csv(self.run_dir / "identity_mappings.csv", index=False)
        self.measurements = measurements
        return measurements

    def _filtering_audit(self, cohort):
        raw_value = pd.to_numeric(cohort["standard_value"], errors="coerce")
        normalized = pd.to_numeric(cohort["IC50"], errors="coerce")
        keep_a = raw_value <= ACTIVITY_THRESHOLD
        keep_b = normalized <= ACTIVITY_THRESHOLD
        a_only = cohort.loc[keep_a & ~keep_b].copy()
        b_only = cohort.loc[keep_b & ~keep_a].copy()
        different = pd.concat([a_only, b_only]).copy()
        keep_a_table = cohort.loc[keep_a]
        keep_b_table = cohort.loc[keep_b]
        columns = ["InChIkey", "cleaned_smiles", "standard_value", "source_standard_units", "IC50", "PIC50", "standard_relation", "standard_type"]
        different = different.reindex(columns=columns)
        different = different.rename(columns={"source_standard_units": "original_standard_units"})
        different.insert(0, "retained_by_current_raw_value_filter", different.index.isin(a_only.index))
        different.to_csv(self.run_dir / "filtering_differences.csv", index=False)

        by_unit = []
        for unit, group in cohort.groupby("source_standard_units", dropna=False, sort=False):
            unit_a = pd.to_numeric(group["standard_value"], errors="coerce") <= ACTIVITY_THRESHOLD
            unit_b = pd.to_numeric(group["IC50"], errors="coerce") <= ACTIVITY_THRESHOLD
            by_unit.append({
                "original_standard_units": "<NULL>" if pd.isna(unit) else str(unit),
                "candidate_records": len(group),
                "retained_A_raw_standard_value": int(unit_a.sum()),
                "retained_B_normalized_IC50": int(unit_b.sum()),
                "retained_only_by_A": int((unit_a & ~unit_b).sum()),
                "retained_only_by_B": int((unit_b & ~unit_a).sum()),
                "removed_only_by_A": int((~unit_a & unit_b).sum()),
                "removed_only_by_B": int((~unit_b & unit_a).sum()),
            })
        pd.DataFrame(by_unit).to_csv(self.run_dir / "filtering_differences_by_unit.csv", index=False)

        def summary(name, frame):
            return {
                "filter": name,
                "records_retained": int(len(frame)),
                "unique_InChIKeys_retained": molecule_key_count(frame),
                "pIC50_min": number_stats(frame["PIC50"])["min"],
                "pIC50_max": number_stats(frame["PIC50"])["max"],
                "pIC50_mean": number_stats(frame["PIC50"])["mean"],
                "pIC50_median": number_stats(frame["PIC50"])["median"],
                "pIC50_std": number_stats(frame["PIC50"])["std"],
            }
        comparison = [summary("A_standard_value <= 10000", keep_a_table), summary("B_IC50_nM <= 10000", keep_b_table)]
        pd.DataFrame(comparison).to_csv(self.run_dir / "filtering_comparison.csv", index=False)
        affected = int(len(different))
        affected_pct = 0.0 if len(cohort) == 0 else 100 * affected / len(cohort)
        stats = {
            "candidate_records_after_relation_filter": int(len(cohort)),
            "A_retained": int(keep_a.sum()),
            "B_retained": int(keep_b.sum()),
            "removed_by_A_only": int((~keep_a & keep_b).sum()),
            "removed_by_B_only": int((~keep_b & keep_a).sum()),
            "records_differently_classified": affected,
            "percent_candidate_records_differently_classified": affected_pct,
            "unique_InChIKeys_affected": int(different["InChIkey"].nunique(dropna=True)),
            "A_summary": comparison[0],
            "B_summary": comparison[1],
        }
        if affected:
            self.finding(
                "WARNING",
                "raw_standard_value_vs_normalized_IC50_threshold",
                f"The current raw-value threshold and a normalized nM threshold classify {affected} post-relation records differently ({affected_pct:.3f}% of the candidate cohort). Current behavior remains unchanged.",
                affected,
                "Review the original unit semantics and intended threshold scale before changing pipeline behavior.",
            )
        return keep_a_table.copy(), keep_b_table.copy(), stats

    def _scaffold_audit(self, final):
        matrix = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_SIZE)
        scaffold_values = final["cleaned_smiles"].apply(scaffold_smiles)
        scaffold_keys = scaffold_values.apply(safe_inchikey).astype("object")
        scaffold_keys = scaffold_keys.where(scaffold_keys.notna(), "__NO_SCAFFOLD__")
        scaffold_groups = pd.DataFrame(
            {"Scaffold_InChI": scaffold_keys}, index=final.index
        ).groupby("Scaffold_InChI").groups
        ordered_scaffolds = sorted(
            scaffold_groups.keys(),
            key=lambda key: len(scaffold_groups[key]),
            reverse=True,
        )
        scaffold_sizes = pd.Series({key: len(indices) for key, indices in scaffold_groups.items()})
        train_indices = []
        test_indices = []
        target_size = int(SCAFFOLD_TRAIN_FRACTION * len(final))
        for key in ordered_scaffolds:
            row_indices = final.index[scaffold_keys == key]
            if len(train_indices) < target_size:
                train_indices.extend(row_indices)
            else:
                test_indices.extend(row_indices)
        train_set = set(train_indices)
        test_set = set(test_indices)
        train_indices_ordered = sorted(train_set)
        test_indices_ordered = sorted(test_set)
        train_scaffolds = set(scaffold_keys.loc[train_indices_ordered].tolist()) if train_set else set()
        test_scaffolds = set(scaffold_keys.loc[test_indices_ordered].tolist()) if test_set else set()
        overlap_scaffolds = train_scaffolds.intersection(test_scaffolds)
        train_keys = set(final.loc[train_indices_ordered, "InChIkey"].dropna()) if train_set else set()
        test_keys = set(final.loc[test_indices_ordered, "InChIkey"].dropna()) if test_set else set()
        train_smiles = set(final.loc[train_indices_ordered, "cleaned_smiles"].dropna()) if train_set else set()
        test_smiles = set(final.loc[test_indices_ordered, "cleaned_smiles"].dropna()) if test_set else set()

        assignment = pd.DataFrame({
            "source_index": final.index,
            "InChIkey": final["InChIkey"].values,
            "cleaned_smiles": final["cleaned_smiles"].values,
            "PIC50": final["PIC50"].values,
            "scaffold_InChI": scaffold_keys.values,
            "scaffold_size": scaffold_keys.map(scaffold_sizes).values,
            "split": ["train" if index in train_set else "test" for index in final.index],
        })
        assignment.to_csv(self.run_dir / "scaffold_assignments.csv", index=False)

        sizes = pd.DataFrame({"scaffold_InChI": scaffold_sizes.index, "molecules": scaffold_sizes.values})
        split_by_scaffold = assignment.groupby("scaffold_InChI", dropna=False)["split"].first()
        sizes["split"] = sizes["scaffold_InChI"].map(split_by_scaffold)
        sizes.to_csv(self.run_dir / "scaffold_analysis.csv", index=False)

        top_contribution = {}
        for percent in [1, 5, 10]:
            n_top = max(1, math.ceil(len(scaffold_sizes) * percent / 100)) if len(scaffold_sizes) else 0
            top_contribution[f"largest_{percent}_percent_scaffolds_molecule_fraction"] = (
                float(scaffold_sizes.sort_values(ascending=False).head(n_top).sum() / len(final)) if len(final) else None
            )
        train_sizes = assignment.loc[assignment["split"] == "train", "scaffold_size"]
        test_sizes = assignment.loc[assignment["split"] == "test", "scaffold_size"]
        overlap_keys = train_keys.intersection(test_keys)
        overlap_smiles = train_smiles.intersection(test_smiles)
        stats = {
            "unique_scaffolds": int(len(scaffold_sizes)),
            "largest_scaffold_size": int(scaffold_sizes.max()) if len(scaffold_sizes) else 0,
            "median_scaffold_size": float(scaffold_sizes.median()) if len(scaffold_sizes) else None,
            "mean_scaffold_size": float(scaffold_sizes.mean()) if len(scaffold_sizes) else None,
            "scaffolds_train": int(assignment.loc[assignment["split"] == "train", "scaffold_InChI"].nunique()),
            "scaffolds_test": int(assignment.loc[assignment["split"] == "test", "scaffold_InChI"].nunique()),
            "molecules_train": int(len(train_set)),
            "molecules_test": int(len(test_set)),
            "train_fraction": float(len(train_set) / len(final)) if len(final) else None,
            "test_fraction": float(len(test_set) / len(final)) if len(final) else None,
            "allocation_order": "Scaffolds are currently ordered by group size before allocation.",
            "train_scaffold_size_mean": float(train_sizes.mean()) if not train_sizes.empty else None,
            "train_scaffold_size_median": float(train_sizes.median()) if not train_sizes.empty else None,
            "test_scaffold_size_mean": float(test_sizes.mean()) if not test_sizes.empty else None,
            "test_scaffold_size_median": float(test_sizes.median()) if not test_sizes.empty else None,
            "largest_train_scaffold": int(train_sizes.max()) if not train_sizes.empty else 0,
            "largest_test_scaffold": int(test_sizes.max()) if not test_sizes.empty else 0,
            "shared_scaffold_count": len(overlap_scaffolds),
            "shared_scaffolds": sorted(map(str, overlap_scaffolds)),
            "shared_InChIKey_count": len(overlap_keys),
            "shared_cleaned_SMILES_count": len(overlap_smiles),
            **top_contribution,
        }
        self.split_assignments = assignment
        self.scaffold_sizes = scaffold_sizes
        if overlap_scaffolds:
            self.finding("CRITICAL", "scaffold_split_overlap", f"{len(overlap_scaffolds)} scaffolds are present in both splits.", len(overlap_scaffolds), "Inspect scaffold assignment and group labels.")
        else:
            self.finding("PASS", "scaffold_split_overlap", "No scaffold overlap between train and test.", 0)
        if overlap_keys or overlap_smiles:
            self.finding("CRITICAL", "molecule_split_overlap", f"Train/test share {len(overlap_keys)} InChIKeys and {len(overlap_smiles)} cleaned SMILES.", len(overlap_keys) + len(overlap_smiles), "Inspect row identity and split membership.")
        else:
            self.finding("PASS", "molecule_split_overlap", "No InChIKey or cleaned SMILES overlap between train and test.", 0)
        target_rows = []
        for scope, subset in [("train", final.loc[train_indices_ordered]), ("test", final.loc[test_indices_ordered])]:
            summary = target_stats(subset["PIC50"])
            target_rows.append({"scope": scope, **{key: value for key, value in summary.items() if key != "quantiles"}})
            for name, value in summary["quantiles"].items():
                target_rows.append({"scope": scope, "statistic": f"quantile_{name}", "value": value})
        pd.DataFrame(target_rows).to_csv(self.run_dir / "split_target_distribution.csv", index=False)
        return stats, matrix

    def _feature_audit(self, final, generator, structure_frame):
        fingerprints = []
        active_bits = []
        fingerprint_bytes = []
        invalid_smiles = []
        zero_atoms = []
        disconnected = []
        unusual_elements = []
        atom_counts = []
        common_elements = {1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 25, 26, 27, 29, 30, 34, 35, 53}
        for index, smiles in structure_frame["cleaned_smiles"].items():
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
            if molecule is None:
                invalid_smiles.append(index)
                continue
            atom_count = molecule.GetNumAtoms()
            atom_counts.append(atom_count)
            if atom_count == 0:
                zero_atoms.append(index)
            if len(Chem.GetMolFrags(molecule)) > 1:
                disconnected.append(index)
            unusual = sorted({atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() not in common_elements})
            if unusual:
                unusual_elements.append({"source_index": index, "InChIkey": structure_frame.at[index, "InChIkey"], "cleaned_smiles": smiles, "elements": ";".join(unusual)})

        structure_warnings = []
        for index in invalid_smiles:
            structure_warnings.append({"source_index": index, "InChIkey": structure_frame.at[index, "InChIkey"], "cleaned_smiles": structure_frame.at[index, "cleaned_smiles"], "warning": "RDKit parse failed or cleaned SMILES is empty"})
        for index in zero_atoms:
            structure_warnings.append({"source_index": index, "InChIkey": structure_frame.at[index, "InChIkey"], "cleaned_smiles": structure_frame.at[index, "cleaned_smiles"], "warning": "Molecule has zero atoms"})
        for index in disconnected:
            structure_warnings.append({"source_index": index, "InChIkey": structure_frame.at[index, "InChIkey"], "cleaned_smiles": structure_frame.at[index, "cleaned_smiles"], "warning": "Disconnected fragments remain after salt removal"})
        for item in unusual_elements:
            structure_warnings.append({**item, "warning": "Unusual element(s): " + item["elements"]})

        for index, smiles in final["cleaned_smiles"].items():
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
            fingerprint = np.asarray(generator.GetFingerprint(molecule), dtype=np.uint8) if molecule is not None else np.zeros(MORGAN_SIZE, dtype=np.uint8)
            active = int(fingerprint.sum())
            fingerprints.append(fingerprint)
            active_bits.append(active)
            fingerprint_bytes.append(fingerprint.tobytes())
        fingerprint_matrix = np.vstack(fingerprints) if fingerprints else np.empty((0, MORGAN_SIZE), dtype=np.uint8)
        fingerprint_hashes = pd.Series(fingerprint_bytes)
        fp_duplicates = int(fingerprint_hashes.duplicated(keep="first").sum())
        fp_unique = int(fingerprint_hashes.nunique())
        pd.DataFrame(structure_warnings, columns=["source_index", "InChIkey", "cleaned_smiles", "warning", "elements"]).to_csv(self.run_dir / "structure_warnings.csv", index=False)

        descriptor_rows = [descriptor_values(smiles) for smiles in final["cleaned_smiles"]]
        descriptors = pd.DataFrame(descriptor_rows, index=final.index, columns=list(DESCRIPTOR_FUNCTIONS))
        feature_rows = []
        for name in descriptors.columns:
            values = pd.to_numeric(descriptors[name], errors="coerce")
            finite = values[np.isfinite(values)]
            variance = float(finite.var(ddof=0)) if not finite.empty else None
            feature_rows.append({
                "feature": name,
                "family": "RDKit descriptor",
                "dtype": str(descriptors[name].dtype),
                "missing": int(values.isna().sum()),
                "nonfinite": int((values.notna() & ~np.isfinite(values)).sum()),
                "min": float(finite.min()) if not finite.empty else None,
                "max": float(finite.max()) if not finite.empty else None,
                "mean": float(finite.mean()) if not finite.empty else None,
                "median": float(finite.median()) if not finite.empty else None,
                "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
                "variance": variance,
                "zero_variance": bool(variance == 0) if variance is not None else None,
                "near_zero_variance_lt_1e-8": bool(variance is not None and 0 <= variance < 1e-8),
                "suspiciously_large_magnitude_abs_ge_1e6": bool(not finite.empty and finite.abs().max() >= 1e6),
            })
        for column_index in range(fingerprint_matrix.shape[1]):
            values = fingerprint_matrix[:, column_index] if len(fingerprint_matrix) else np.array([], dtype=np.uint8)
            feature_rows.append({
                "feature": f"morgan_{column_index}",
                "family": "Morgan bit",
                "dtype": str(fingerprint_matrix.dtype),
                "missing": 0,
                "nonfinite": 0,
                "min": int(values.min()) if values.size else None,
                "max": int(values.max()) if values.size else None,
                "mean": float(values.mean()) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "std": float(values.std(ddof=1)) if values.size > 1 else None,
                "variance": float(values.var()) if values.size else None,
                "zero_variance": bool(values.min() == values.max()) if values.size else None,
                "near_zero_variance_lt_1e-8": None,
                "suspiciously_large_magnitude_abs_ge_1e6": False,
            })
        pd.DataFrame(feature_rows).to_csv(self.run_dir / "feature_sanity.csv", index=False)

        descriptor_stats = descriptors.describe().replace({np.nan: None}).to_dict()
        chosen_features = [f"morgan_{index}" for index in range(MORGAN_SIZE)] + list(DESCRIPTOR_FUNCTIONS)
        fingerprint_frame = pd.DataFrame(
            fingerprint_matrix,
            index=final.index,
            columns=[f"morgan_{index}" for index in range(MORGAN_SIZE)],
        )
        self.model_feature_frame = pd.concat([fingerprint_frame, descriptors], axis=1)
        pd.DataFrame({"model_feature_column": chosen_features}).to_csv(self.run_dir / "model_feature_columns.csv", index=False)
        data = {
            "molecules": int(len(final)),
            "morgan": {
                "molecules": int(fingerprint_matrix.shape[0]),
                "columns": int(fingerprint_matrix.shape[1]),
                "expected_columns": MORGAN_SIZE,
                "dtype": str(fingerprint_matrix.dtype),
                "missing": int(np.isnan(fingerprint_matrix).sum()) if fingerprint_matrix.size else 0,
                "nonfinite": int((~np.isfinite(fingerprint_matrix)).sum()) if fingerprint_matrix.size else 0,
                "all_zero_fingerprints": int(sum(bits == 0 for bits in active_bits)),
                "duplicate_fingerprints_excess": fp_duplicates,
                "unique_fingerprints": fp_unique,
                "average_active_bits": float(np.mean(active_bits)) if active_bits else None,
                "min_active_bits": int(min(active_bits)) if active_bits else None,
                "max_active_bits": int(max(active_bits)) if active_bits else None,
                "configuration_matches_radius_2_size_1024": fingerprint_matrix.shape[1] == MORGAN_SIZE,
            },
            "descriptors": {
                "columns": list(DESCRIPTOR_FUNCTIONS),
                "count": len(DESCRIPTOR_FUNCTIONS),
                "summary": descriptor_stats,
                "missing_values": int(descriptors.isna().sum().sum()),
                "nonfinite_values": int((~np.isfinite(descriptors.select_dtypes(include=[np.number]))).sum().sum()),
            },
            "structures": {
                "failed_RDKit_parse": len(invalid_smiles),
                "empty_or_invalid_cleaned_SMILES": int(structure_frame["cleaned_smiles"].isna().sum() + structure_frame["cleaned_smiles"].eq("").sum()),
                "zero_atom_molecules": len(zero_atoms),
                "disconnected_molecules_after_salt_removal": len(disconnected),
                "unusual_element_rows": len(unusual_elements),
                "atom_count_min": min(atom_counts) if atom_counts else None,
                "atom_count_median": float(np.median(atom_counts)) if atom_counts else None,
                "atom_count_max": max(atom_counts) if atom_counts else None,
                "atom_count_over_100": sum(count > 100 for count in atom_counts),
            },
            "model_feature_columns": chosen_features,
        }
        self.structure_data = data["structures"]
        if data["morgan"]["columns"] == MORGAN_SIZE:
            self.finding("PASS", "Morgan fingerprint dimension", "Generated Morgan matrix has 1024 columns using radius 2 and fpSize 1024.", 0)
        else:
            self.finding("CRITICAL", "Morgan fingerprint dimension", f"Expected 1024 bits, got {data['morgan']['columns']}.", len(final), "Check Morgan generator configuration.")
        if len(invalid_smiles) or len(disconnected) or len(unusual_elements) or len(zero_atoms):
            affected = len(set(invalid_smiles + disconnected + zero_atoms)) + len(unusual_elements)
            self.finding("WARNING", "chemical structure sanity", f"Found parse failures={len(invalid_smiles)}, disconnected structures={len(disconnected)}, zero-atom structures={len(zero_atoms)}, unusual-element rows={len(unusual_elements)}.", affected, "Review the exported structure_warnings.csv and source structures; audit did not remove rows.")
        return data

    def _leakage_audit(self, frame, feature_audit):
        selected = feature_audit["model_feature_columns"]
        identifiers = {"InChIkey", "molecule_chembl_id", "assay_chembl_id", "target_chembl_id", "record_id", "parent_molecule_chembl_id", "cleaned_smiles", "scaffold", "Scaffold_InChI"}
        leaked_identifiers = sorted(set(selected).intersection(identifiers))
        target_names = [name for name in selected if any(token in name.lower() for token in ["pic50", "ic50", "standard_value", "activity", "potency"])]
        direct_value_matches = []
        feature_frame = self.model_feature_frame
        for column in selected:
            if column not in feature_frame or column == "PIC50":
                continue
            left = pd.to_numeric(feature_frame[column], errors="coerce")
            right = pd.to_numeric(frame["PIC50"], errors="coerce")
            mask = left.notna() & right.notna()
            if mask.any() and np.array_equal(left[mask].to_numpy(), right[mask].to_numpy()):
                direct_value_matches.append(column)
        leakage = {
            "selected_feature_count": len(selected),
            "selected_columns": selected,
            "identifier_columns_in_selected_features": leaked_identifiers,
            "target_related_feature_names": target_names,
            "features_directly_equal_to_PIC50": direct_value_matches,
            "target_leakage_status": "WARNING" if target_names or direct_value_matches else "PASS",
            "identifier_leakage_status": "CRITICAL" if leaked_identifiers else "PASS",
        }
        if leaked_identifiers:
            self.finding("CRITICAL", "identifier leakage", f"Identifier/scaffold columns are in model features: {leaked_identifiers}.", len(leaked_identifiers), "Exclude identifiers from model matrices while preserving scaffold labels for grouping.")
        else:
            self.finding("PASS", "identifier leakage", "Selected model features exclude molecular, assay, target, record, and scaffold identifiers.", 0)
        if target_names or direct_value_matches:
            self.finding("WARNING", "target leakage", f"Target-related names={target_names}; directly target-equal values={direct_value_matches}.", len(set(target_names + direct_value_matches)), "Inspect these columns and their provenance before fitting.")
        else:
            self.finding("PASS", "target leakage", "Selected numeric feature names and values show no direct pIC50 feature match.", 0)
        return leakage

    def _write_warnings(self):
        write_json(self.run_dir / "warnings.json", self.findings)

    def _config(self, raw, final):
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception:
            sklearn_version = None
        try:
            import scipy
            scipy_version = scipy.__version__
        except Exception:
            scipy_version = None
        dataset_stable_hash = stable_hash(final, ["InChIkey", "cleaned_smiles", "PIC50"])
        try:
            input_hash = hashlib.sha256(self.raw_csv.read_bytes()).hexdigest()
        except Exception:
            input_hash = None
        return {
            "target_chembl_id": self.target_id,
            "activity_endpoint": "IC50",
            "raw_csv": str(self.raw_csv),
            "allowed_units": ALLOWED_UNITS,
            "unit_to_nM_factors_used_by_current_code": UNIT_TO_NM,
            "descriptor_list": list(DESCRIPTOR_FUNCTIONS),
            "morgan_radius": MORGAN_RADIUS,
            "morgan_fingerprint_size": MORGAN_SIZE,
            "duplicate_grouping": "InChIkey",
            "duplicate_aggregation": "median PIC50; rowwise abs deviation; median deviation per InChIKey",
            "group_mad_threshold": MAD_THRESHOLD,
            "exact_duplicate_columns": ["InChIkey", "cleaned_smiles", "PIC50"],
            "relation_filter": RELATION_FILTER,
            "activity_filter_current": "standard_value <= 10000",
            "activity_filter_comparison": "IC50_nM <= 10000",
            "scaffold_split_fraction": SCAFFOLD_TRAIN_FRACTION,
            "scaffold_allocation": "scaffolds sorted by group size descending; whole groups added to train while train count is below target",
            "random_seeds_in_model_code": {"optuna_tpe": 50, "random_forest": 50},
            "python_version": sys.version,
            "rdkit_version": rdBase.rdkitVersion,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn_version,
            "scipy_version": scipy_version,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_value(self.root, ["git", "rev-parse", "HEAD"]),
            "git_branch": git_value(self.root, ["git", "branch", "--show-current"]),
            "script_filename": "CDDgem.py",
            "raw_row_count": len(raw),
            "final_row_count_current_filter": len(final),
            "raw_csv_sha256": input_hash,
            "final_sorted_molecular_data_sha256": dataset_stable_hash,
        }

    def run(self):
        raw, current_final = self._cleaning_stages()
        after_dup = self.snapshots["potential_duplicate removal"]
        unit_stage = self.snapshots["Supported-unit filter and IC50 normalization"]
        converted = self.snapshots["pIC50 conversion"]
        after_mad = self.snapshots["Exact duplicate removal after MAD"]
        relation_cohort = self.snapshots['standard_relation == "=" filter']
        self._unit_audit(raw, after_dup, unit_stage)
        identity_frame = self.snapshots["InChIKey generation"]
        duplicate_stats = self._duplicate_audit(converted, after_mad, current_final, identity_frame)
        current_a, normalized_b, filtering_stats = self._filtering_audit(relation_cohort)

        relation_counts = raw.get("standard_relation", pd.Series(index=raw.index, dtype=object)).value_counts(dropna=False)
        relation_table = pd.DataFrame({
            "standard_relation": ["<NULL>" if pd.isna(value) else str(value) for value in relation_counts.index],
            "records": relation_counts.values,
            "percent_of_raw": 100 * relation_counts.values / max(1, len(raw)),
            "retained_by_current_equals_filter": [value == RELATION_FILTER for value in relation_counts.index],
        })
        relation_table.to_csv(self.run_dir / "relation_distribution.csv", index=False)
        rejected_relation_count = int((raw.get("standard_relation", pd.Series(index=raw.index, dtype=object)) != RELATION_FILTER).sum())
        self.finding("PASS", "standard relation filter recorded", f"Current pipeline retains only relation '='; {rejected_relation_count} raw records have a different relation or null.", rejected_relation_count)

        final_target_stats = target_stats(current_final["PIC50"])
        numeric_standard = pd.to_numeric(raw.get("standard_value", pd.Series(index=raw.index, dtype=float)), errors="coerce")
        cutoff_near = {}
        normalized_values = pd.to_numeric(current_final["IC50"], errors="coerce")
        for percent in [1, 5, 10]:
            cutoff_near[f"within_{percent}_percent_of_10000_nM"] = int((abs(normalized_values - ACTIVITY_THRESHOLD) <= ACTIVITY_THRESHOLD * percent / 100).sum())
        target_checks = {
            **final_target_stats,
            "zero_IC50": int((pd.to_numeric(current_final["IC50"], errors="coerce") == 0).sum()),
            "negative_IC50": int((pd.to_numeric(current_final["IC50"], errors="coerce") < 0).sum()),
            "zero_pIC50": int((pd.to_numeric(current_final["PIC50"], errors="coerce") == 0).sum()),
            "negative_pIC50": int((pd.to_numeric(current_final["PIC50"], errors="coerce") < 0).sum()),
            "extreme_pIC50_below_0_or_above_20": int(((pd.to_numeric(current_final["PIC50"], errors="coerce") < 0) | (pd.to_numeric(current_final["PIC50"], errors="coerce") > 20)).sum()),
            "normalized_IC50_stats_nM": number_stats(current_final["IC50"]),
            "threshold_boundary_counts": cutoff_near,
        }
        target_checks["raw_standard_value_sanity"] = {
            "null_or_non_numeric": int(numeric_standard.isna().sum()),
            "nonfinite": int((numeric_standard.notna() & ~np.isfinite(numeric_standard)).sum()),
            "zero": int((numeric_standard == 0).sum()),
            "negative": int((numeric_standard < 0).sum()),
        }
        invalid_final_target = target_checks["missing"] + target_checks["nonfinite"]
        if invalid_final_target:
            self.finding("CRITICAL", "finite model target", f"Final pIC50 contains {invalid_final_target} missing or nonfinite values.", invalid_final_target, "Inspect target conversion and input values; audit did not remove records.")
        else:
            self.finding("PASS", "finite model target", "Final pIC50 has no missing or nonfinite values.", 0)
        write_json(self.run_dir / "target_statistics.json", target_checks)

        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_SIZE)
        scaffold_stats, _ = self._scaffold_audit(current_a)
        structure_frame = self.snapshots["InChIKey generation"]
        feature_audit = self._feature_audit(current_a, morgan_generator, structure_frame)
        leakage = self._leakage_audit(current_a, feature_audit)
        target_distributions = self._target_distribution(current_a, self.split_assignments)
        final_target_stats = target_distributions["final_current_filter"]

        relation_counts_dict = {("<NULL>" if pd.isna(key) else str(key)): int(value) for key, value in relation_counts.items()}
        unit_counts = raw["standard_units"].value_counts(dropna=False) if "standard_units" in raw else pd.Series(dtype=int)
        unit_summary = {("<NULL>" if pd.isna(key) else str(key)): int(value) for key, value in unit_counts.items()}
        group_mad_over = duplicate_stats.loc[duplicate_stats["MAD_over_1"]]
        affected_measurements = int(group_mad_over["number_of_measurements"].sum())
        mad_summary = {
            "InChIKey_groups": int(len(duplicate_stats)),
            "groups_MAD_over_1": int(len(group_mad_over)),
            "percent_groups_MAD_over_1": 0.0 if len(duplicate_stats) == 0 else 100 * len(group_mad_over) / len(duplicate_stats),
            "measurements_in_groups_MAD_over_1": affected_measurements,
            "percent_measurements_in_groups_MAD_over_1": 0.0 if len(converted) == 0 else 100 * affected_measurements / len(converted),
            "maximum_measurements_per_InChIKey": int(duplicate_stats["number_of_measurements"].max()) if len(duplicate_stats) else 0,
            "MAD_distribution": number_stats(duplicate_stats["MAD"]),
            "within_group_pIC50_range_distribution": number_stats(duplicate_stats["pIC50_range"]),
        }
        identity_frame = structure_frame
        structure_identity = {
            "identity_check_stage": "after salt/SMILES cleaning and InChIKey generation, before downstream filters",
            "unique_cleaned_SMILES": int(identity_frame["cleaned_smiles"].nunique(dropna=True)),
            "unique_InChIKeys": molecule_key_count(identity_frame),
            "InChIKeys_with_multiple_cleaned_SMILES": int((identity_frame.dropna(subset=["InChIkey"]).groupby("InChIkey")["cleaned_smiles"].nunique() > 1).sum()),
            "cleaned_SMILES_with_multiple_InChIKeys": int((identity_frame.dropna(subset=["cleaned_smiles"]).groupby("cleaned_smiles")["InChIkey"].nunique() > 1).sum()),
            "final_unique_InChIKeys_after_current_filters": molecule_key_count(current_final),
            **self.structure_data,
        }
        unit_conversion_check = {}
        for unit in ALLOWED_UNITS:
            expected = 1.0 if unit == "nM" else 10 ** int(unit.split("'")[1].split("pM")[0]) / 1000
            unit_conversion_check[unit] = {
                "factor_to_nM_used_by_code": UNIT_TO_NM[unit],
                "factor_to_nM_from_unit_prefix_math": expected,
                "conversion_matches_prefix_math": bool(np.isclose(UNIT_TO_NM[unit], expected)),
            }
            if not np.isclose(UNIT_TO_NM[unit], expected):
                self.finding("WARNING", "unit conversion formula", f"The current {unit} multiplier is {UNIT_TO_NM[unit]}, while prefix math gives {expected} nM per source unit.", None, "Review the encoded unit and conversion factor; audit did not change it.")
        unit_summary_data = {
            "original_unit_counts_before_normalization": unit_summary,
            "unsupported_or_null_unit_rows_raw": int((~raw["standard_units"].isin(ALLOWED_UNITS)).sum()) if "standard_units" in raw else len(raw),
            "unit_conversion_factors_and_math": unit_conversion_check,
            "representative_conversion_examples_csv": "unit_conversion_examples.csv",
        }
        config = self._config(raw, current_final)
        write_json(self.run_dir / "config_snapshot.json", config)
        write_json(self.run_dir / "warnings.json", self.findings)

        stage_table = pd.DataFrame(self.stages)
        stage_table.to_csv(self.run_dir / "stage_counts.csv", index=False)
        summary = {
            "dataset": {
                "raw_retrieval_is_saved_local_snapshot_not_live_network_request": True,
                "raw_path": str(self.raw_csv),
                "raw_records": len(raw),
                "final_records_current_behavior": len(current_final),
                "final_unique_InChIKeys": molecule_key_count(current_final),
                "final_unique_cleaned_SMILES": smiles_key_count(current_final),
                "raw_sha256": config["raw_csv_sha256"],
                "stable_final_sha256": config["final_sorted_molecular_data_sha256"],
            },
            "config_snapshot": config,
            "units": unit_summary_data,
            "target": target_checks,
            "duplicates": mad_summary,
            "structures": structure_identity,
            "features": feature_audit,
            "leakage": leakage,
            "filtering_comparison": filtering_stats,
            "standard_relation_counts": relation_counts_dict,
            "relation_rows_removed_from_raw": rejected_relation_count,
            "scaffold_split": scaffold_stats,
            "stage_counts": self.stages,
            "findings": self.findings,
            "audit_output_directory": str(self.run_dir),
        }
        write_json(self.run_dir / "audit_summary.json", summary)
        self._write_markdown(summary)
        self._print_summary(summary)
        critical = [entry for entry in self.findings if entry["severity"] == "CRITICAL"]
        if critical:
            raise AssertionError(f"Audit hard invariant failure(s); reports saved to {self.run_dir}")
        return summary

    def _write_markdown(self, summary):
        counts = pd.DataFrame(self.stages)
        columns = ["stage_order", "stage", "before_rows", "rows_after", "rows_removed", "percent_removed", "unique_InChIKeys", "unique_cleaned_SMILES"]
        lines = [
            "# AChE Data Audit", "",
            "## 1. Run Information", "",
            f"- Timestamp: {summary['dataset']['raw_records'] and self.run_dir.name}",
            f"- Target: `{self.target_id}`",
            f"- Input snapshot: `{self.raw_csv}`",
            f"- Input SHA-256: `{summary['dataset']['raw_sha256']}`",
            f"- Stable final molecular data SHA-256: `{summary['dataset']['stable_final_sha256']}`",
            f"- Git branch / commit: `{summary['config_snapshot']['git_branch']}` / `{summary['config_snapshot']['git_commit']}`",
            "- This mode reads the saved raw CSV snapshot; it does not query ChEMBL or modify datasets.", "",
            "## 2. Dataset Flow", "",
            counts[columns].to_markdown(index=False), "",
            "The existing validity filter runs before salt cleaning. Unit support filtering and unit normalization occur in one current method; the audit reports them together. Median aggregation is a transform and does not itself reduce row count. The current code then applies group MAD filtering, exact duplicate removal, relation filtering, and the raw `standard_value` activity cutoff.", "",
            "## 3. Unit Audit", "",
            f"- Original units before normalization: `{json.dumps(summary['units']['original_unit_counts_before_normalization'], ensure_ascii=True)}`",
            f"- Unsupported or null raw unit records: {summary['units']['unsupported_or_null_unit_rows_raw']}",
            "- Conversion examples and the code's conversion factors are in `unit_conversion_examples.csv`.",
            "- Unit distributions are in `unit_distribution.csv`.", "",
            "### Raw-value versus normalized-concentration filter", "",
            f"- Current A (`standard_value <= 10000`) retains {summary['filtering_comparison']['A_retained']} records.",
            f"- B (`IC50 in nM <= 10000`) retains {summary['filtering_comparison']['B_retained']} records.",
            f"- Differently classified: {summary['filtering_comparison']['records_differently_classified']} ({summary['filtering_comparison']['percent_candidate_records_differently_classified']:.4f}% of post-relation candidates); {summary['filtering_comparison']['unique_InChIKeys_affected']} unique InChIKeys affected.",
            "- No threshold behavior was changed. Detailed rows are in `filtering_differences.csv` and `filtering_differences_by_unit.csv`.", "",
            "## 4. Target Audit", "",
            f"- Final pIC50 stats: `{json.dumps(summary['target'], ensure_ascii=True)}`",
            "- Full target and split distributions are in `target_distribution.csv` and `split_target_distribution.csv`.", "",
            "## 5. Duplicate / Replicate Audit", "",
            f"- Replicate group summary: `{json.dumps(summary['duplicates'], ensure_ascii=True)}`",
            "- Group-level statistics and replicate counts are in `duplicate_analysis.csv` and `replicate_count_distribution.csv`.",
            "- The 20 largest within-InChIKey pIC50 ranges are in `high_variability_molecules.csv`.",
            "- Exact duplicate signature counts are in `exact_duplicate_checks.csv`.", "",
            "## 6. Molecular Identity Audit", "",
            f"- Identity summary: `{json.dumps({key: value for key, value in summary['structures'].items() if key.startswith('unique_') or 'multiple_' in key}, ensure_ascii=True)}`",
            "- Suspicious multi-mappings, if present, are in `identity_mappings.csv`.", "",
            "## 7. Structure Sanity", "",
            f"- Structure checks: `{json.dumps(summary['structures'], ensure_ascii=True)}`",
            "- Unusual elements are reported in `structure_warnings.csv`; no structures were removed by the audit.", "",
            "## 8. Feature Sanity", "",
            f"- Morgan: `{json.dumps(summary['features']['morgan'], ensure_ascii=True)}`",
            f"- Descriptor count: {summary['features']['descriptors']['count']}; descriptors: `{', '.join(summary['features']['descriptors']['columns'])}`",
            "- Per-feature ranges and missing/nonfinite counts are in `feature_sanity.csv`.",
            "- Exact model columns are in `model_feature_columns.csv`.", "",
            "## 9. Leakage Checks", "",
            f"- Target leakage: **{summary['leakage']['target_leakage_status']}**; {summary['leakage']['target_related_feature_names'] or 'no target-named or directly target-equal selected features found'}.",
            f"- Identifier leakage: **{summary['leakage']['identifier_leakage_status']}**; {summary['leakage']['identifier_columns_in_selected_features'] or 'none'}.", "",
            "## 10. Scaffold Split Audit", "",
            f"- Allocation: {summary['scaffold_split']['allocation_order']}",
            f"- Scaffolds train/test: {summary['scaffold_split']['scaffolds_train']} / {summary['scaffold_split']['scaffolds_test']}",
            f"- Molecules train/test: {summary['scaffold_split']['molecules_train']} / {summary['scaffold_split']['molecules_test']}",
            f"- Scaffold overlap: {summary['scaffold_split']['shared_scaffold_count']} {summary['scaffold_split']['shared_scaffolds']}",
            f"- Molecule overlap: {summary['scaffold_split']['shared_InChIKey_count']} InChIKeys, {summary['scaffold_split']['shared_cleaned_SMILES_count']} cleaned SMILES.",
            f"- Scaffold size summary: `{json.dumps({key: value for key, value in summary['scaffold_split'].items() if 'fraction' in key or 'size' in key or 'largest' in key}, ensure_ascii=True)}`",
            "- Per-scaffold sizes and per-molecule assignments are in `scaffold_analysis.csv` and `scaffold_assignments.csv`.", "",
            "## 11. Train/Test Distribution", "",
            "Train and test pIC50 statistics and quantiles are in `split_target_distribution.csv`.", "",
            "## 12. Findings", "",
        ]
        for severity in ["PASS", "WARNING", "CRITICAL"]:
            lines.extend([f"### {severity}", ""])
            relevant = [entry for entry in self.findings if entry["severity"] == severity]
            if not relevant:
                lines.append("None.")
            else:
                lines.extend([f"- **{entry['check']}**: {entry['message']}" for entry in relevant])
            lines.append("")
        lines.extend(["## 13. Recommended Next Actions", ""])
        recommendations = [entry["recommendation"] for entry in self.findings if entry.get("recommendation")]
        lines.extend([f"- {recommendation}" for recommendation in recommendations] if recommendations else ["- No follow-up is indicated by the audit checks."])
        (self.run_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _print_summary(self, summary):
        findings = pd.Series([entry["severity"] for entry in self.findings]).value_counts()
        
        print("AChE DATA AUDIT COMPLETE")
        
        print(f"Rows:\n    Raw:        {summary['dataset']['raw_records']}\n    Final:      {summary['dataset']['final_records_current_behavior']}")
        print(f"\nUnique molecules:\n    {summary['dataset']['final_unique_InChIKeys']}")
        target = summary["target"]
        print(f"\npIC50:\n    Min:        {target['min']}\n    Median:     {target['median']}\n    Max:        {target['max']}")
        print(f"\nUnits:\n    nM:         {summary['units']['original_unit_counts_before_normalization'].get('nM', 0)} records")
        print(f"    other:      {sum(value for unit, value in summary['units']['original_unit_counts_before_normalization'].items() if unit != 'nM')} records")
        print(f"\nDuplicate groups:\n    {summary['duplicates']['InChIKey_groups']}")
        print(f"Groups MAD > 1:\n    {summary['duplicates']['groups_MAD_over_1']}")
        print(f"\nScaffolds:\n    Train:      {summary['scaffold_split']['scaffolds_train']}\n    Test:       {summary['scaffold_split']['scaffolds_test']}\n    Overlap:    {summary['scaffold_split']['shared_scaffold_count']}")
        print(f"\nMolecule overlap:\n    InChIKey:   {summary['scaffold_split']['shared_InChIKey_count']}\n    SMILES:     {summary['scaffold_split']['shared_cleaned_SMILES_count']}")
        print(f"\nFeature sanity:\n    Morgan:     {'PASS' if summary['features']['morgan']['configuration_matches_radius_2_size_1024'] else 'CRITICAL'}\n    Descriptors: {summary['features']['descriptors']['count']} audited")
        print(f"\nFindings:\n    PASS:       {int(findings.get('PASS', 0))}\n    WARNING:    {int(findings.get('WARNING', 0))}\n    CRITICAL:   {int(findings.get('CRITICAL', 0))}")
        print(f"\nAudit report:\n    {self.run_dir.relative_to(self.root)}")
        print("=" * 50)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only AChE data and feature audit")
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--target-id", default="CHEMBL220")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    raw_csv = args.raw_csv or root / "database" / "csv" / "Acetylcholinesterase_Homo_sapiens.csv"
    if not raw_csv.exists():
        parser.error(f"Raw ChEMBL CSV snapshot not found: {raw_csv}")
    try:
        DataAudit(raw_csv, args.target_id, root, args.output_dir).run()
        return 0
    except AssertionError:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
