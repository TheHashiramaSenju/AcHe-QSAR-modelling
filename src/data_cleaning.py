from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from pathlib import Path
import numpy as np
import pandas as pd


class DataCleaning:

    remover = SaltRemover()

    def __init__(self, path: Path):
        self.path = Path(path)
        self.mismatch = []
        self.filename = self.path.name if self.path.exists() else "unknown_file"

    def catch_bad_row(self, bad_line):
        self.mismatch.append(bad_line)
        return None

    def loading_csv(self, path: Path) -> pd.DataFrame:
        print(f"\nLoading and parsing CSV file: {path.name}")
        dataset_clean = pd.read_csv(
            path,
            engine="python",
            on_bad_lines=self.catch_bad_row
        )
        if self.mismatch:
            print(f"Skipped {len(self.mismatch)} corrupted/mismatched rows.")
        return dataset_clean

    def get_clean_filename(self):
        filename = self.filename
        print(filename)
        if filename.startswith("."):
            filename = filename[1:]
        return filename.split(".")[0]

    @staticmethod
    def stip_salt(smiles_string):
        if pd.isna(smiles_string):
            return None

        mol = Chem.MolFromSmiles(smiles_string)

        if mol is None:
            return None

        stripped_mol = DataCleaning.remover.StripMol(mol)
        return Chem.MolToSmiles(stripped_mol)

    @staticmethod
    def InChIKeyConversion(smiles_string):
        if pd.isna(smiles_string):
            return None

        mol = Chem.MolFromSmiles(smiles_string)

        if mol is None:
            return None

        if mol.GetNumAtoms() == 0:
            return None

        return Chem.MolToInchiKey(mol)

    def null_and_columnhandler(self, path: Path) -> Path:
        dataset_clean = self.loading_csv(path)

        dataset_clean = dataset_clean.dropna(axis=1, how="all")

        cols_to_drop = [
            "qudt_units",
            "uo_units",
            "toid",
            "document_chembl_id",
            "_journal",
            "_year",
            "assay_descriptions",
            "activity_comment",
            "upper_value",
            "molecule_pref_name",
            "type",
            "units",
            "value",
            "document_journal",
            "document_year",
            "assay_description",
            "activity_id",
            "activity_properties",
            "ligand_efficiency"
        ]

        existing_cols_to_drop = [
            col for col in cols_to_drop
            if col in dataset_clean.columns
        ]

        if existing_cols_to_drop:
            dataset_clean = dataset_clean.drop(
                columns=existing_cols_to_drop
            )
            print(
                f"Dropped unneeded columns: {existing_cols_to_drop}"
            )

        targets_data_validity_comment = [
            "Values appear to be an order of magnitude different from previously reported, so units may be incorrect",
            "Potential transcription error"
        ]

        targets_data_validity_desc = [
            "Values for this activity type are unusually large/small, so may not be accurate",
            "Values appear to be an order of magnitude different from previously reported, so units may be incorrect"
        ]

        validity_comment = dataset_clean.get(
            "data_validity_comment",
            pd.Series(index=dataset_clean.index, dtype=object)
        )

        validity_description = dataset_clean.get(
            "data_validity_description",
            pd.Series(index=dataset_clean.index, dtype=object)
        )

        mask_1 = validity_comment.isin(
            targets_data_validity_comment
        )

        mask_2 = validity_description.isin(
            targets_data_validity_desc
        )

        combined_bad_rows = mask_1 | mask_2

        rows_to_drop = dataset_clean[combined_bad_rows].index

        dataset_clean = dataset_clean.drop(
            rows_to_drop,
            axis=0
        )

        dataset_clean = dataset_clean.drop(
            columns=[
                "data_validity_comment",
                "data_validity_description"
            ],
            errors="ignore"
        )

        clean_filename = self.get_clean_filename()

        output_path = (
            path.parent /
            f"{clean_filename}_cleaned.csv"
        )

        dataset_clean.to_csv(
            output_path,
            index=False
        )

        print(
            f"Cleaned CSV saved successfully: {output_path}"
        )

        return output_path

    def InChIstandardization(self) -> pd.DataFrame:
        cleaned_csv_path = self.null_and_columnhandler(
            self.path
        )

        df = pd.read_csv(cleaned_csv_path)

        if "canonical_smiles" not in df.columns:
            raise KeyError(
                "Input CSV must contain a 'canonical_smiles' column."
            )

        df["cleaned_smiles"] = (
            df["canonical_smiles"]
            .apply(DataCleaning.stip_salt)
        )

        df.drop(
            "canonical_smiles",
            axis=1,
            inplace=True
        )

        df["InChIkey"] = (
            df["cleaned_smiles"]
            .apply(DataCleaning.InChIKeyConversion)
        )

        df.to_csv(
            cleaned_csv_path,
            index=False
        )

        print(
            f"Standardized CSV saved successfully: {cleaned_csv_path}"
        )

        return df

    def basic_duplicate_resolution(self) -> pd.DataFrame:
        dataframe = self.InChIstandardization()

        if "potential_duplicate" in dataframe.columns:
            dataframe = dataframe[
                dataframe["potential_duplicate"] != 1
            ].copy()

        return dataframe

    def IC50_units_standardization(self) -> pd.DataFrame:
        dataframe = self.basic_duplicate_resolution().copy()

        supported_units = [
            "10'5pM",
            "10'6pM",
            "10'3pM",
            "nM"
        ]

        dataframe = dataframe[
            dataframe["standard_units"].isin(
                supported_units
            )
        ].copy()

        dataframe["original_standard_units"] = (
            dataframe["standard_units"]
        )

        unit_factor = {
            "10'5pM": 100.0,
            "10'6pM": 1000.0,
            "10'3pM": 1.0,
            "nM": 1.0
        }

        standard_value = pd.to_numeric(
            dataframe["standard_value"],
            errors="coerce"
        )

        dataframe["IC50_nM"] = (
            standard_value *
            dataframe["standard_units"]
            .map(unit_factor)
        )

        dataframe["IC50"] = dataframe["IC50_nM"]

        dataframe["standard_units_normalized"] = "nM"

        return dataframe

    def IC50_to_PIC50Conv(self) -> pd.DataFrame:
        dataframe = (
            self.IC50_units_standardization()
            .copy()
        )

        standard_value = pd.to_numeric(
            dataframe["standard_value"],
            errors="coerce"
        )

        conditions = [
            (
                dataframe["standard_type"] == "IC50"
            ) & (
                dataframe["IC50_nM"] > 0
            ),
            dataframe["standard_type"] == "Log IC50",
            dataframe["standard_type"] == "pIC50",
            dataframe["standard_type"] == "Log IC50(nM)"
        ]

        choices = [
            -np.log10(
                dataframe["IC50_nM"] * 1e-9
            ),
            -standard_value,
            standard_value,
            9 - standard_value
        ]

        dataframe["PIC50"] = np.select(
            conditions,
            choices,
            default=np.nan
        )

        dataframe["PIC50"] = pd.to_numeric(
            dataframe["PIC50"],
            errors="coerce"
        )

        dataframe = dataframe[
            np.isfinite(dataframe["PIC50"])
        ].copy()

        dataframe["PIC50"] = (
            dataframe["PIC50"]
            .round(5)
        )

        return dataframe

    def relationalvalue(self) -> Path:
        dataframe = (
            self.IC50_to_PIC50Conv()
            .copy()
        )

        safe_relations = (
            dataframe["standard_relation"] == "="
        )

        valid_ic50 = (
            pd.to_numeric(
                dataframe["IC50_nM"],
                errors="coerce"
            ) <= 10000
        )

        valid_positive_ic50 = (
            pd.to_numeric(
                dataframe["IC50_nM"],
                errors="coerce"
            ) > 0
        )

        dataframe = dataframe[
            safe_relations
            & valid_ic50
            & valid_positive_ic50
        ].copy()

        dataframe["median"] = (
            dataframe
            .groupby("InChIkey")["PIC50"]
            .transform("median")
        )

        dataframe["compared_median"] = (
            dataframe["PIC50"]
            - dataframe["median"]
        ).abs()

        dataframe["group_MAD"] = (
            dataframe
            .groupby("InChIkey")["compared_median"]
            .transform("median")
        )

        mad_threshold = 1.0

        dataframe = dataframe[
            dataframe["group_MAD"] <= mad_threshold
        ].copy()

        dataframe["PIC50"] = dataframe["median"]

        final_df = dataframe.drop_duplicates(
            subset=["InChIkey"],
            keep="first"
        ).copy()

        final_df = final_df.drop(
            columns=[
                "median",
                "compared_median",
                "group_MAD"
            ],
            errors="ignore"
        )

        final_df = final_df.drop(
            columns=[
                "pchembl_value",
                "potential_duplicate",
                "relation"
            ],
            errors="ignore"
        )

        clean_filename = self.get_clean_filename()

        output_path = (
            self.path.parent /
            f"{clean_filename}_cleaned.csv"
        )

        final_df.to_csv(
            output_path,
            index=False
        )

        print(
            f"Cleaned CSV saved successfully: {output_path}"
        )

        return output_path


def stip_salt(smiles_string):
    return DataCleaning.stip_salt(smiles_string)


def InChIKeyConversion(smiles_string):
    return DataCleaning.InChIKeyConversion(smiles_string)


def calculate_molecular_descriptors(smiles):
    
    def num_aliphatic_nitrogens(mol):
        return sum(
            atom.GetAtomicNum() == 7
            and not atom.GetIsAromatic()
            for atom in mol.GetAtoms()
        )

    bio_descriptors = {
        
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
        "NumAliphaticNitrogens": num_aliphatic_nitrogens,
        "NumFormalCharge": Chem.GetFormalCharge
    
    }

    empty_features = {
        key: None
        for key in bio_descriptors
    }

    if not isinstance(smiles, str) or not smiles.strip():
        return empty_features

    raw_mol = Chem.MolFromSmiles(smiles)

    if raw_mol is None:
        return empty_features

    mol = SaltRemover().StripMol(raw_mol)

    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

        features = {}

        for name, descriptor in bio_descriptors.items():
            value = descriptor(mol)

            if (
                value != value
                or value == float("inf")
                or value == float("-inf")
            ):
                features[name] = None
            else:
                features[name] = value

        return features

    except Exception:
        return empty_features