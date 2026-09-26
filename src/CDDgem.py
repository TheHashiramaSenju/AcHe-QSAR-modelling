from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Optional
import numpy as np
from chembl_webresource_client.new_client import new_client
import duckdb
import pandas as pd
from rdkit import Chem 
from rdkit.Chem.SaltRemover import SaltRemover
from scipy.stats import median_abs_deviation
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import optuna 
from sklearn.metrics import mean_absolute_error
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import multiprocessing as mp 
from sklearn.model_selection import GroupKFold
import joblib 
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

OUTPUT_DIR = "/media/notshadow/d5dd988b-c393-4302-aa45-32bcfc8463c2/WorkFolder/DrugDiscovery-BioInformatics/output"



def data_retrieval_desc(target_name: str) -> pd.DataFrame:
    if not target_name or not target_name.strip():
        raise ValueError("Target name cannot be empty.")

    try:
        target = new_client.target
        target_query = target.search(str(target_name).strip())
        targets = pd.DataFrame.from_dict(target_query)
    except Exception as e:
        raise RuntimeError("Failed to retrieve target data from ChEMBL.") from e

    if targets.empty:
         raise ValueError(f"No targets found matching: '{target_name}'")

    display_cols = [c for c in ["target_chembl_id", "pref_name", "organism", "target_type"] if c in targets.columns]
    print("\nTop 10 Target Matches:")
    print(targets[display_cols].head(10))

    return targets


def select_target(target_index: int, targets: pd.DataFrame) -> list:
    
    if not isinstance(target_index, int):
        raise TypeError("target_index must be an integer.")

    if targets is None or targets.empty:
        raise ValueError("The targets DataFrame is empty or None.")

    if "target_chembl_id" not in targets.columns:
        raise KeyError("Column 'target_chembl_id' not found in targets DataFrame.")

    if target_index < 0 or target_index >= len(targets):
        raise IndexError(f"target_index {target_index} is out of bounds (0 to {len(targets) - 1}).")

    selected_row = targets.iloc[target_index]
    
    selected_target = str(selected_row["target_chembl_id"])
    pref_name = str(selected_row.get("pref_name", "UnknownTarget"))
    organism = str(selected_row.get("organism", "UnknownOrganism"))

    clean_name = re.sub(r"\W+", "_", pref_name).strip("_")
    clean_org = re.sub(r"\W+", "_", organism).strip("_")

    return [selected_target, [clean_name, clean_org]]


class SQLEngine(ABC):
    
    @abstractmethod
    def create_and_load_csv(self) -> tuple[Path, str]:
        pass

    @abstractmethod
    def create_database(self, csv_filepath: Path, table_name: str) -> None:
        pass

    @abstractmethod
    def query_check(self, table_name: str) -> pd.DataFrame:
        pass


class DuckDBEngine(SQLEngine):
    
    def __init__(self, connection: Optional[duckdb.DuckDBPyConnection], targeted: list, root_folder: Path):
        self.conn = connection if connection else duckdb.connect() 
        self.target_chembl_id = targeted[0]
        self.pref_name, self.organism = targeted[1]
        self.root_folder = Path(root_folder)

        if not self.target_chembl_id:
            raise ValueError("Target CHEMBL ID is missing from configuration.")

    def create_and_load_csv(self) -> tuple[Path, str]:
        csv_folder = self.root_folder / "database" / "csv"
        csv_folder.mkdir(parents=True, exist_ok=True)

        table_name = f"{self.pref_name}_{self.organism}"
        csv_file_path = csv_folder / f"{table_name}.csv"

        print(f"\nFetching complete IC50 activity table for {self.target_chembl_id} ({table_name})...")
        print("Network transfer started. This may take a few minutes for large targets.")

        activity = new_client.activity
        res = activity.filter(
            target_chembl_id=self.target_chembl_id,
            standard_value__isnull=False,
            standard_type__in=["IC50"]
        )

        records = []
        for i, record in enumerate(res, 1):
            records.append(record)
            if i % 1000 == 0:
                print(f"Successfully downloaded {i} records...")

        if not records:
            raise ValueError(f"No IC50 activity records found for {self.target_chembl_id}.")

        print(f"Finished downloading {len(records)} total records. Writing to CSV...")
        df = pd.DataFrame(records)
        df.to_csv(csv_file_path, index=False)
        print(f"Saved: {csv_file_path}")

        return csv_file_path, table_name

    def create_database(self, csv_filepath: Path, table_name: str) -> None:
        
        if not csv_filepath or not csv_filepath.exists():
            raise FileNotFoundError(f"Source CSV file not found: {csv_filepath}")

        db_folder = self.root_folder / "database" / "db"
        parquet_folder = db_folder / "parquet"
        db_folder.mkdir(parents=True, exist_ok=True)
        parquet_folder.mkdir(parents=True, exist_ok=True)

        safe_table = f'"{table_name}"'
        posix_csv = csv_filepath.as_posix()

        print("\nBuilding DuckDB tables and persistent storage")

        self.conn.execute(f"CREATE OR REPLACE TABLE {safe_table} AS SELECT * FROM read_csv_auto('{posix_csv}');")

        target_parquet = parquet_folder / f"{table_name}.parquet"
        self.conn.execute(f"COPY {safe_table} TO '{target_parquet.as_posix()}' (FORMAT PARQUET);")

        target_db = db_folder / f"{table_name}.db"
        self.conn.execute(f"ATTACH '{target_db.as_posix()}' AS disk_db;")
        self.conn.execute(f"CREATE OR REPLACE TABLE disk_db.{safe_table} AS SELECT * FROM {safe_table};")
        self.conn.execute("DETACH disk_db;")

        print(f"Parquet saved: {target_parquet}")
        print(f"DB File saved: {target_db}")

    def query_check(self, table_name: str) -> pd.DataFrame:
        print("\nRunning verification query (LIMIT 5):")
        safe_table = f'"{table_name}"'
        return self.conn.execute(f"SELECT * FROM {safe_table} LIMIT 5;").fetchdf()


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
            engine='python', 
            on_bad_lines=self.catch_bad_row
        ) 
        
        if self.mismatch:
            print(f"Skipped {len(self.mismatch)} corrupted/mismatched rows.")
        return dataset_clean 
    
    def get_clean_filename(self):
        filename = self.filename
        print(filename)
        if filename.startswith('.'):
            filename = filename[1:]
        return filename.split('.')[0] #we handled null pointer exception in this case where we unindented the return statement to return something on 
            
    '''
    For pandas
    If you hand it a list of True/False values, it filters rows.
    If you hand it a list of names or numbers, it tries to fetch columns.
    
    '''
    #helper function for entire dataframe
    @staticmethod
    def stip_salt(smiles_string):
    
        if pd.isna(smiles_string):
            return None 
        
        mol = Chem.MolFromSmiles(smiles_string)
        
        if mol is None:
            return None
        
        #now we will remove those salts 
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
    
        inchikey = Chem.MolToInchiKey(mol)
        
        return inchikey
        
        #NOTE : INCHI and INCHI key -> INCHI is too BIG and hence INCHI key we will be using for database wide comparisons and analysis        
        

        
    def null_and_columnhandler(self, path: Path) -> Path:
        
        dataset_clean = self.loading_csv(path)
        
        dataset_clean = dataset_clean.dropna(axis=1, how="all")
        
        cols_to_drop = ["qudt_units", "uo_units", "toid", "document_chembl_id", "_journal", "_year", 
                        "assay_descriptions", "activity_comment", "upper_value", 
                        "molecule_pref_name", "type", "units", "value","document_journal", "document_year", 
                        "assay_description", "activity_id", "activity_properties", "ligand_efficiency" ] #added "assay_chembl_id", "target_tax_id" finally here
        
        existing_cols_to_drop = [col for col in cols_to_drop if col in dataset_clean.columns]
        
        if existing_cols_to_drop:
            dataset_clean = dataset_clean.drop(columns=existing_cols_to_drop)
            print(f"Dropped unneeded columns: {existing_cols_to_drop}")
        
        targets_data_validity_comment = ["Values appear to be an order of magnitude different from previously reported, so units may be incorrect", 
                                         "Potential transcription error"]
        
        targets_data_validity_desc = ["Values for this activity type are unusually large/small, so may not be accurate", 
                                      "Values appear to be an order of magnitude different from previously reported, so units may be incorrect"]
        
        # These fields are not guaranteed to be present in every ChEMBL export.
        #REFACTORED - the .get() and fallback options 
        validity_comment = dataset_clean.get("data_validity_comment", pd.Series(index=dataset_clean.index, dtype=object)) #REFACTORED - .index for the series to match the dataset_clean index, ensuring proper alignment for boolean masking.
        validity_description = dataset_clean.get("data_validity_description", pd.Series(index=dataset_clean.index, dtype=object)) #just an fallback we are not going to join this anywhere. so that the rows to drop wont be sitting there with an error
        mask_1 = validity_comment.isin(targets_data_validity_comment)
        mask_2 = validity_description.isin(targets_data_validity_desc)
        
        combined_bad_rows = mask_1 | mask_2        
    
        rows_to_drop = dataset_clean[combined_bad_rows].index
        dataset_clean = dataset_clean.drop(rows_to_drop, axis = 0)
        
        dataset_clean = dataset_clean.drop(
            columns=["data_validity_comment", "data_validity_description"],
            errors="ignore",
        )
        
        clean_filename = self.get_clean_filename()
        output_path = path.parent / f"{clean_filename}_cleaned.csv"
        
        dataset_clean.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved successfully: {output_path}")
        
        return output_path

    
    def InChIstandardization(self) -> pd.DataFrame:
        """Remove salts and add stable molecular identifiers to the cleaned data."""
        
        cleaned_csv_path: Path = self.null_and_columnhandler(self.path)
        df = pd.read_csv(cleaned_csv_path)
        
        #remember to add validation and pre-check before anything 
        if "canonical_smiles" not in df.columns:
            raise KeyError("Input CSV must contain a 'canonical_smiles' column.")
        df["cleaned_smiles"] = df["canonical_smiles"].apply(DataCleaning.stip_salt)
        
        df.drop("canonical_smiles", axis=1, inplace=True)
        
        #now creating a InChI key 
        df["InChIkey"] = df["cleaned_smiles"].apply(DataCleaning.InChIKeyConversion)
        
        df.to_csv(cleaned_csv_path, index=False)
        print(f"Standardized CSV saved successfully: {cleaned_csv_path}")
        
        return df  
        
    
    def basic_duplicate_resolution(self) ->pd.DataFrame:
        dataframe = self.InChIstandardization()
        # Remove records already flagged as duplicates when that metadata exists.
        if "potential_duplicate" in dataframe.columns:
            dataframe = dataframe[dataframe["potential_duplicate"] != 1].copy()
        
        return dataframe

        
    '''
    Since, standard_value is 0% missing data. So we dont need
    special imputation techniques for values
    
    We will add in the columns to drop program in the null_column_handler part 
    '''
    
    def IC50_units_standardization(self) -> pd.DataFrame :
        
        dataframe = self.basic_duplicate_resolution()
        units = ["10'5pM", "10'6pM", "10'3pM", "nM"]
        
        dataframe = dataframe[dataframe["standard_units"].isin(units)].copy()
        
        # pIC50 must be calculated from the normalized nM concentration.
        conditions = [

            (dataframe["standard_units"] == "10'5pM"), 
            (dataframe["standard_units"] == "10'6pM"),
            (dataframe["standard_units"] == "10'3pM"),
            (dataframe["standard_units"] == "nM")
        ]
        
        choices = [
            #pico-molar is thousand times smaller than nano-molar (10 ** 3) and hence conversion factor will be 10 ** -3
            (dataframe["standard_value"] * 100),
            (dataframe["standard_value"] * 1000),
            (dataframe["standard_value"] * 1), 
            (dataframe["standard_value"] * 1) 
            
        ]
        
        dataframe["IC50"] = np.select(conditions, choices)
        dataframe["standard_units"] = "nM"
               
        return dataframe
          
    def IC50_to_PIC50Conv(self) -> pd.DataFrame: 
        
        dataframe = self.IC50_units_standardization()
        
        conditions = [
            (dataframe["standard_type"] == "IC50") & (dataframe["IC50"] > 0),
            (dataframe["standard_type"] == "Log IC50"),
            (dataframe["standard_type"] == "pIC50"), 
            (dataframe["standard_type"] == "Log IC50(nM)")
        ]
        choices = [
            -1 * np.log10(dataframe["IC50"] * 10 ** -9),
            -1 * dataframe["standard_value"],
            dataframe["standard_value"],
            9 - dataframe["standard_value"]
        ]
        
        dataframe["PIC50"] = np.select(conditions, choices) #dropping instinct
        dataframe["PIC50"] = dataframe["PIC50"].round(5)

        
        return dataframe        
        
    def InChI_based_duplicate_res(self) -> pd.DataFrame:
        
        #now there are non-distinct InChI values, they might be from multiple tests so here we address those specific set of values 
        #mean-absolute-deviation
        dataframe = self.IC50_to_PIC50Conv()
        df = dataframe.copy()
        df["median"] = df.groupby("InChIkey")["PIC50"].transform("median")
        df["compared_median"] = abs(df["median"] - df["PIC50"])
        
        df["group_MAD"] = df.groupby("InChIkey")["compared_median"].transform("median") #masking function taking place 
        
        mad_threshold: float = 1.0 #reserch more about this values.
        cleaned_df = df[df["group_MAD"] <= mad_threshold].copy()
        cleaned_df["PIC50"] = cleaned_df["median"]
        
        final_df = cleaned_df.drop_duplicates(subset=["InChIkey", "cleaned_smiles", "PIC50"]).copy()
        final_df = final_df.drop(columns = ["median", "compared_median", "group_MAD"])
        
        return final_df

    def relationalvalue(self) -> Path:
        
        #dropping in-efficient medications > greater than the highest number
        #thermodynamic hard-limit of 10,000
        
        dataframe = self.InChI_based_duplicate_res()
        safe_relations = dataframe["standard_relation"] == "="
        valid_negatives = dataframe["standard_value"] <= 10000
        
        dataframe = dataframe[safe_relations & valid_negatives].copy()
        
        clean_filename = self.get_clean_filename()
        output_path = self.path.parent / f"{clean_filename}_cleaned.csv"
        
        dataframe = dataframe.drop(
            columns=["pchembl_value", "potential_duplicate", "relation"],
            errors="ignore",
        )
        dataframe.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved successfully: {output_path}")
        
        return output_path
    

def calculate_molecular_descriptors(smiles):
    def num_aliphatic_nitrogens(mol):
        return sum(
            atom.GetAtomicNum() == 7 and not atom.GetIsAromatic()
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
        "NumFormalCharge": Chem.GetFormalCharge,
    }

    empty_features = {key: None for key in bio_descriptors}
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
            if value != value or value == float("inf") or value == float("-inf"):
                features[name] = None
            else:
                features[name] = value
        return features
    except Exception:
        return empty_features


class DataEng:

    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024,
    )

    def __init__(self, path:Path):
        
        self.path = Path(path) if isinstance(path, str) else path
        if not self.path.exists():
            raise FileNotFoundError(f"Feature CSV file not found: {self.path}")

        # Load once and retain the original row index for every later operation.
        self.df = pd.read_csv(self.path)
        self.filename = self.path.name if self.path.exists() else "unknown_file"

        
        if "cleaned_smiles" not in self.df.columns:
            raise KeyError("Feature CSV must contain a 'cleaned_smiles' column.")
        
        self.csv_file_path = self.path.parent
        
        
        
    
    @staticmethod
    def morgan_fingerprinting_s_method(smiles_string: str, radius: int = 2, nBits: int = 1024) -> np.ndarray:       
        if pd.isna(smiles_string):
            return np.zeros(nBits, dtype=np.uint8)

        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return np.zeros(nBits, dtype=np.uint8)
        
        fingerprint = DataEng.morgan_generator.GetFingerprint(mol)
        return np.asarray(fingerprint, dtype=np.uint8)
    
    @staticmethod
    def mscl(smiles):
        
        if pd.isna(smiles):
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() == 0:
            return None

        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        return scaffold_smiles
    
    
    def finaldrop(self) -> pd.DataFrame:
        
        final = Path(self.filename).stem
        pipeline_path = self.path.parent/final
        
        columns_to_drop = ["action_type", "target_pref_name", "target_tax_id", "bao_endpoint", 
                           "assay_chembl_id", "target_chembl_id", "record_id", "molecule_chembl_id", 
                           "parent_molecule_chembl_id", "bao_label", "standard_type", "standard_units", "standard_flag"]
        
        self.df = self.df.drop(columns=columns_to_drop, errors="ignore")
        
        self.df.to_csv(f"{pipeline_path}_pipeline", index=False)
        
        return self.df
    
        
    def morgan_fingerprinting(self):
        """Create indexed Morgan fingerprints without discarding row identity."""
        
        self.df = self.finaldrop()
        self.df['morgan_fp'] = self.df["cleaned_smiles"].apply(
            lambda x: self.morgan_fingerprinting_s_method(x, radius=2, nBits=1024)
        )
        return pd.DataFrame(
            np.vstack(self.df["morgan_fp"].to_numpy()),
            index=self.df.index,
        )
        
    def scaffold(self) -> pd.DataFrame: 
        """Split whole scaffold groups while preserving source row indexes."""
        
        y_vector = self.df["PIC50"]
        if y_vector is None:
            raise ValueError("y_vector is required to create a supervised split.")

        X_matrix = self.morgan_fingerprinting()
        y_vector = np.asarray(y_vector)
        if len(y_vector) != len(self.df):
            raise ValueError("y_vector must have one value for every CSV row.")

        df = self.df[["cleaned_smiles"]].copy()
        df["scaffold"] = df["cleaned_smiles"].apply(DataEng.mscl)
        
        #now giving meaning using InChIkey
        
        df["Scaffold_InChI"] = df["scaffold"].apply(DataCleaning.InChIKeyConversion)
        
        df["Scaffold_InChI"] = df["Scaffold_InChI"].fillna("__NO_SCAFFOLD__")
        df["Scaffold_InChI"] = df["Scaffold_InChI"].astype("object").where(
            df["Scaffold_InChI"].notna(),
            "__NO_SCAFFOLD__",
        )
        groups = df.groupby("Scaffold_InChI").groups
        
        #so here the entire index is preserved for ordering and re-ordering. 
        sorted_keys = sorted(groups.keys(), key = lambda k : len(groups[k]), reverse = True) #ascending or descenfing comes fromthe lamda here actually and we reverse fo rbg uckets to be on the top
        
        target_train_size = int(0.8 * len(df))
        train_indices = []
        test_indices = []
        
        for scaffold in sorted_keys:
            
            row_numbers = groups[scaffold]
            
            if len(train_indices) < target_train_size:
                train_indices.extend(row_numbers)
            
            else:
                test_indices.extend(row_numbers)
                
        X_train = X_matrix.loc[train_indices]
        y_train = y_vector[train_indices]
        X_test = X_matrix.loc[test_indices]
        y_test = y_vector[test_indices]
        
        return X_train, y_train, X_test, y_test
              
    
    def variance_thresholding(self):
        
        # Fit on training data only; the test set receives the learned mask.
        X_train, y_train, X_test, y_test = self.scaffold()
        
        thresholder = VarianceThreshold(threshold=0.0475) #reasoning in PDF in a more cleaner format
        
        X_train_new = thresholder.fit_transform(X_train)
        X_test_new = thresholder.transform(X_test)
        
        updated_masks = thresholder.get_support()
        
        return (
            pd.DataFrame(X_train_new, index=X_train.index),
            pd.DataFrame(X_test_new, index=X_test.index),
            y_train,
            y_test,
            updated_masks
        )
        
        
    def correlation_handling(self, threshold = 0.90) -> pd.DataFrame:
        
        """Remove correlated columns using correlations learned from training data."""
        X_train, X_test, y_train, y_test, updated_masks = self.variance_thresholding()
        #drops one of them to prevent *Impoortance Dilution* in Random-Forests
        
        df_train = pd.DataFrame(X_train)
        df_test = pd.DataFrame(X_test)
        
        #first check for mirror columns to eliminate them
        
        corr_matrix = df_train.corr().abs()
        
        #corr-matrix -- mirror (Double deletion prevention - eliminate the lower matrix)
        
        
        raw_upper_cols = np.triu(corr_matrix, k=1)
        upper_triangle = pd.DataFrame(raw_upper_cols, columns=corr_matrix.columns)
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        X_train_clean = df_train.drop(columns=to_drop)
        X_test_clean = df_test.drop(columns=to_drop)
        
        return X_train_clean, X_test_clean, y_train, y_test
    
    def modelledReduction(self)-> pd.DataFrame:
        
        """Select features using a forest fitted only on training targets."""
        X_train, X_test, y_train, y_test = self.correlation_handling()
        rf = RandomForestRegressor(n_estimators=100, random_state = 50, n_jobs=-1 )
        filterer = SelectFromModel(rf, threshold="mean")
        
        filterer.fit(X_train, y_train)
        
        X_train_raw = filterer.transform(X_train)
        X_test_raw = filterer.transform(X_test)
        
        boolean_array = filterer.get_support()
        surviving_names = X_train.columns[boolean_array]
        
        X_train_final = pd.DataFrame(X_train_raw, columns = surviving_names, index = X_train.index) #here use stencil analogy to row level data manipulation 
        X_test_final = pd.DataFrame(X_test_raw, columns = surviving_names, index = X_test.index)
        
        return X_train_final, X_test_final, y_test,  y_train
    
    def column_addition(self) -> pd.DataFrame:
        
        """Reattach selected features to the original rows by index."""
        # we have to add back in such a way that the scaffold data does not inherently affect the dimensions here, but here it does not matter 
        
        '''
        My initial thoughts - Even in row concatneation even with scaffolds this would not actually matter because 
        we are some how gonna exterminate the rows that did not survive entirely essentially deleting a dimension. So we can use masking to 
        make the boolean match and work (stencil) for matching dimensions, else use ~ signs or .iloc with truth labels whcih essentially by itself is masking
        
        New learning - Numpy arrays erases pandas' memory layout and hence to perform vector operations we have to acutally keep datadeames as dataframes itself. 
        .values and all destroy the memory -- any (.) operator will actually kill it. So, we must do preservation of indexes 
        '''
        
        d1, d2, _, __ = self.modelledReduction()
        
        joinee = pd.concat([d1, d2], axis = 0)
        merger = self.df.loc[joinee.index]
        final_df = pd.concat([merger, joinee], axis = 1)
        final_df["scaffold"] = final_df["cleaned_smiles"].apply(DataEng.mscl)
        final_df["Scaffold_InChI"] = final_df["scaffold"].apply(
            DataCleaning.InChIKeyConversion
        )
        cleanfile = Path(self.filename).stem #overrriding concepts can be seen here 
        
        morgan_filename = f"{cleanfile}_morganbased"
        morgan_path = self.path.parent/morgan_filename
        
        final_df.to_csv(f"{morgan_path}.csv", index=False )
        
        # here we dont want loc based index shuffle since we did not do random shuffling we just did scaffolding
        return final_df

    
    @staticmethod
    def _split_frame(frame: pd.DataFrame):
        frame = frame.copy()
        frame["Scaffold_InChI"] = frame["Scaffold_InChI"].astype("object").where(
            frame["Scaffold_InChI"].notna(),
            "__NO_SCAFFOLD__",
        )
        groups = frame.groupby("Scaffold_InChI").groups
        sorted_keys = sorted(
            groups.keys(),
            key=lambda key: len(groups[key]),
            reverse=True,
        )

        target_train_size = int(0.8 * len(frame))
        train_indices = []
        test_indices = []

        for scaffold in sorted_keys:
            row_indices = groups[scaffold]
            if len(train_indices) < target_train_size:
                train_indices.extend(row_indices)
            else:
                test_indices.extend(row_indices)

        X_matrix = frame.drop(columns=["PIC50"])
        y_vector = frame["PIC50"]
        return (
            X_matrix.loc[train_indices],
            y_vector.loc[train_indices],
            X_matrix.loc[test_indices],
            y_vector.loc[test_indices],
        )

    def scaffold_based_split(self) -> pd.DataFrame:
        morgan_df, _ = self.molecular_desc()
        return self._split_frame(morgan_df)
    
    def scaffoldX_CrossValidation(self):
        X_train, _, _, _ = self.scaffold_based_split()
        return X_train["Scaffold_InChI"]

    
    def molecular_desc(self):
        df1 = self.finaldrop().copy()

        smiles_list = df1["cleaned_smiles"].tolist()

        with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
            results = pool.map(calculate_molecular_descriptors, smiles_list)

        descriptor_df = pd.DataFrame(results, index=df1.index)
        fingerprint_df = pd.DataFrame(
            np.vstack(df1["cleaned_smiles"].apply(self.morgan_fingerprinting_s_method)),
            index=df1.index,
            columns=[f"morgan_{number}" for number in range(1024)],
        )
        scaffold_df = pd.DataFrame(index=df1.index)
        scaffold_df["scaffold"] = df1["cleaned_smiles"].apply(self.mscl)
        scaffold_df["Scaffold_InChI"] = scaffold_df["scaffold"].apply(
            DataCleaning.InChIKeyConversion
        )

        morgan_df = pd.concat([df1, fingerprint_df, scaffold_df], axis=1)
        combined_df = pd.concat([df1, fingerprint_df, descriptor_df, scaffold_df], axis=1)

        return morgan_df, combined_df
    
    def scaffold_RDKit(self):
        _, combined_df = self.molecular_desc()
        descriptor_columns = [
            "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
            "MaxPartialCharge", "MinPartialCharge", "NumHeteroatoms",
            "NumRotatableBonds", "FractionCSP3", "NumAromaticRings",
            "RingCount", "NumAliphaticNitrogens", "NumFormalCharge",
        ]
        descriptor_df = combined_df[
            ["PIC50", "Scaffold_InChI"] + descriptor_columns
        ]
        return self._split_frame(descriptor_df)
    
    def scaffold_combined(self):
        _, combined_df = self.molecular_desc()
        return self._split_frame(combined_df)
        
        
class Model:

    def __init__(self, path: Path):
        
        self.de = DataEng(path=path)
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
                'n_jobs' : 1, 
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
                "n_jobs": 1,
                "random_state": 50,
            }
            model = xgb.XGBRegressor(**params)
        
        if model_type == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1500),
                "max_depth": trial.suggest_int("lgb_max_depth", 3, 12),
                "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 255),  
                "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("lgb_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgb_lambda", 1e-8, 10.0, log=True),
                "n_jobs": 1,
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

    def _optimize_and_evaluate(self, model_type, variant, study_name, file_name, n_trials=100):
        X_train, y_train, X_test, y_test = self._prepare_data(variant)
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
        sns.scatterplot(x=y_test, y=y_pred, ax=axes[0])
        axes[0].set_title(f"{model_name}: predictions")
        axes[0].set_xlabel("Actual pIC50")
        axes[0].set_ylabel("Predicted pIC50")
        sns.histplot(residuals, kde=True, ax=axes[1])
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

    def output_locations(self):
        return {
            "models": self.model_path,
            "figures": self.figure_path,
            "logs": self.log_path,
        }


class Validation:

    def __init__(self, path: Path):
        self.data = DataEng(path)

    def random_split(self):
        from sklearn.model_selection import train_test_split
        frame, _ = self.data.molecular_desc()
        X = frame.drop(columns=["PIC50"])
        y = frame["PIC50"]
        return train_test_split(X, y, test_size=0.2, random_state=50)

    def group_split(self):
        return self.data.scaffold_based_split()

    def external_validation(self):
        raise NotImplementedError("Provide an external validation dataset.")


class AblationStudies:

    def __init__(self, path: Path):
        self.model = Model(path)

    def full_model(self, n_trials=20):
        return self.model._optimize_and_evaluate(
            "RandomForest", "combined", "Ablation_Full_Model", "ablation-full-model.joblib", n_trials
        )

    def fingerprint_only(self, n_trials=20):
        return self.model._optimize_and_evaluate(
            "RandomForest", "morgan", "Ablation_Fingerprint_Only", "ablation-fingerprint-only.joblib", n_trials
        )

    def descriptors_only(self, n_trials=20):
        return self.model._optimize_and_evaluate(
            "RandomForest", "descriptors", "Ablation_Descriptors_Only", "ablation-descriptors-only.joblib", n_trials
        )

    def assay_context(self):
        raise NotImplementedError("Assay context features are not defined in the current dataset.")

    def bao_context(self):
        raise NotImplementedError("BAO context features are not defined in the current dataset.")


def run_pipeline(root_folder, target_input, target_index, n_trials=20, compare=True):
    target_results = data_retrieval_desc(target_name=target_input)
    selected_meta = select_target(target_index, target_results)
    engine = DuckDBEngine(
        connection=duckdb.connect(),
        targeted=selected_meta,
        root_folder=root_folder,
    )
    csv_path, table_name = engine.create_and_load_csv()
    engine.create_database(csv_filepath=csv_path, table_name=table_name)
    print(engine.query_check(table_name=table_name))

    clean_path = DataCleaning(path=csv_path).relationalvalue()
    data_engine = DataEng(path=clean_path)
    model_runner = Model(path=clean_path)

    if compare:
        comparison = model_runner.compare_models(n_trials=n_trials)
    else:
        model_runner.RFmodelc()
        comparison = None

    print(f"Generated database CSV: {csv_path}")
    print(f"Generated cleaned CSV: {clean_path}")
    for name, path in model_runner.output_locations().items():
        print(f"{name}: {path}")
    return {
        "engine": engine,
        "data_engine": data_engine,
        "model_runner": model_runner,
        "comparison": comparison,
        "clean_path": clean_path,
    }


def main():
    root_folder = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
    target_input = input("Enter the target name (e.g., acetylcholinesterase): ").strip()
    target_results = data_retrieval_desc(target_name=target_input)
    print(target_results.head(10))
    target_index = int(input("\nEnter the index of the target you want to select: "))
    trial_text = input("Optuna trials per model [20]: ").strip()
    n_trials = int(trial_text) if trial_text else 20
    compare_text = input("Run all model comparisons? [Y/n]: ").strip().lower()
    compare = compare_text not in {"n", "no"}
    return run_pipeline(
        root_folder=root_folder,
        target_input=target_input,
        target_index=target_index,
        n_trials=n_trials,
        compare=compare,
    )


if __name__ == "__main__":
    main()

    