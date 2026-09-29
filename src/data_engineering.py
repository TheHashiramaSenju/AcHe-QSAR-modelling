from pathlib import Path 
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import multiprocessing as mp
import data_cleaning as DataCleaning

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
            results = pool.map(DataCleaning.calculate_molecular_descriptors, smiles_list)

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