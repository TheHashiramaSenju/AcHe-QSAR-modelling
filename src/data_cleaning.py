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