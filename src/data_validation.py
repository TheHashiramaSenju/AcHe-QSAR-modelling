

from pathlib import Path
from data_engineering import DataEng

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