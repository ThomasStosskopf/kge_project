from pathlib import Path
import argparse
from pandas import read_csv, DataFrame

class FedCoderDataPreparer:

    def __init__(self, path1):
        self.path = Path(path1)
        self.drugs_entity_ids = read_csv(self.path.joinpath("drug/entity_ids.del"), sep="\t")
        self.disease_entity_ids = read_csv(self.path.joinpath("disease/entity_ids.del"), sep="\t")

    def get_drugs_entity_ids(self):
        return self.drugs_entity_ids

    def get_disease_entity_ids(self):
        return self.disease_entity_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare the split data ")
    parser.add_argument("--input", type=str, help="Path to the first kg data folder")
    parser.add_argument("--graph2", type=str, help="Path to the second kg data folder")
    parser.add_argument("--align", type=str, help="Path to output of align folder.")
    args = parser.parse_args()

    preparer = FedCoderDataPreparer(path1=args.input)

    print(preparer.get_disease_entity_ids())
