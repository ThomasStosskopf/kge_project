import torch
from pathlib import Path
import argparse

class CongifCreator:

    def __init__(self, path):
        self.path = Path(path)
        self.disease_model = torch.load(self.path.joinpath("disease/model.pt"))
        self.drug_model = torch.load(self.path.joinpath("drug/model.pt"))

    def get_disease_model(self):
        return self.disease_model.embedding_dim



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare the config file")
    parser.add_argument("--input", type=str, help="Path to models folder")
    args = parser.parse_args()

    config = CongifCreator(args.input)

    print(config.get_disease_model())
