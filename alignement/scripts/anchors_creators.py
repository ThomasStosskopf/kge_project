from pathlib import Path
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
import argparse
import os


class AnchorsCreator:

    def __init__(self, path):
        self.path = Path(path)
        column_names = ['id', 'label']
        self.drug_entity_id = read_csv(self.path.joinpath("drug/entity_ids.del"), sep="\t", names=column_names)
        self.disease_entity_id = read_csv(self.path.joinpath("disease/entity_ids.del"), sep="\t", names=column_names)
        self.common_entity = self.find_similar_entity()
        self.train, self.test, self.val = self.split_train_test_val()

    def find_similar_entity(self):
        merged_df = self.drug_entity_id.merge(self.disease_entity_id, how='inner', on=['label'])
        saving_path = self.path.joinpath("align")
        if not saving_path.exists():
            saving_path.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(saving_path.joinpath("align-graph1-graph2.csv"), sep=",", index=False)
        return merged_df

    def take_idx_and_idy_together(self):
        id_x = self.common_entity["id_x"]
        id_y = self.common_entity["id_y"]
        common_ids_df = DataFrame({'id_x': id_x, 'id_y': id_y})
        return common_ids_df

    def split_train_test_val(self, test_size=0.2, val_size=0.1, random_state=None ):
        df_to_split = self.take_idx_and_idy_together()
        train, test = train_test_split(df_to_split, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train, test_size=val_size, random_state=random_state)
        return train, test, val

    def save_train_test_val(self) -> None:
        self.train.to_csv(self.path.joinpath("train.csv"), sep=",", index=False, header=False)
        self.test.to_csv(self.path.joinpath("test.csv"), sep=",", index=False, header=False)
        if self.val is not None:
            self.val.to_csv(self.path.joinpath("val.csv"), sep=",", index=False, header=False)
        self.drug_entity_id["id"].to_csv(self.path.joinpath("drug/vocab.json"), index=False, header=False)
        self.disease_entity_id["id"].to_csv(self.path.joinpath("disease/vocab.json"), index=False, header=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare the split data ")
    parser.add_argument("--input", type=str, help="Path to the trained model folder")
    args = parser.parse_args()

    anchors_creator = AnchorsCreator(path=args.input)
    print(anchors_creator.disease_entity_id)
    anchors_creator.save_train_test_val()
