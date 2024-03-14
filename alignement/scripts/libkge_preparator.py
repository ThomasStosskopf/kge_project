from pathlib import Path
from argparse import ArgumentParser
from pandas import read_csv, concat, DataFrame
from typing import List, Iterable, Dict
import yaml
from numpy import random

class LibKgePreprocessor:

    def __init__(self, folder):
        headers = ["h", "r", "t"]
        self.folder = Path(folder)
        self.train = read_csv(self.folder.joinpath("train.txt"), sep="\t", names=headers)
        self.test = read_csv(self.folder.joinpath("test.txt"), sep="\t", names=headers)
        self.val = read_csv(self.folder.joinpath("val.txt"), sep="\t", names=headers)
        self.all_data = self.concat_train_test_val()
        self.entity_ids = self.get_entity_id()
        self.relation_ids = self.get_relation_id()
        self.train_ids = self.map_entity_and_relation_ids(dataframe=self.train, entity_ids=self.entity_ids,
                                                          relation_ids=self.relation_ids)
        self.test_ids = self.map_entity_and_relation_ids(dataframe=self.test, entity_ids=self.entity_ids,
                                                         relation_ids=self.relation_ids)
        self.val_ids = self.map_entity_and_relation_ids(dataframe=self.val, entity_ids=self.entity_ids,
                                                        relation_ids=self.relation_ids)
        self.test_without_unseen = self.create_sample_file(graph=self.test_ids)
        self.valid_without_unseen = self.create_sample_file(graph=self.test_ids)
        self.train_sample = self.create_sample_file(graph=self.train_ids)
        self.metadata = self.create_yaml_file()

    def get_train(self):
        return self.test

    def get_folder(self):
        return self.folder

    def get_all_data(self):
        return self.all_data

    def concat_train_test_val(self) -> DataFrame:
        # Concatenate train, test, and val DataFrames vertically
        concatenated_df = concat([self.train, self.test], axis=0, ignore_index=True)
        return concatenated_df

    def get_unique_value(self, graph: DataFrame, col_name: str) -> List :
        return self.graph[col_name].unique()

    def get_entity_id(self):
        head = self.all_data["h"].unique()
        tail = self.all_data["t"].unique()
        # Concatenate head and tail into a single column
        combined_column = concat([DataFrame(head, columns=["combined"]), DataFrame(tail, columns=["combined"])], axis=0, ignore_index=True)
        entities = DataFrame( combined_column["combined"].unique(), columns=["label"] )
        # Reset index and rename the index column to 'id'
        entities.reset_index(inplace=True)
        entities.rename(columns={'index': 'id'}, inplace=True)
        return entities

    def get_relation_id(self):
        rel = DataFrame(self.all_data["r"].unique(), columns=["relation"])
        rel.reset_index(inplace=True)
        rel.rename(columns={'index':'id'}, inplace=True)
        return rel

    def map_entity_and_relation_ids(self, dataframe: DataFrame, entity_ids: DataFrame, relation_ids: DataFrame) -> DataFrame:
        # Map entity IDs
        dataframe['h_id'] = dataframe['h'].map(entity_ids.set_index('label')['id'])
        dataframe['t_id'] = dataframe['t'].map(entity_ids.set_index('label')['id'])

        # Map relation IDs
        dataframe['r_id'] = dataframe['r'].map(relation_ids.set_index('relation')['id'])
        id_dataframe = dataframe[['h_id', 'r_id', 't_id']]
        dataframe = dataframe.drop(columns=['h_id', 'r_id', 't_id'])
        return id_dataframe

    def create_sample_file(self, graph: DataFrame) -> DataFrame:
        '''
        TODO
        finir cette fonction pour faire un split random 
        '''
        sample_size: int = round(0.2*len(graph))
        sample: Iterable[int] = None

        sample = random.choice(len(graph), sample_size, False)
        df_selection = graph.iloc[sample]
        return df_selection

    def create_yaml_file(self):
        data = {
            'dataset': {
                'files.entity_ids.filename': 'entity_ids.del',
                'files.entity_ids.type': 'map',
                'files.entity_strings.filename': 'entity_strings.del',
                'files.entity_strings.type': 'idmap',
                'files.relation_ids.filename': 'relation_ids.del',
                'files.relation_ids.type': 'map',
                'files.test.filename': 'test.del',
                'files.test.size': self.test_ids.shape[0],
                'files.test.split_type': 'test',
                'files.test.type': 'triples',
                'files.test_without_unseen.filename': 'test_without_unseen.del',
                'files.test_without_unseen.size': self.test_without_unseen.shape[0],
                'files.test_without_unseen.split_type': 'test',
                'files.test_without_unseen.type': 'triples',
                'files.train.filename': 'train.del',
                'files.train.size': self.train_ids.shape[0],
                'files.train.split_type': 'train',
                'files.train.type': 'triples',
                'files.train_sample.filename': 'train_sample.del',
                'files.train_sample.size': self.train_sample.shape[0],
                'files.train_sample.split_type': 'train',
                'files.train_sample.type': 'triples',
                'files.valid.filename': 'valid.del',
                'files.valid.size': self.val_ids.shape[0],
                'files.valid.split_type': 'valid',
                'files.valid.type': 'triples',
                'files.valid_without_unseen.filename': 'valid_without_unseen.del',
                'files.valid_without_unseen.size': self.valid_without_unseen.shape[0],
                'files.valid_without_unseen.split_type': 'valid',
                'files.valid_without_unseen.type': 'triples',
                'name': 'fb15k',
                'num_entities': self.entity_ids.shape[0],
                'num_relations': self.relation_ids.shape[0]
            }
        }
        return data


    def save_files(self):
        self.relation_ids.to_csv(self.folder.joinpath("relation_ids.del"), sep="\t", header=False, index=False)
        self.entity_ids.to_csv(self.folder.joinpath("entity_ids.del"), sep="\t", header=False, index=False)
        self.train_ids.to_csv(self.folder.joinpath("train.del"), sep="\t", header=False, index=False)
        self.val_ids.to_csv(self.folder.joinpath("valid.del"), sep="\t", header=False, index=False)
        self.test_ids.to_csv(self.folder.joinpath("test.del"), sep="\t", header=False, index=False)
        self.test_without_unseen.to_csv(self.folder.joinpath("test_without_unseen.del"), sep="\t", header=False, index=False)
        self.valid_without_unseen.to_csv(self.folder.joinpath("valid_without_unseen.del"), sep="\t", header=False, index=False)
        self.train_sample.to_csv(self.folder.joinpath("train_sample.del"), sep="\t", header=False, index=False)
        with open(self.folder.joinpath('dataset.yaml'), 'w') as yaml_file:
            yaml.dump(self.metadata, yaml_file, default_flow_style=False)

if __name__ == "__main__":

    parser = ArgumentParser(description="Prepare data to be used in libkge")
    parser.add_argument("--folder", type=str, help="path to the folder with train.txt test.txt and val.txt")

    args = parser.parse_args()

    folder_path = args.folder
    preprocessor = LibKgePreprocessor(folder=folder_path)
    print(preprocessor.metadata)
    preprocessor.save_files()

