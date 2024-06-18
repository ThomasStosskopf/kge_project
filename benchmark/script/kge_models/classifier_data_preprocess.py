from pathlib import Path
import argparse
from pandas import DataFrame, read_csv, concat, merge
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class ClassifierDataPreProcess:

    def __init__(self, path, path_to_method4_preprocess, m4=None):
        self.input_path = Path(path)
        self.path_to_method4_preprocess = Path(path_to_method4_preprocess)
        self.type_to_entities = self.prepare_type_entity_df()  # type to entity df
        self.kg_triples = read_csv(self.path_to_method4_preprocess.joinpath("KG_edge_list.txt"), sep="\t")
        self.entities_to_id = read_csv(self.input_path.joinpath("training_triples/entity_to_id.tsv.gz"), sep="\t")
        self.training_relation_to_id = read_csv(self.input_path.joinpath("training_triples/relation_to_id.tsv.gz"),
                                                sep="\t")
        self.training_numeric_triples = read_csv(self.input_path.joinpath("training_triples/numeric_triples.tsv.gz"),
                                                 sep="\t")
        self.testing_numeric_triples = read_csv(self.input_path.joinpath("testing_triples/numeric_triples.tsv.gz"),
                                                sep="\t")
        self.entities_to_id_m4_with_m3_id = None
        self.entities_to_id_m4 = None
        self.m4 = None
        if m4 is not None:
            self.m4 = Path(m4)
            self.model_m4 = torch.load(self.m4.joinpath("trained_model.pkl"), map_location=torch.device('cpu'))
            self.entities_to_id_m4 = read_csv(self.m4.joinpath("training_triples/entity_to_id.tsv.gz"), sep="\t")

        self.model = torch.load(self.input_path.joinpath("trained_model.pkl"), map_location=torch.device('cpu'))
        self.entity_embedding = self.load_embedding_as_numpy_array(self.model)  # vector representation of entities
        self.id_to_embedding = self.map_id_to_embedding()  # dict id: embedding
        self.df_entity_type_id = self.merge_type_to_entity_and_entity_to_id()
        self.df_train, self.df_test = self.create_data_to_classify()

    def create_the_good_embedding_id_file(self):
        """
        transform self.entities_to_id_m4 en dictionnaire pour mapper les id de self.
        """
        
        dict_id_to_emb_m4 = self.create_dict_from_df(dataframe=self.entities_to_id_m4, col1='id', col2='label')
        # create a dataframe with the M4's embedding
        entity_embedding_m4 = self.load_embedding_as_numpy_array(self.model_m4)
        df_emb = DataFrame(columns=['id','embedding'])
        df_emb['embedding'] = [row for row in entity_embedding_m4]
        df_emb['id'] = df_emb.index
        # replace the id in the df_emb by the corresponding label of M4
        df_emb['id'] = df_emb['id'].map(dict_id_to_emb_m4)
        # replace the label by the id from m3
        dict_entity_id_m3 = self.create_dict_from_df(dataframe=self.entities_to_id, col1='label', col2='id')
        df_emb['id'] = df_emb['id'].map(dict_entity_id_m3)
        dict_id_emb = self.map_id_to_embedding()
        return dict_id_emb

    def load_embedding_as_numpy_array(self, model):
        return model.entity_representations[0](indices=None).detach().numpy()

    def get_embedd(self):
        return self.entity_embedding

    def find_id_from_label(self, dataframe: DataFrame, label: str = "drug;indication;disease") -> int:
        """
        Find the ID associated with a specific label in the DataFrame.

        Args:
            dataframe (DataFrame): The DataFrame containing 'id' and 'label' columns.
            label (str): The label to search for. Default is "drug;indication;disease".

        Returns:
            int: The ID associated with the specified label, or -1 if the label is not found.
        """
        # Filter rows based on the given label
        filtered_rows = dataframe[dataframe['label'] == label]
        # Check if any rows were found
        if not filtered_rows.empty:
            # Retrieve the first ID (assuming there is only one ID for each label)
            return filtered_rows.iloc[0]['id']
        else:
            # Return -1 if the label is not found
            return -1

    def prepare_type_entity_df(self) -> DataFrame:
        entity_to_type = read_csv(self.path_to_method4_preprocess.joinpath("type_to_entities.csv"), sep="\t")
        x_df = entity_to_type[["x_name", "x_type"]].copy()
        y_df = entity_to_type[["y_name", "y_type"]].copy()
        x_df.rename(columns={"x_name": "label", "x_type": "type"}, inplace=True)
        y_df.rename(columns={"y_name": "label", "y_type": "type"}, inplace=True)
        df_entities_to_type = concat([x_df, y_df], axis=0)
        return df_entities_to_type

    def merge_type_to_entity_and_entity_to_id(self) -> DataFrame:
        return merge(self.type_to_entities, self.entities_to_id, on='label').drop_duplicates()

    def filter_kg_triples(self, dataframe: DataFrame, rel_to_filter: str = "drug;indication;disease") -> DataFrame:
        # Create a mask to filter rows where the 'rel' column does not match the specified relationship
        mask = dataframe['rel'] == rel_to_filter
        # Apply the mask to filter the DataFrame
        filtered_df = dataframe[mask]
        return filtered_df

    def make_tuples_with_df(self, dataframe: DataFrame) -> set:
        list_pairs = []
        for index, row in dataframe.iterrows():
            list_pairs.append((row['from'], row['to']))
        set_pairs = set(list_pairs)
        return set_pairs

    def generate_negative_sample(self, dataframe: DataFrame, df_train: DataFrame = None, df_test: DataFrame = None) \
            -> DataFrame:
        set_positive_pairs = self.make_tuples_with_df(dataframe=dataframe)
        set_pos_train = []
        set_pos_test = []

        if df_train is not None:
            set_pos_train = self.make_tuples_with_df(dataframe=df_train)

        if df_test is not None:
            set_pos_test = self.make_tuples_with_df(dataframe=df_test)

        list_neg_sample_pairs = []
        list_col_1 = dataframe['from'].tolist()
        list_col_2 = dataframe['to'].to_list()
        while len(list_neg_sample_pairs) < len(set_positive_pairs):
            drug = random.choice(list_col_1)
            neg_sample = (drug, random.choice(list_col_2))
            if neg_sample not in set_positive_pairs and neg_sample not in list_neg_sample_pairs:
                list_neg_sample_pairs.append(neg_sample)

        return DataFrame(list_neg_sample_pairs, columns=['head', 'tail'])

    def add_mark_col(self, dataframe: DataFrame, mark: int) -> DataFrame:
        dataframe['mark'] = mark
        return dataframe

    def create_dict_from_df(self, dataframe: DataFrame, col1: str = None, col2: str = None):

        if col1 != None and col2 != None:
            name_to_id = dict(zip(dataframe[col1], dataframe[col2]))
            return name_to_id
        else:
            return "type existing col name."

    def filter_df_with_selected_pairs(self, dataframe: DataFrame, label1: str = "drug", labe2: str = "disease"):
        # TODO
        dataframe_to_filter = dataframe.copy()
        dict_type_to_entity = self.create_dict_from_df(dataframe=self.type_to_entities, col1="label", col2="type")
        return self.type_to_entities

    def filter_with_specific_id(self, dataframe: DataFrame, id_to_filter: int):
        #list_pairs = [(row["head"], row["tail"]) for _, row in dataframe.iterrows() if row["relation"] == id_to_filter]
        filtered_df = dataframe[dataframe['relation'] == id_to_filter]
        return filtered_df

    def create_data_to_classify(self):
        df_dd_triples = self.filter_kg_triples(dataframe=self.kg_triples)
        df_dd_pairs = self.remove_chosen_column(dataframe=df_dd_triples, col_name="rel")

        dict_id_entity = self.create_dict_from_df(dataframe=self.entities_to_id, col1='id', col2='label')
        df_train = self.training_numeric_triples.copy()
        df_test = self.testing_numeric_triples.copy()
        # Replace ids by the real label in 'head' and 'tail' column
        df_train_id_replaced = self.replace_data_in_df(dataframe=df_train, col1='head', col2='tail',
                                                       dict_id=dict_id_entity)
        df_test_id_replaced = self.replace_data_in_df(dataframe=df_test, col1='head', col2='tail',
                                                      dict_id=dict_id_entity)
        # filter the two dataset train and test
        id_indication = self.find_id_from_label(dataframe=self.training_relation_to_id)
        df_train_filtered = self.filter_with_specific_id(dataframe=df_train_id_replaced, id_to_filter=id_indication)
        df_test_filtered = self.filter_with_specific_id(dataframe=df_test_id_replaced, id_to_filter=id_indication)
        # add a mark column in train and test, it marks the positive pairs
        df_train_filtered = self.add_mark_col(dataframe=df_train_filtered, mark=1)
        df_test_filtered = self.add_mark_col(dataframe=df_test_filtered, mark=1)
        # generate negative sample and create a mark column with 0 to mark the negative sample before concat with
        # train and test
        df_neg_sample = self.generate_negative_sample(dataframe=df_dd_pairs)
        df_neg_sample = self.add_mark_col(dataframe=df_neg_sample, mark=0)
        # split the neg sample before the concat with train and test
        neg_sample_train, neg_sample_test = train_test_split(df_neg_sample, test_size=0.2, random_state=42)
        # remove the relation column from train and test, it is useless from now
        df_train_filtered = self.remove_chosen_column(dataframe=df_train_filtered, col_name="relation")
        df_test_filtered = self.remove_chosen_column(dataframe=df_test_filtered, col_name="relation")
        # now concat train + neg_sample_train and test + neg_sample_test
        df_train = concat([df_train_filtered, neg_sample_train]).reset_index(drop=True)
        df_test = concat([df_test_filtered, neg_sample_test]).reset_index(drop=True)
        # shuffle the row in df_train and df_test
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_test = df_test.sample(frac=1).reset_index(drop=True)
        dict_id_entity = self.create_dict_from_df(dataframe=self.entities_to_id, col1="label", col2="id")
        # remap entity to ids for train
        df_train['head'] = df_train['head'].map(dict_id_entity)
        df_train['tail'] = df_train['tail'].map(dict_id_entity)
        df_train = df_train.dropna()
        df_train = df_train.astype(int)
        # remap entity to ids for test
        df_test['head'] = df_test['head'].map(dict_id_entity)
        df_test['tail'] = df_test['tail'].map(dict_id_entity)
        df_test = df_test.dropna()
        df_test = df_test.astype(int)

        if self.m4 is not None:
            # replace id by the vector representation in train
            df_train = self.replace_data_in_df(dataframe=df_train, col1='head', col2='tail',
                                               dict_id=self.id_to_embedding)
            # replace id by the vector representation in test
            df_test = self.replace_data_in_df(dataframe=df_test, col1='head', col2='tail',
                                              dict_id=self.id_to_embedding)
        else:
            # replace id by the vector representation in train
            df_train = self.replace_data_in_df(dataframe=df_train, col1='head', col2='tail',
                                               dict_id=self.id_to_embedding)
            # replace id by the vector representation in test
            df_test = self.replace_data_in_df(dataframe=df_test, col1='head', col2='tail',
                                              dict_id=self.id_to_embedding)

        return df_train, df_test

    def map_id_to_embedding(self) -> dict:
        id_to_embedding = {}
        for index, row in self.entities_to_id.iterrows():
            entity_id = row['id']
            embedding_index = entity_id
            embedding_vector = self.entity_embedding[embedding_index]
            id_to_embedding[entity_id] = embedding_vector
        return id_to_embedding

    def remove_chosen_column(self, dataframe: DataFrame, col_name: str) -> DataFrame:
        if col_name in dataframe.columns:
            dataframe.drop(columns=[col_name], inplace=True)
            return dataframe
        else:
            print(f"Column '{col_name}' not found in the DataFrame.")
            return dataframe

    def replace_data_in_df(self, dataframe: DataFrame, dict_id: dict, col1: str, col2: str) -> DataFrame:
        """
        Replace values in specified columns of a DataFrame using a dictionary.

        Args:
            dataframe (DataFrame): The DataFrame to be modified.
            dict_id (dict): Dictionary mapping old values to new values.
            col1 (str): Name of the first column to be modified.
            col2 (str): Name of the second column to be modified.

        Returns:
            DataFrame: The modified DataFrame.
        """
        if col1 not in dataframe.columns or col2 not in dataframe.columns:
            raise ValueError("Specified column(s) not found in the DataFrame.")

        new_df = dataframe.copy()  # Create a copy to avoid modifying the original DataFrame
        for col in [col1, col2]:
            new_df[col] = new_df[col].map(dict_id)

        return new_df

    def save_files(self) -> None:
        print("vector_vector_mask.tsv saved")

        if self.m4 is not None:
            directory_path = self.m4.joinpath('data_preprocess_classifier')
        else:
            directory_path = self.input_path.joinpath('data_preprocess_classifier')
        file_path_train = directory_path.joinpath('train.tsv')
        file_path_test = directory_path.joinpath('test.tsv')
        # Create the directory if it doesn't exist
        directory_path.mkdir(parents=True, exist_ok=True)
        # Save the DataFrame to the file
        self.df_train['head'] = self.df_train['head'].apply(
            lambda x: ','.join(map(str, x)))
        self.df_train['tail'] = self.df_train['tail'].apply(lambda x: ','.join(map(str, x)))
        self.df_train.to_csv(file_path_train, sep='\t', index=False)

        self.df_test['head'] = self.df_test['head'].apply(
            lambda x: ','.join(map(str, x)))
        self.df_test['tail'] = self.df_test['tail'].apply(lambda x: ','.join(map(str, x)))
        self.df_test.to_csv(file_path_test, sep='\t', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add description here")

    parser.add_argument("--folder", type=str, help="Path to the trained model' folder.")
    parser.add_argument("--mapping", type=str, help="Path to the file mapping the entities and their id.")
    parser.add_argument("--m4", type=str, help=".")
    parser.add_argument("--output", type=str, help="Path the file where the image will be saved.")

    args = parser.parse_args()

    an_input = args.folder
    classifier = ClassifierDataPreProcess(path=an_input, path_to_method4_preprocess=args.mapping, m4=args.m4)


    print(classifier.save_files())
    #print(classifier.get_embedd())
