from pandas import DataFrame, read_csv
from typing import Tuple


class PrepareData:
    """
    A class for preparing and encoding knowledge graph data.

    Parameters:
    - data_path (str): The file path to the knowledge graph data.

    Methods:
    - load_data: Loads knowledge graph data from CSV files into a DataFrame.
    - encode_relations_and_save: Encodes relations, saves the mapping, and returns the updated DataFrame.
    - segment_data_by_mask: Segments data into training, validation, and test sets based on a 'mask' feature.
    - create_dataframes: Creates DataFrames with encoded relations from provided data dictionaries.
    - count_unique_items: Counts the number of unique items in a specified column of a DataFrame.
    - main: Main method that orchestrates the data preparation process.
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path


    def encode_relations_and_save(self, path: str) -> DataFrame:
        """
            Encodes relations in a DataFrame, saves the relation-to-index mapping in a CSV file, and returns the updated DataFrame.

            Parameters:
            - path (str): The file path to the CSV file containing the DataFrame.

            Returns:
            - pd.DataFrame: The DataFrame with encoded relations.
        """
        df = read_csv(path, sep="\t", low_memory=False)
        relation_dict = {relation: idx for idx, relation in enumerate(set(df['full_relation']))}

        print(relation_dict)
        # Save relation and their index in a csv file
        relation_df = DataFrame(list(relation_dict.items()), columns=['relation', 'index'])
        relation_df.to_csv("benchmark/data/relation_id.csv", index=False)

        df['full_relation'] = df['full_relation'].map(relation_dict)
        return df

    def segment_data_by_mask(self, df: DataFrame) -> Tuple[dict, dict, dict]:
        """
        Segments a DataFrame into training, validation, and test sets based on a 'mask' feature.

        Parameters:
        - df (DataFrame): The input DataFrame containing the 'mask' feature.

        Returns:
        - Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]]]: 
        A tuple containing three dictionaries:
            - train_data_dict: Dictionary containing 'x_idx', 'full_relation', and 'y_idx' for the training set.
            - val_data_dict: Dictionary containing 'x_idx', 'full_relation', and 'y_idx' for the validation set.
            - test_data_dict: Dictionary containing 'x_idx', 'full_relation', and 'y_idx' for the test set.
        """
        train_data_dict = {"from": [], "rel": [], "to": []}
        val_data_dict = {"from": [], "rel": [], "to": []}
        test_data_dict = {"from": [], "rel": [], "to": []}

        for index, row in df.iterrows():

            feature = row["mask"]

            if feature == "train":
                train_data_dict["from"].append(row['x_idx'])
                train_data_dict["rel"].append(row['full_relation'])
                train_data_dict["to"].append(row['y_idx'])

            elif feature == "val":
                val_data_dict["from"].append(row['x_idx'])
                val_data_dict["rel"].append(row['full_relation'])
                val_data_dict["to"].append(row['y_idx'])

            elif feature == "test":
                test_data_dict["from"].append(row['x_idx'])
                test_data_dict["rel"].append(row['full_relation'])
                test_data_dict["to"].append(row['y_idx'])

        return train_data_dict, val_data_dict, test_data_dict

    def create_dataframes(self, train_data_dict: dict, val_data_dict: dict, test_data_dict: dict) -> Tuple[
        DataFrame, DataFrame, DataFrame]:
        """
        Encodes relations in the provided data dictionaries and returns DataFrames with encoded relations.

        Parameters:
        - train_data_dict (dict): Dictionary containing 'from', 'rel', and 'to' for the training set.
        - val_data_dict (dict): Dictionary containing 'from', 'rel', and 'to' for the validation set.
        - test_data_dict (dict): Dictionary containing 'from', 'rel', and 'to' for the test set.

        Returns:
        - Tuple[DataFrame, DataFrame, DataFrame]: 
        A tuple containing three DataFrames with encoded relations for training, validation, and test sets.
        """
        train_data = DataFrame(train_data_dict)
        val_data = DataFrame(val_data_dict)
        test_data = DataFrame(test_data_dict)

        return train_data, val_data, test_data

    def count_unique_items(self, path: str, column_name: str) -> int:
        """
        Count the number of unique items in a specified column of a DataFrame.

        Parameters:
        - path (str): The file path to the CSV file containing the DataFrame.
        - column_name (str): The name of the column for which unique items are counted.

        Returns:
        - int: The count of unique items in the specified column.
        """
        df = read_csv(path, sep="\t", low_memory=False)
        unique_items_count = df[column_name].nunique()
        return unique_items_count

    def main(self) -> None:
        print(
            f"Number of different relation in the knowledge graph : {self.count_unique_items(self.data_path, 'full_relation')}")
        train_data_dict, val_data_dict, test_data_dict = self.segment_data_by_mask(
            self.encode_relations_and_save(self.data_path))
        print("\nCreated dictionary : \n - train_data_dict \n - val_data_dict \n - test_data_dict \n")
        train_data, val_data, test_data = self.create_dataframes(train_data_dict, val_data_dict, test_data_dict)
        print("train_set head : \n", train_data.head())

        train_data.to_csv('benchmark/data/train_set.csv', index=False, header=False)
        val_data.to_csv('benchmark/data/val_set.csv', index=False, header=False)
        test_data.to_csv('benchmark/data/test_set.csv', index=False, header=False)


if __name__ == "__main__":
    prepare_data = PrepareData('benchmark/data/KG_edgelist_mask.txt')
    print(prepare_data.main())

    # print(prepare_data.split_kg(kg_df))
