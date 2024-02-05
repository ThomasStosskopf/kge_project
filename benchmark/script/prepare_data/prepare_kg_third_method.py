from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from prepare_kg_second_method import PrepareKGSecondMethod

class PrepareKGThirdMethod(PrepareKGSecondMethod):
    """
    A class for preparing knowledge graphs using the third method.

    This class extends the functionality of the PrepareKGSecondMethod class and provides methods
    to split the graph based on relations and concatenate the split sets. We split the full graph into different
    dataframe each dataframe containing only a specific relationship. Then each dataframe is split into train and
    test set. 80% in train, 20% in test. At the end all the train set are concat together and all the test set are
    concat together.

    Attributes:
    - kg_path (str): The path to the knowledge graph file.
    - output_nodes_map (str): The path to the output file containing node mappings.
    - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
    - output_train (str): The path to save the training set.
    - output_test (str): The path to save the testing set.
    - output_val (str): The path to save the validation set.

    Methods:
    - __init__(self, kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val):
        Initialize the PrepareKGThirdMethod object.

    - get_unique_values(self, graph, column_name):
        Get unique values from a specific column in the graph DataFrame.

    - split_dataframe_based_on_relation(self, graph, column_name, unique_rel):
        Split the DataFrame based on relation types.

    - split_each_dataframe_into_train_test_val(self, relations_dict_dataframe, test_size, val_size, random_state):
        Split each DataFrame representing a relation into train-test-validation sets.

    - concat_split_sets(self, relation_train_test_val_splits):
        Concatenate the split sets for each relation type into one big DataFrame per set type.

    - main(self):
        Perform the main processing steps including graph expansion, relation removal,
        splitting into train-test-validation sets, and saving the processed sets.
    """

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_third_method.csv",
                 output_test="benchmark/data/test_set_third_method.csv",
                 output_val="benchmark/data/val_set_third_method.csv"):
        """
        Initialize the PrepareKGThirdMethod object.

        Parameters:
        - kg_path (str): The path to the knowledge graph file.
        - output_nodes_map (str): The path to the output file containing node mappings.
        - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
        - output_train (str, optional): The path to save the training set. Defaults to "benchmark/data/train_set_third_method.csv".
        - output_test (str, optional): The path to save the testing set. Defaults to "benchmark/data/test_set_third_method.csv".
        - output_val (str, optional): The path to save the validation set. Defaults to "benchmark/data/val_set_third_method.csv".
        """
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)

    def get_unique_values(self, graph: DataFrame, column_name: str) -> list:
        """
        Get unique values from a specific column in the graph DataFrame.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.
        - column_name (str): The name of the column from which unique values are extracted.

        Returns:
        - list: A list of unique values from the specified column.
        """
        return graph[column_name].unique()

    def split_dataframe_based_on_relation(self, graph: DataFrame, column_name: str, unique_rel: list) -> dict:
        """
        Split the DataFrame based on relation types.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.
        - column_name (str): The name of the column representing relations.
        - unique_rel (list): A list of unique relation types.

        Returns:
        - dict: A dictionary where keys are relation types and values are DataFrames containing data
                corresponding to each relation type.
        """
        relation_dataframes = {}
        for rel in unique_rel:
            rel_df = graph[graph[column_name] == rel].copy()
            relation_dataframes[rel] = rel_df
        return relation_dataframes

    def split_each_dataframe_into_train_test_val(self, relations_dict_dataframe: dict, test_size=0.2,
                                                 val_size=0.1, random_state=None) -> dict:
        """
        Split each DataFrame representing a relation into train-test-validation sets.

        Parameters:
        - relations_dict_dataframe (dict): A dictionary where keys are relation types, and the corresponding
                                           values are DataFrames containing data for each relation type.
        - test_size (float, optional): The proportion of the data to include in the test set. Defaults to 0.2.
        - val_size (float, optional): The proportion of the training set to include in the validation set. Defaults to 0.1.
        - random_state (int or None, optional): Controls the randomness of the splitting. Defaults to None.

        Returns:
        - dict: A dictionary where keys are relation types, and the corresponding values are tuples
                containing DataFrames representing the training, testing, and validation sets, respectively.
        """
        relation_train_test_splits = {}
        for rel, rel_df in relations_dict_dataframe.items():
            train_set, test_set = train_test_split(rel_df, test_size=test_size, random_state=random_state)
            # train_set, val_set = train_test_split(train_set, test_size=val_size, random_state=random_state)
            relation_train_test_splits[rel] = (train_set, test_set)
        return relation_train_test_splits

    def concat_split_sets(self, relation_train_test_splits:dict) -> tuple:
        """
        Concatenate the split sets for each relation type into one big dataframe per set type.

        Parameters:
        - relation_train_test_val_splits (dict): A dictionary where each key is a relation type, and the corresponding
                                                 value is a tuple containing DataFrames representing the training,
                                                 testing, and validation sets, respectively.

        Returns:
        - tuple[DataFrame, DataFrame, DataFrame]: A tuple containing DataFrames representing the concatenated
                                                   training, testing, and validation sets, respectively.
        """
        # Initialize empty DataFrames for concatenated sets
        concatenated_train_set = DataFrame()
        concatenated_test_set = DataFrame()
        concatenated_val_set = DataFrame()

        # Concatenate sets for each relation type
        for rel, (train_set, test_set) in relation_train_test_splits.items():
            concatenated_train_set = concat([concatenated_train_set, train_set])
            concatenated_test_set = concat([concatenated_test_set, test_set])
            # concatenated_val_set = concat([concatenated_val_set, val_set])

        return concatenated_train_set, concatenated_test_set

    def main(self) -> None:
        """
        Perform the main processing steps including graph expansion, relation removal,
        splitting into train-test-validation sets, and saving the processed sets.
        """
        full_graph, new_nodes = self.generate_edgelist()

        full_graph = self.expand_graph_relations(full_graph)
        print(f"FULL_GRAPH BEFORE SAVING:\n{full_graph}")
        self.saving_dataframe(full_graph, new_nodes)
        self.print_relations_count(full_graph)
        unique_rel = self.get_unique_values(graph=full_graph, column_name="rel")
        rel_df = self.split_dataframe_based_on_relation(graph=full_graph, column_name="rel", unique_rel=unique_rel)
        dict_train_test_split = self.split_each_dataframe_into_train_test_val(relations_dict_dataframe=rel_df,
                                                                              random_state=3)

        train_set, test_set = self.concat_split_sets(relation_train_test_splits=dict_train_test_split)

        train_set = self.remove_reverse_or_redundant_in_train(train_set=train_set, test_set=test_set)

        self.save_train_test_val(train=train_set, test=test_set)


if __name__ == "__main__":
    prepare_kg = PrepareKGThirdMethod(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_THIRD_METHOD.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_THIRD_METHOD.txt")

    prepare_kg.main()
