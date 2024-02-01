from pandas import DataFrame, concat
from prepare_kg_second_method import PrepareKGSecondMethod


class PrepareKGFourthMethod(PrepareKGSecondMethod):
    """
    A class for preparing knowledge graphs using the fourth method.

    This class extends the functionality of the PrepareKGSecondMethod class and provides methods
    to handle specific relations in the graph.

    Attributes:
    - kg_path (str): The path to the knowledge graph file.
    - output_nodes_map (str): The path to the output file containing node mappings.
    - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
    - output_train (str): The path to save the training set.
    - output_test (str): The path to save the testing set.
    - output_val (str): The path to save the validation set.

    Methods:
    - __init__(self, kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val):
        Initialize the PrepareKGFourthMethod object.

    - find_specific_relation(self, df, column, relation):
        Find a specific relation in the DataFrame.

    - percentage_of_specific_rel_in_dataset(self, graph, df_relation_specific):
        Calculate the percentage of a specific relation in the dataset.

    - concat_test_and_specific_relation(self, test_set, specific_relation):
        Concatenate the test set with data specific to a certain relation.

    - main(self):
        Perform the main processing steps including graph expansion, relation removal,
        splitting into train-test-validation sets, and saving the processed sets.
    """

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_fourth_method.csv",
                 output_test="benchmark/data/test_set_fourth_method.csv",
                 output_val="benchmark/data/val_set_fourth_method.csv"):
        """
        Initialize the PrepareKGFourthMethod object.

        Parameters:
        - kg_path (str): The path to the knowledge graph file.
        - output_nodes_map (str): The path to the output file containing node mappings.
        - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
        - output_train (str, optional): The path to save the training set. Defaults to "benchmark/data/train_set_fourth_method.csv".
        - output_test (str, optional): The path to save the testing set. Defaults to "benchmark/data/test_set_fourth_method.csv".
        - output_val (str, optional): The path to save the validation set. Defaults to "benchmark/data/val_set_fourth_method.csv".
        """
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)

    def find_specific_relation(self, df: DataFrame, column: str, relation: str) -> tuple[DataFrame, DataFrame]:
        """
        Find a specific relation in the DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame to search for the specific relation.
        - column (str): The name of the column where relations are stored.
        - relation (str): The specific relation to find.

        Returns:
        - tuple[DataFrame, DataFrame]: A tuple containing two DataFrames:
                                       - The first DataFrame contains data excluding the specific relation.
                                       - The second DataFrame contains data specific to the given relation.
        """
        return df[~(df[column].str.contains(relation))], df[(df[column].str.contains(relation))]

    def percentage_of_specific_rel_in_dataset(self, graph: DataFrame, df_relation_specific: DataFrame) -> float:
        """
        Calculate the percentage of a specific relation in the dataset.

        Parameters:
        - graph (DataFrame): The original DataFrame representing the entire dataset.
        - df_relation_specific (DataFrame): The DataFrame containing data specific to a certain relation.

        Returns:
        - float: The percentage of the specific relation in the dataset.
        """
        return round((len(df_relation_specific)/len(graph)), 3)

    def concat_test_and_specific_relation(self, test_set: DataFrame, specific_relation: DataFrame) -> DataFrame:
        """
        Concatenate the test set with data specific to a certain relation.

        Parameters:
        - test_set (DataFrame): The original test set DataFrame.
        - specific_relation (DataFrame): The DataFrame containing data specific to a certain relation.

        Returns:
        - DataFrame: The concatenated DataFrame containing the original test set and data specific to a certain relation.
        """
        return concat([test_set, specific_relation], axis=0)

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
        full_graph = self.remove_reverse_relation(full_graph)
        full_graph = self.remove_redundant_relation(full_graph)
        full_graph_filtered, df_specific_rel = self.find_specific_relation(df=full_graph,
                                                                           column="rel",
                                                                           relation="indication")
        percentage_rel_specific = self.percentage_of_specific_rel_in_dataset(graph=full_graph,
                                                                             df_relation_specific=df_specific_rel)
        train, test, val = self.split_train_test_val(graph=full_graph_filtered,
                                                    test_size=0.2-percentage_rel_specific,
                                                   random_state=3)

        test = self.concat_test_and_specific_relation(test_set=test, specific_relation=df_specific_rel)

        self.save_train_test_val(test=test, train=train, val=val)

        print(percentage_rel_specific)


if __name__ == "__main__":
    prepare_kg = PrepareKGFourthMethod(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_FOURTH_METHOD.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_FOURTH_METHOD.txt")

    prepare_kg.main()