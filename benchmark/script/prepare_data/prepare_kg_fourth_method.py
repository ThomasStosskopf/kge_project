from pandas import DataFrame, concat, read_csv
from prepare_kg_third_method import PrepareKGThirdMethod
from typing import Union
import argparse

class PrepareKGFourthMethod(PrepareKGThirdMethod):
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

    def __init__(self, kg_path: str, output_folder: str):
        """
        Initialize the PrepareKGThirdMethod object.

        Parameters:
        - kg_path (str): The path to the knowledge graph file.
        - output_nodes_map (str): The path to the output file containing node mappings.
        - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
        - output_train (str, optional): The path to save the training set. Defaults to "benchmark/data/train_set_second_method.csv".
        - output_test (str, optional): The path to save the testing set. Defaults to "benchmark/data/test_set_second_method.csv".
        - output_val (str, optional): The path to save the validation set. Defaults to "benchmark/data/val_set_second_method.csv".
        """
        super().__init__(kg_path, output_folder)

    def find_specific_relation(self, df: DataFrame, column: str, relations: Union[str, list]) -> tuple[
        DataFrame, DataFrame]:
        """
        Find specific relation(s) in the DataFrame and remove them.

        Parameters:
        - df (DataFrame): The DataFrame to search for the specific relation(s).
        - column (str): The name of the column where relations are stored.
        - relations (Union[str, list]): The relation(s) to find and remove. It can be a single relation or a list of relations.

        Returns:
        - tuple[DataFrame, DataFrame]: A tuple containing two DataFrames:
                                       - The first DataFrame contains data excluding the specific relation(s).
                                       - The second DataFrame contains data specific to the given relation(s).
        """
        # Convert single relation to a list if it's not already a list
        if isinstance(relations, str):
            relations = [relations]

        # Create boolean masks for each relation in the list
        relation_masks = [df[column].str.contains(rel) for rel in relations]

        # Combine the masks with logical OR to get a mask for all relations to remove
        mask_to_remove = concat(relation_masks, axis=1).any(axis=1)

        # Invert the mask to get a mask for data excluding the specific relation(s)
        mask_to_keep = ~mask_to_remove

        # Create DataFrames based on the masks
        df_excluding_relations = df[mask_to_keep]
        df_specific_relations = df[mask_to_remove]

        return df_excluding_relations, df_specific_relations

    def percentage_of_specific_rel_in_dataset(self, graph: DataFrame, df_relation_specific: DataFrame) -> float:
        """
        Calculate the percentage of a specific relation in the dataset.

        Parameters:
        - graph (DataFrame): The original DataFrame representing the entire dataset.
        - df_relation_specific (DataFrame): The DataFrame containing data specific to a certain relation.

        Returns:
        - float: The percentage of the specific relation in the dataset.
        """
        return round((len(df_relation_specific) / len(graph)), 3)

    def concat_test_and_specific_relation(self, test_set: DataFrame, specific_relation: DataFrame) -> DataFrame:
        """
        Concatenate the test set with the data we removed from the full_graph knowledge graph.

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

        # Vérifiez les chemins de sauvegarde
        print("Output kg path: ", self.kg_path)
        print("Output train path:", self.output_train)
        print("Output test path:", self.output_test)

        full_graph = self.expand_graph_relations(full_graph)
        print(f"FULL_GRAPH BEFORE SAVING:\n{full_graph}")
        self.saving_dataframe(full_graph, new_nodes)
        self.print_relations_count(full_graph)
        list_relation_to_remove_from_train = [";indication;", ";off-label use;"]
        # Find the specify relation and put it in a dataframe for later
        full_graph_filtered, df_specific_rel = self.find_specific_relation(df=full_graph,
                                                                           column="rel",
                                                                           relations=list_relation_to_remove_from_train)
        percentage_rel_specific = self.percentage_of_specific_rel_in_dataset(graph=full_graph,
                                                                             df_relation_specific=df_specific_rel)
        print(f"#################\n{full_graph_filtered}\n####################")
        print(f"#################\n{df_specific_rel}\n####################")
        full_graph_filtered.to_csv("benchmark/data/random1/full_graph_filtered.csv", sep=",", index=False)

        unique_rel = self.get_unique_values(graph=full_graph_filtered, column_name="rel")
        # Starting the 80-20% split train-test between all types of edges here
        rel_df = self.split_dataframe_based_on_relation(graph=full_graph_filtered, column_name="rel",
                                                        unique_rel=unique_rel)
        dict_train_test_split = self.split_each_dataframe_into_train_test_val(relations_dict_dataframe=rel_df,
                                                                              random_state=3)
        train_set, test_set = self.concat_split_sets(relation_train_test_splits=dict_train_test_split)

        # Add the relation that we removed at the beginning
        test_set = self.concat_test_and_specific_relation(test_set=test_set, specific_relation=df_specific_rel)
        train_set = self.remove_reverse_or_redundant_in_train(train_set=train_set, test_set=test_set)
        # Change columns orders to suit pykeen
        test_set = self.organize_col(test_set)
        train_set = self.organize_col(train_set)
        # Save train and test kg files
        self.save_train_test_val(train=train_set, test=test_set)

        # train_data = read_csv('benchmark/data/train_set_fourth_method.csv', sep="\t", low_memory=False)
        # test_data = read_csv("benchmark/data/test_set_fourth_method.csv", sep="\t", low_memory=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare kg data with the second method of our benchmark")
    parser.add_argument("--input", type=str, help="Path to the kg")
    parser.add_argument("--output", type=str, help="Path to output folder.")
    args = parser.parse_args()

    prepare_kg = PrepareKGFourthMethod(kg_path=args.input,
                                       output_folder=args.output)

    prepare_kg.main()
