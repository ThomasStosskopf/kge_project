from pandas import DataFrame, merge
from sklearn.model_selection import train_test_split
from prepare_kg import PrepareKG


class PrepareKGSecondMethod(PrepareKG):
    """
    A class for preparing knowledge graphs using the second method.

    This class extends the functionality of the PrepareKG class and provides methods
    to process knowledge graphs based on specific rules and criteria. THe methods are used to remove
    reverse relationship and redundant relationships.

    Attributes:
    - kg_path (str): The path to the knowledge graph file.
    - output_nodes_map (str): The path to the output file containing node mappings.
    - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
    - output_train (str): The path to save the training set.
    - output_test (str): The path to save the testing set.
    - output_val (str): The path to save the validation set.

    Methods:
    - __init__(self, kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val):
        Initialize the PrepareKGSecondMethod object.

    - find_reverse_to_remove(self, graph):
        Find reverse relations to be removed from the graph.

    - remove_reverse_relation(self, graph):
        Remove reverse relations from the graph.

    - find_redundant_relation(self, graph):
        Find redundant relations in the graph.

    - remove_redundant_relation(self, graph):
        Remove redundant relations from the graph.

    - split_train_test_val(self, graph, test_size, val_size, random_state):
        Split the graph into training, testing, and validation sets.

    - main(self):
        Perform the main processing steps including graph expansion, relation removal,
        splitting into train-test-validation sets, and saving the processed sets.
    """

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_second_method.csv",
                 output_test="benchmark/data/test_set_second_method.csv",
                 output_val="benchmark/data/val_set_second_method.csv"):
        """
        Initialize the PrepareKGSecondMethod object.

        Parameters:
        - kg_path (str): The path to the knowledge graph file.
        - output_nodes_map (str): The path to the output file containing node mappings.
        - output_kg_edge_list (str): The path to the output file containing the knowledge graph edge list.
        - output_train (str, optional): The path to save the training set. Defaults to "benchmark/data/train_set_second_method.csv".
        - output_test (str, optional): The path to save the testing set. Defaults to "benchmark/data/test_set_second_method.csv".
        - output_val (str, optional): The path to save the validation set. Defaults to "benchmark/data/val_set_second_method.csv".
        """
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)

    def find_reverse_to_remove(self, graph: DataFrame) -> DataFrame:
        """
        Find reverse relations to be removed from the graph.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.

        Returns:
        - DataFrame: DataFrame containing reverse relations to be removed.
        """
        merged_df = merge(graph, graph, left_on=['from', 'to'], right_on=['to', 'from'], suffixes=('_first', '_reverse'))
        reverse_to_remove = merged_df[['from_reverse', 'to_reverse', 'rel_reverse']].copy()
        reverse_to_remove.columns = ['from','to','rel']
        return reverse_to_remove

    def remove_reverse_relation(self, graph: DataFrame) -> DataFrame:
        """
        Remove reverse relations from the graph.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.

        Returns:
        - DataFrame: DataFrame representing the graph with reverse relations removed.
        """
        reverse_to_remove = self.find_reverse_to_remove(graph)
        graph_wo_reverse = graph[
            ~graph.set_index(['from', 'to']).index.isin(reverse_to_remove.set_index(['from', 'to']).index)]
        return graph_wo_reverse

    def find_redundant_relation(self, graph):
        """
        Find redundant relations in the graph.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.

        Returns:
        - DataFrame: DataFrame containing redundant relations in the graph.
        """
        # Group the data by the "from" and "to" columns and count the number of unique values in the "rel" column.
        grouped = graph.groupby(['from', 'to'])['rel'].nunique()
        # Filter groups where the number of unique values for "rel" is greater than 1 (redundant relationships)
        redundant_groups = grouped[grouped > 1].reset_index()
        # Merge original data with redundant groups to obtain complete redundant relationships
        redundant_relations = merge(graph, redundant_groups, on=['from', 'to'], how='inner')
        reduced_relation = redundant_relations.drop(redundant_relations.index[::2]).reset_index(drop=True)
        redundant_relations = reduced_relation[['from', 'to', 'rel_x']].copy()
        redundant_relations.columns = ['from', 'to', 'rel']
        return redundant_relations


    def remove_redundant_relation(self, graph: DataFrame) -> DataFrame:
        """
        Remove redundant relations from the graph.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.

        Returns:
        - DataFrame: DataFrame representing the graph with redundant relations removed.
        """
        redundant_to_remove = self.find_redundant_relation(graph)
        graph_wo_redundant = graph[
            ~graph.set_index(['from', 'to', 'rel']).index.isin(redundant_to_remove.set_index(['from', 'to', 'rel']).index)]

        return graph_wo_redundant

    def split_train_test_val(self, graph: DataFrame, test_size=0.2, val_size=0.1, random_state=None) -> (
            tuple)[DataFrame, DataFrame, DataFrame]:
        """
        Split the input graph DataFrame into training, testing, and validation sets.

        This function divides the input graph DataFrame into three subsets: training set, testing set, and validation set.
        The data splitting is performed based on the specified proportions for the test and validation sets.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.
        - test_size (float, optional): The proportion of the graph to include in the test set. Defaults to 0.2.
        - val_size (float, optional): The proportion of the training set to include in the validation set.
                                      Defaults to 0.1.
        - random_state (int or None, optional): Controls the randomness of the splitting.
                                                 If specified, it ensures reproducibility of the splitting.
                                                 Defaults to None.

        Returns:
        - tuple[DataFrame, DataFrame, DataFrame]: A tuple containing DataFrames representing the training, testing,
                                                   and validation sets, respectively.

        Note:
        - The sum of test_size and val_size should be less than 1.0 to ensure that there is data left for the training set.
        - If random_state is set, the data splitting will be reproducible across multiple function calls.
        - The function utilizes the train_test_split function from scikit-learn to perform the data splitting.
        """
        train_set, test_set = train_test_split(graph, test_size=test_size, random_state=random_state)
        train_set, val_size = train_test_split(train_set, test_size=val_size, random_state=random_state)
        return train_set, test_set, val_size

    def main(self):
        """
        Perform the main processing steps including graph expansion, relation removal,
        splitting into train-test-validation sets, and saving the processed sets.
        """
        full_graph, new_nodes = self.generate_edgelist()

        full_graph = self.expand_graph_relations(full_graph)
        print(f"FULL_GRAPH BEFORE SAVING:\n{full_graph}")
        self.saving_dataframe(full_graph, new_nodes)
        full_graph = self.remove_reverse_relation(full_graph)
        full_graph = self.remove_redundant_relation(full_graph)
        self.print_relations_count(full_graph)
        print(full_graph)
        train, test, val = self.split_train_test_val(full_graph, random_state=3)
        self.save_train_test_val(train=train, test=test, val=val)


        proportion_rev_added, proportion_rev_not_added, proportion_false_rev = (
            self.calculate_reverse_relation_proportion(train, test, val))
        print(f"Proportion of reverse relation in test that got "
              f"their reverse in train: {proportion_rev_added}%\n"
              f"Proportion of reverse relation that where already in the data: {proportion_rev_not_added}%\n"
              f"Proportion in test set of reverse relation with different relation name: {proportion_false_rev}%")

if __name__ == "__main__":
    prepare_kg = PrepareKGSecondMethod(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_SECOND_METHOD.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_SECOND_METHOD.txt")

    prepare_kg.main()
