from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from prepare_kg_second_method import PrepareKGSecondMethod


class PrepareKGThirdMethod(PrepareKGSecondMethod):

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_third_method.csv",
                 output_test="benchmark/data/test_set_third_method.csv",
                 output_val="benchmark/data/val_set_third_method.csv"):
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)

    def get_unique_values(self, graph: DataFrame, column_name: str) -> list:
        return graph[column_name].unique()

    def split_dataframe_based_on_relation(self, graph: DataFrame, column_name: str, unique_rel: list) -> dict:
        relation_dataframes = {}
        for rel in unique_rel:
            rel_df = graph[graph[column_name] == rel].copy()
            relation_dataframes[rel] = rel_df
        return relation_dataframes

    def split_each_dataframe_into_train_test_val(self, relations_dict_dataframe: dict, test_size=0.2,
                                                 val_size=0.1, random_state=None) -> dict:
        relation_train_test_val_splits = {}
        for rel, rel_df in relations_dict_dataframe.items():
            train_set, test_set = train_test_split(rel_df, test_size=test_size, random_state=random_state)
            train_set, val_set = train_test_split(train_set, test_size=val_size, random_state=random_state)
            relation_train_test_val_splits[rel] = (train_set, test_set, val_set)
        return relation_train_test_val_splits

    def concat_split_sets(self, relation_train_test_val_splits:dict) -> tuple:
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
        for rel, (train_set, test_set, val_set) in relation_train_test_val_splits.items():
            concatenated_train_set = concat([concatenated_train_set, train_set])
            concatenated_test_set = concat([concatenated_test_set, test_set])
            concatenated_val_set = concat([concatenated_val_set, val_set])

        return concatenated_train_set, concatenated_test_set, concatenated_val_set

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
        unique_rel = self.get_unique_values(graph=full_graph, column_name="rel")
        rel_df = self.split_dataframe_based_on_relation(graph=full_graph, column_name="rel", unique_rel=unique_rel)
        dict_train_test_split = self.split_each_dataframe_into_train_test_val(relations_dict_dataframe=rel_df,
                                                                              random_state=3)

        train_set, test_set, val_set = self.concat_split_sets(relation_train_test_val_splits=dict_train_test_split)
        self.save_train_test_val(train=train_set, test=test_set, val=val_set)


if __name__ == "__main__":
    prepare_kg = PrepareKGThirdMethod(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_THIRD_METHOD.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_THIRD_METHOD.txt")

    prepare_kg.main()
