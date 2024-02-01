from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from prepare_kg_second_method import PrepareKGSecondMethod


class PrepareKGFourthMethod(PrepareKGSecondMethod):

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_fourth_method.csv",
                 output_test="benchmark/data/test_set_fourth_method.csv",
                 output_val="benchmark/data/val_set_fourth_method.csv"):
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)


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

if __name__ == "__main__":
    prepare_kg = PrepareKGFourthMethod(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_THIRD_METHOD.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_THIRD_METHOD.txt")

    prepare_kg.main()