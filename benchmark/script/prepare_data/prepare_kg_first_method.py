from pandas import concat, DataFrame, merge
from prepare_kg import PrepareKG


class PrepareKGFirstMethod(PrepareKG):

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/first_method/train_set_first_method_modif.csv",
                 output_test="benchmark/data/first_method/test_set_first_method_modif.csv",
                 output_val="benchmark/data/first_method/val_set_first_method_modif.csv",
                 output_type_to_entities="benchmark/data/first_method/type_to_entities_first.csv"):
        super().__init__(kg_path, output_nodes_map)
        self.output_train = output_train
        self.output_test = output_test
        self.output_val = output_val
        self.output_type_to_entities = output_type_to_entities

    def find_reciprocal_relationships(self, graph: DataFrame) -> DataFrame:
        """
        Find reciprocal relationships in a graph DataFrame.
        """
        merged = merge(graph, graph, left_on=['x_idx', 'y_idx', 'relation'], right_on=['y_idx', 'x_idx', 'relation'])
        true_reverse = merged[(merged['x_idx_x'] == merged['y_idx_y']) & (merged['y_idx_x'] == merged['x_idx_y'])]

        df_relation = true_reverse["relation"]
        df_x = true_reverse[[col for col in true_reverse.columns if col.endswith('_x')]]
        df_y = true_reverse[[col for col in true_reverse.columns if col.endswith('_y')]]

        df_x.columns = df_x.columns.str.rstrip('_x')
        df_y.columns = df_y.columns.str.rstrip('_y')

        df_x = concat([df_relation, df_x], axis=1)
        df_y = concat([df_relation, df_y], axis=1)

        df_x.columns = ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name', 'x_source', 'y_id',
                        'y_idx', 'y_type', 'y_name', 'y_source']
        df_y.columns = ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name', 'x_source', 'y_id',
                        'y_idx', 'y_type', 'y_name', 'y_source']

        result = concat((df_x, df_y), ignore_index=True).drop_duplicates().reset_index(drop=True)
        return result

    def add_reverse_edges(self, graph: DataFrame) -> DataFrame:
        """
        Add reverse edges to a graph DataFrame.
        """
        true_reverse = self.find_reciprocal_relationships(graph)
        rev_edges = graph[["x_name", "x_type", "relation", "y_name", "y_type"]].copy()
        reverse_relations = true_reverse[["x_name", "x_type", "relation", "y_name", "y_type"]].copy()

        merged = rev_edges.merge(reverse_relations, how='left', indicator=True)
        merged.to_csv("benchmark/output/merged.csv", sep="\t", index=False)

        rev_edges_wo_true_rev = merged[merged['_merge'] == 'left_only']
        rev_edges_wo_true_rev.to_csv("benchmark/output/tableau_1_unique.csv", sep="\t", index=False)

        rev_edges_wo_true_rev = rev_edges_wo_true_rev.drop(columns=['_merge'])
        rev_edges = rev_edges_wo_true_rev
        rev_edges.columns = ["y_name", "y_type", "relation", "x_name", "x_type"]

        rev_edge_eqtype = rev_edges.query('x_type == y_type')
        rev_edge_eqtype["relation"] = rev_edge_eqtype["relation"] + "_rev"
        rev_edge_neqtype = rev_edges.query('x_type != y_type')

        rev_edges = concat((rev_edge_eqtype, rev_edge_neqtype)).drop_duplicates(ignore_index=True).reset_index()
        return rev_edges

    def main(self):
        full_graph, new_nodes = self.generate_edgelist()

        print("Starting to get reverse edges")
        rev_edges = self.add_reverse_edges(full_graph)
        print(f"REV_EDGES {rev_edges}")

        full_graph = self.expand_graph_relations(full_graph, rev_edges)
        print(f"FULL_GRAPH BEFORE SAVING:\n{full_graph}")

        self.print_relations_count(full_graph)
        self.saving_dataframe(full_graph, new_nodes)
        train_set, test_set = self.split_train_test_val(full_graph, random_state=3)
        test_set = self.organize_col(test_set)
        train_set = self.organize_col(train_set)

        self.save_train_test_val(train=train_set, test=test_set)

        print(f"TRAIN_SET HERE:\n{train_set}")

        proportion_rev_added, proportion_rev_not_added, proportion_false_rev = (
            self.calculate_reverse_relation_proportion(train_set, test_set))

        print(f"Proportion of reverse relation in test that got their reverse in train: {proportion_rev_added}%\n"
              f"Proportion of reverse relation that were already in the data: {proportion_rev_not_added}%\n"
              f"Proportion in test set of reverse relation with different relation name: {proportion_false_rev}%")


if __name__ == "__main__":
    prepare_kg = PrepareKGFirstMethod(
        kg_path='benchmark/data/kg_giant_orphanet.csv',
        output_nodes_map="benchmark/data/first_method/KG_node_map_FIRST_METHOD_modif.txt",
        output_kg_edge_list="benchmark/data/first_method/KG_edgelist_mask_FIRST_METHOD_modif.txt",
        output_type_to_entities="benchmark/data/first_method/type_to_entities_first.csv"
    )
    prepare_kg.main()
