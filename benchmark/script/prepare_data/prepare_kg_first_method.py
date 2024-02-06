from pandas import concat, DataFrame, merge
from prepare_kg import PrepareKG


class PrepareKGFirstMethod(PrepareKG):

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_first_method.csv",
                 output_test="benchmark/data/test_set_first_method.csv",
                 output_val="benchmark/data/val_set_first_method.csv"):
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)

    def find_reciprocal_relationships(self, graph: DataFrame) -> DataFrame:
        """
        Find reciprocal relationships in a graph DataFrame.

        This function identifies reciprocal relationships within a DataFrame representing a graph.
        Reciprocal relationships occur when two nodes are connected by edges in both directions,
        forming bidirectional relationships.

        Parameters:
        - graph (pd.DataFrame): A DataFrame representing the graph containing relationships between nodes.
                                It should have columns 'x_idx', 'y_idx', and 'relation'.

        Returns:
        - pd.DataFrame: A DataFrame containing reciprocal relationships found in the graph.
                        The DataFrame includes columns representing relationships between nodes.

        Note:
        - Reciprocal relationships are identified by merging the graph DataFrame with itself based
          on the 'x_idx', 'y_idx', and 'relation' columns.
        - Only rows where 'x_idx_x' equals 'y_idx_y' and 'y_idx_x' equals 'x_idx_y' are retained,
          indicating bidirectional relationships.
        - The resulting DataFrame represents reciprocal relationships within the graph.
        """
        # Merge the graph with itself on 'x_idx' and 'y_idx'
        merged = merge(graph, graph, left_on=['x_idx', 'y_idx', 'relation'], right_on=['y_idx', 'x_idx', 'relation'])
        # Keep only the rows where 'x_idx_x' equals 'y_idx_y' and 'y_idx_x' equals 'x_idx_y'
        true_reverse = merged[(merged['x_idx_x'] == merged['y_idx_y']) & (merged['y_idx_x'] == merged['x_idx_y'])]
        # Split the DataFrame into two based on the column names
        df_relation = true_reverse["relation"]
        df_x = true_reverse[[col for col in true_reverse.columns if col.endswith('_x')]]
        df_y = true_reverse[[col for col in true_reverse.columns if col.endswith('_y')]]
        # Rename the columns to remove the suffixes for concatenation
        df_x.columns = df_x.columns.str.rstrip('_x')
        df_y.columns = df_y.columns.str.rstrip('_y')
        df_x = concat([df_relation, df_x], axis=1)
        df_y = concat([df_relation, df_y], axis=1)
        df_x.columns = ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name',
                        'x_source', 'y_id', 'y_idx', 'y_type', 'y_name', 'y_source']
        df_y.columns = ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name',
                        'x_source', 'y_id', 'y_idx', 'y_type', 'y_name', 'y_source']
        # Concatenate and reset the index
        result = concat((df_x, df_y), ignore_index=True).drop_duplicates().reset_index(drop=True)
        return result

    def add_reverse_edges(self, graph: DataFrame) -> DataFrame:
        """
        Add reverse edges to a graph DataFrame.

        This function takes a DataFrame representing a graph and adds reverse edges to it. Reverse edges
        are created by identifying reciprocal relationships and appending corresponding reverse relations to
        the graph. Reverse relations are distinguished by adding a "_rev" suffix to the original relation.

        Parameters:
        - graph (pd.DataFrame): A DataFrame representing the graph containing relationships between nodes.
                                It should have columns 'x_idx', 'x_type', 'relation', 'y_idx', and 'y_type'.

        Returns:
        - pd.DataFrame: A DataFrame containing the original graph along with additional reverse edges.
                        The DataFrame includes columns 'x_idx', 'x_type', 'y_idx', 'y_type', 'relation',
                        and 'full_relation'. The 'full_relation' column concatenates the types and relations
                        of nodes and edges for further analysis.

        Note:
        - The function utilizes the 'find_reciprocal_relationships' function to identify reciprocal relationships
          and creates reverse edges accordingly.
        - It saves intermediate results and final output to CSV files for benchmarking purposes.
        - Reverse edges are created by appending reverse relations to the original graph. If nodes have different
          types, their reverse relations are distinguished by adding a "_rev" suffix to the original relation.
        - The resulting DataFrame includes both the original graph and the newly added reverse edges.
        - The 'full_relation' column concatenates the types and relations of nodes and edges for comprehensive
          analysis and understanding of relationships in the graph.
        """
        print(graph.columns)
        true_reverse = self.find_reciprocal_relationships(graph)

        rev_edges = graph[["x_idx", "x_type", "relation", "y_idx", "y_type"]].copy()

        reverse_relations = true_reverse[["x_idx", "x_type", "relation", "y_idx", "y_type"]].copy()

        # Comparez les lignes entre les deux DataFrames en utilisant merge avec l'option indicator=True
        merged = rev_edges.merge(reverse_relations, how='left', indicator=True)
        merged.to_csv("benchmark/output/nerged.csv", sep="\t", index=False)

        # Sélectionnez les lignes qui ne sont présentes que dans tableau_1
        rev_edges_wo_true_rev = merged[merged['_merge'] == 'left_only']
        rev_edges_wo_true_rev.to_csv("benchmark/output/tableau_1_unique.csv", sep="\t", index=False)

        # Supprimez la colonne '_merge' qui a été ajoutée par merge
        rev_edges_wo_true_rev = rev_edges_wo_true_rev.drop(columns=['_merge'])

        rev_edges = rev_edges_wo_true_rev
        rev_edges.columns = ["y_idx", "y_type", "relation", "x_idx", "x_type"]

        rev_edge_eqtype = rev_edges.query('x_type == y_type')

        rev_edge_eqtype["relation"] = rev_edge_eqtype["relation"] + "_rev"
        rev_edge_neqtype = rev_edges.query('x_type != y_type')
        rev_edges = concat((rev_edge_eqtype, rev_edge_neqtype)).drop_duplicates(ignore_index=True).reset_index()
        return rev_edges



    def main(self):
        full_graph, new_nodes = self.generate_edgelist()
        print("Starting to get reverse edges")
        rev_edges = self.add_reverse_edges(full_graph)
        full_graph = self.expand_graph_relations(full_graph, rev_edges)
        print(f"FULL_GRAPH BEFORE SAVING:\n{full_graph}")
        self.saving_dataframe(full_graph, new_nodes)
        self.print_relations_count(full_graph)
        train_set, test_set = self.split_train_test_val(full_graph, random_state=3)
        # Change columns orders to suit pykeen
        test_set = self.organize_col(test_set)
        train_set = self.organize_col(train_set)
        # Save train and test kg files
        self.save_train_test_val(train=train_set, test=test_set)

        print(f"TRAIN_SET HERE:\n{train_set}")


        proportion_rev_added, proportion_rev_not_added, proportion_false_rev = (
            self.calculate_reverse_relation_proportion(train_set, test_set))
        print(f"Proportion of reverse relation in test that got "
              f"their reverse in train: {proportion_rev_added}%\n"
              f"Proportion of reverse relation that where already in the data: {proportion_rev_not_added}%\n"
              f"Proportion in test set of reverse relation with different relation name: {proportion_false_rev}%")


if __name__ == "__main__":
    prepare_kg = PrepareKGFirstMethod(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_FIRST_METHOD.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_FIRST_METHOD.txt")
    prepare_kg.main()