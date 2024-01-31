from pandas import concat, DataFrame, merge
from sklearn.model_selection import train_test_split
from prepare_kg import PrepareKG


class PrepareKGSecondMethod(PrepareKG):

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train="benchmark/data/train_set_second_method.csv",
                 output_test="benchmark/data/test_set_second_method.csv",
                 output_val="benchmark/data/val_set_second_method.csv"):
        super().__init__(kg_path, output_nodes_map, output_kg_edge_list, output_train, output_test, output_val)

    def find_reverse_to_remove(self, graph: DataFrame) -> DataFrame:
        merged_df = merge(graph, graph, left_on=['from', 'to'], right_on=['to', 'from'], suffixes=('_first', '_reverse'))
        reverse_to_remove = merged_df[['from_reverse', 'to_reverse', 'rel_reverse']].copy()
        reverse_to_remove.columns = ['from','to','rel']
        print(reverse_to_remove)
        return reverse_to_remove

    def remove_reverse_relation(self, graph: DataFrame) -> DataFrame:
        reverse_to_remove = self.find_reverse_to_remove(graph)
        graph_wo_reverse = graph[
            ~graph.set_index(['from', 'to']).index.isin(reverse_to_remove.set_index(['from', 'to']).index)]
        graph_wo_reverse.to_csv('benchmark/data/graph_wo_reverse.csv', sep="\t", index=False)
        return graph_wo_reverse

    def find_redundant_relation(self, graph):
        # Grouper les données par les colonnes "from" et "to" et compter le nombre de valeurs uniques pour la colonne "rel"
        grouped = graph.groupby(['from', 'to'])['rel'].nunique()
        # Filtrer les groupes où le nombre de valeurs uniques pour "rel" est supérieur à 1 (relations redondantes)
        redundant_groups = grouped[grouped > 1].reset_index()
        # Fusionner les données d'origine avec les groupes redondants pour obtenir les relations redondantes complètes
        redundant_relations = merge(graph, redundant_groups, on=['from', 'to'], how='inner')
        reduced_relation = redundant_relations.drop(redundant_relations.index[::2]).reset_index(drop=True)
        redundant_relations = reduced_relation[['from', 'to', 'rel_x']].copy()
        redundant_relations.columns = ['from', 'to', 'rel']
        return redundant_relations


    def remove_redundant_relation(self, graph: DataFrame) -> DataFrame:
        redundant_to_remove = self.find_redundant_relation(graph)
        graph_wo_redundant = graph[
            ~graph.set_index(['from', 'to', 'rel']).index.isin(redundant_to_remove.set_index(['from', 'to', 'rel']).index)]
        print(graph_wo_redundant)
        graph_wo_redundant.to_csv('benchmark/data/graph_wo_redundant.csv', sep="\t", index=False)
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
        full_graph, new_nodes = self.generate_edgelist()

        full_graph = self.expand_graph_relations(full_graph)
        print(f"FULL_GRAPH BEFORE SAVING:\n{full_graph}")
        self.saving_dataframe(full_graph, new_nodes)
        self.print_relations_count(full_graph)
        full_graph = self.remove_reverse_relation(full_graph)
        full_graph = self.remove_redundant_relation(full_graph)
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