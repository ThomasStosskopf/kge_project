
from pandas import concat, merge, read_csv, DataFrame
from igraph import Graph
from collections import Counter
from sklearn.model_selection import train_test_split


class PrepareKG:

    def __init__(self, kg_path: str, output_nodes_map: str, output_kg_edge_list: str,
                 output_train: str, output_test: str, output_val: str):
        """
        Initialize the PrepareKG class with the provided paths for knowledge graph data and output files.

        Parameters:
        - kg_path (str): The file path to the knowledge graph data.
        - output_nodes_map (str): The file path to save the nodes mapping data.
        - output_kg_edge_list (str): The file path to save the knowledge graph edge list data.
        """
        self.kg = self.read_graph(kg_path)
        self.output_nodes_map = output_nodes_map
        self.output_kg_edge_list = output_kg_edge_list
        self.output_train = output_train
        self.output_test = output_test
        self.output_val = output_val

    def read_graph(self, path: str) -> DataFrame:
        """
        Read the knowledge graph data from a CSV file and preprocess it.

        Parameters:
        - path (str): The file path to the knowledge graph data.

        Returns:
        - DataFrame: Processed knowledge graph data as a pandas DataFrame.
        """
        graph = read_csv(path, dtype={"x_id": str, "y_id": str})
        graph = graph[graph["x_name"] != "missing"]
        graph = graph[graph["y_name"] != "missing"]
        return graph

    def clean_edges(self, df: DataFrame) -> DataFrame:
        """
        Clean the edges of the knowledge graph DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame representing the knowledge graph.

        Returns:
        - DataFrame: Cleaned knowledge graph DataFrame with valid edges.
        """
        df = df.get(
            ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name', 'x_source', 'y_id', 'y_idx', 'y_type',
             'y_name', 'y_source'])
        assert len(df[df.isna().any(axis=1)]) == 0
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.query(
            'not ((x_id == y_id) and (x_idx == y_idx) and (x_type == y_type) and (x_source == y_source) and (x_name == y_name))')
        return df

    def get_node_df(self, graph: DataFrame) -> DataFrame:  # Assign nodes to IDs between 0 and N-1 #class1
        """
        Assign node indices to nodes in the knowledge graph.

        Parameters:
        - graph (DataFrame): The DataFrame representing the knowledge graph.

        Returns:
        - DataFrame: DataFrame containing node indices assigned to nodes.
        """
        # Get all nodes
        nodes = concat([graph.get(['x_id', 'x_type', 'x_name', 'x_source']).rename(
            columns={'x_id': 'node_id', 'x_type': 'node_type', 'x_name': 'node_name', 'x_source': 'node_source'}),
                           graph.get(['y_id', 'y_type', 'y_name', 'y_source']).rename(
                               columns={'y_id': 'node_id', 'y_type': 'node_type', 'y_name': 'node_name',
                                        'y_source': 'node_source'})]).drop_duplicates(ignore_index=True)

        # Assign them to 0 to N-1
        nodes = nodes.reset_index().drop('index', axis=1).reset_index().rename(columns={'index': 'node_idx'})
        print(nodes)

        print("Finished assigning all nodes to IDs between 0 to N-1")
        return nodes

    def reindex_edges(self, graph, nodes):  # Assign node indices to nodes in edge

        # Map source nodes
        edges = merge(graph, nodes, 'left', left_on=['x_id', 'x_type', 'x_name', 'x_source'],
                         right_on=['node_id', 'node_type', 'node_name', 'node_source'])
        edges = edges.rename(columns={'node_idx': 'x_idx'})

        # Map target nodes
        edges = merge(edges, nodes, 'left', left_on=['y_id', 'y_type', 'y_name', 'y_source'],
                         right_on=['node_id', 'node_type', 'node_name', 'node_source'])
        edges = edges.rename(columns={'node_idx': 'y_idx'})

        # Subset only node info
        edges = edges.get(['x_idx', 'x_type', 'y_idx', 'y_type', 'relation', 'display_relation']).drop_duplicates(
            ignore_index=True).reset_index()
        print(edges)

        print("Finished updating edge list with new node IDs")
        return edges

    def get_LCC(self, full_graph, nodes):
        edge_index = full_graph.get(['x_idx', 'y_idx']).values.T
        graph = Graph()
        graph.add_vertices(list(range(nodes.shape[0])))
        graph.add_edges([tuple(x) for x in edge_index.T])
        graph = graph.as_undirected(mode='collapse')

        print('Before LCC - Nodes: %d' % graph.vcount())
        print('Before LCC - Edges: %d' % graph.ecount())

        c = graph.components(mode='strong')
        giant = c.giant()

        print('After LCC - Nodes: %d' % giant.vcount())
        print('After LCC - Edges: %d' % giant.ecount())

        assert not giant.is_directed()
        assert giant.is_connected()

        return giant

    def map_to_LCC(self, full_graph, giant, nodes, giant_nodes):
        new_nodes = nodes.query('node_idx in @giant_nodes')
        new_nodes = new_nodes.reset_index().drop('index', axis=1).reset_index().rename(
            columns={'index': 'new_node_idx'})
        assert new_nodes.shape[0] == giant.vcount()
        assert len(new_nodes["node_idx"].to_list()) == len(new_nodes["new_node_idx"].to_list())

        new_edges = full_graph.query('x_idx in @giant_nodes and y_idx in @giant_nodes').copy()
        new_edges = new_edges.reset_index(drop=True)
        assert new_edges.shape[0] == giant.ecount()

        new_kg = merge(new_edges, new_nodes, 'left', left_on='x_idx', right_on='node_idx')
        new_kg = new_kg.rename(columns={'node_id': 'new_x_id', 'node_type': 'new_x_type', 'node_name': 'new_x_name',
                                        'node_source': 'new_x_source', 'new_node_idx': 'new_x_idx'})
        new_kg = merge(new_kg, new_nodes, 'left', left_on='y_idx', right_on='node_idx')
        new_kg = new_kg.rename(columns={'node_id': 'new_y_id', 'node_type': 'new_y_type', 'node_name': 'new_y_name',
                                        'node_source': 'new_y_source', 'new_node_idx': 'new_y_idx'})
        new_kg = new_kg[[c for c in new_kg.columns if "new" in c or "relation" in c]]
        new_kg = new_kg.rename(columns={k: k.split("new_")[1] for k in new_kg.columns if "new" in k})

        new_kg = self.clean_edges(new_kg)

        assert max(new_kg["x_idx"].to_list() + new_kg["y_idx"].to_list()) == giant.vcount() - 1
        assert len(set(new_kg['x_idx'].tolist() + new_kg['y_idx'].tolist())) == len(giant_nodes)
        return new_kg, new_nodes

    def expand_graph_relations(self, graph: DataFrame, rev_edges=None) -> DataFrame:
        """
        Merge reverse edges into the graph DataFrame and derive full relations.

        Parameters:
        - graph: DataFrame representing the graph.
        - rev_edges: DataFrame representing reverse edges. Defaults to None.

        Returns:
        - full_graph: DataFrame with merged edges and derived full relations.
        """
        if rev_edges is not None :
            graph = concat((graph[["x_idx", "x_type", "y_idx", "y_type", "relation"]], rev_edges[
                ["x_idx", "x_type", "y_idx", "y_type", "relation"]]))

        full_graph = graph.drop_duplicates(ignore_index=True).reset_index()

        full_graph["full_relation"] = full_graph["x_type"] + ";" + full_graph["relation"] + ";" + full_graph["y_type"]
        full_graph = full_graph[["x_idx", "y_idx", "full_relation"]]
        full_graph.columns = ["from", "to", "rel"]
        full_graph = full_graph.drop_duplicates(ignore_index=True).reset_index(drop=True)

        return full_graph

    def generate_edgelist(self) -> tuple[DataFrame, DataFrame]:
        """
        Generate edge lists for the knowledge graph.

        Returns:
        - tuple[DataFrame, DataFrame]: A tuple containing DataFrames representing the full graph and new nodes.
        """
        node_map_f = self.output_nodes_map
        mask_f = self.output_kg_edge_list
        graph = self.kg
        print("Starting to process the KG table")
        nodes = self.get_node_df(graph)
        edges = self.reindex_edges(graph, nodes)

        print("Starting to generate the connected KG")
        giant = self.get_LCC(edges, nodes)
        giant_nodes = giant.vs['name']
        new_kg, new_nodes = self.map_to_LCC(edges, giant, nodes, giant_nodes)

        full_graph = new_kg
        print(full_graph)
        return full_graph, new_nodes


    def saving_dataframe(self, full_graph: DataFrame, new_nodes: DataFrame) -> None:
        """
        Save the final processed dataframes to output files.

        Parameters:
        - full_graph (DataFrame): DataFrame representing the full graph.
        - new_nodes (DataFrame): DataFrame representing the new nodes in the graph.

        Returns:
        - None
        """
        print("Starting to save final dataframes")
        new_nodes = new_nodes.get(["new_node_idx", "node_id", "node_type", "node_name", "node_source"]).rename(
            columns={"new_node_idx": "node_idx"})
        new_nodes.to_csv(self.output_nodes_map, sep="\t", index=False)
        #full_graph = self.split_edges(full_graph)
        full_graph.to_csv(self.output_kg_edge_list, sep="\t", index=False)


    def print_relations_count(self, full_graph: DataFrame) -> None:
        """
        Print the count of relations in the knowledge graph.

        Parameters:
        - full_graph (DataFrame): DataFrame representing the full graph.

        Returns:
        - None
        """
        # Count the occurrences of each relation
        relation_counts = Counter(full_graph["rel"])

        print("Final Number of Edges:", len(full_graph))  # Prints the total number of edges

        # Print each relation and its count
        for relation, count in relation_counts.items():
            print(relation, count)

        # Create a DataFrame from the relation counts and save it to a TSV file
        df_relation = DataFrame(relation_counts.items(), columns=['Relation', 'Count'])
        df_relation.to_csv("benchmark/data/relations.tsv", sep="\t", index=False)

    def save_train_test_val(self, train: DataFrame, test: DataFrame, val=None) -> None:
        train.to_csv(self.output_train, sep="\t", index=False, header=False)
        test.to_csv(self.output_test, sep="\t", index=False, header=False)
        if val is not None:
            val.to_csv(self.output_val, sep="\t", index=False, header=False)

    def reverse_relations_not_added(self, merged_df: DataFrame) -> DataFrame:
        """
        Calculate the number of reverse relations not added in the merged DataFrame.

        This function takes a DataFrame containing merged data and calculates the number of reverse relations
        that have not been added to the merged DataFrame. It removes lines where '_rev' is found in either 'rel_train'
        or 'rel_test' columns, compares the elements between the semicolons in 'rel_train' and 'rel_test', and filters
        out lines where these elements differ.

        Parameters:
        - merged_df (DataFrame): A DataFrame containing merged data with columns 'rel_train' and 'rel_test'.

        Returns:
        - int: The number of lines in the DataFrame after filtering, representing the count of reverse relations
               not added to the merged DataFrame.

        Note:
        - The function modifies the DataFrame by filtering out lines where '_rev' is found in 'rel_train' or 'rel_test'
          columns and comparing elements between semicolons in these columns.
        """
        # remove all lines where you can find _rev
        filtered_df = merged_df[~(merged_df['rel_train'].str.contains('_rev') | merged_df['rel_test'].str.contains('_rev'))]

        # Separate elements between semicolons in rel_train and rel_test
        filtered_df['rel_train_elements'] = filtered_df['rel_train'].str.split(';')
        filtered_df['rel_test_elements'] = filtered_df['rel_test'].str.split(';')

        # Compare elements between semicolons in rel_train and rel_test and filter lines
        filtered_df = filtered_df[filtered_df['rel_train_elements'].apply(lambda x: x[1]) == filtered_df['rel_test_elements'].apply(lambda x: x[1])]

        return filtered_df

    def count_reverse_relations_added(self, merged_df: DataFrame) -> int:
        """
        Count the number of reverse relations added to the merged DataFrame.

        This function calculates the number of reverse relations added to the merged DataFrame by inspecting
        the 'rel_train' and 'rel_test' columns.

        Parameters:
        - merged_df (DataFrame): A DataFrame containing merged data with columns 'rel_train' and 'rel_test'.

        Returns:
        - int: The total count of reverse relations added to the merged DataFrame.
        """
        relation_in_test = merged_df["rel_test"]
        relation_in_train = merged_df["rel_train"]
        results_filtered = [relation for relation in relation_in_test if "_rev" in relation]
        filter_rev_in_train = [relation for relation in relation_in_train if "_rev" in relation]
        # here you got the real number of reverse relations added by the prepare_graph.py script
        nb_rev_rel_added = len(results_filtered) + len(filter_rev_in_train)
        return nb_rev_rel_added


    def count_false_reverse_relation(self, merged_df: DataFrame) -> DataFrame:
        """
        Count the number of false reverse relations in the merged DataFrame.

        This function counts the number of false reverse relations in the merged DataFrame by comparing elements
        between semicolons in the 'rel_train' and 'rel_test' columns. It removes lines where '_rev' is found in either
        'rel_train' or 'rel_test' columns and filters out lines where these elements differ.

        Parameters:
        - merged_df (DataFrame): A DataFrame containing merged data with columns 'rel_train' and 'rel_test'.

        Returns:
        - int: The number of lines in the DataFrame after filtering, representing the count of false reverse relations.

        Note:
        - The function modifies the DataFrame by removing lines where '_rev' is found in 'rel_train' or 'rel_test'
          columns and comparing elements between semicolons in these columns.
        """
        # remove all lines where you can find _rev
        filtered_df = merged_df[~(merged_df['rel_train'].str.contains('_rev') | merged_df['rel_test'].str.contains('_rev'))]

        # Separate elements between semicolons in rel_train and rel_test
        filtered_df['rel_train_elements'] = filtered_df['rel_train'].str.split(';')
        filtered_df['rel_test_elements'] = filtered_df['rel_test'].str.split(';')
        # Compare elements between semicolons in rel_train and rel_test and filter lines
        filtered_df = filtered_df[
            filtered_df['rel_train_elements'].apply(lambda x: x[1]) != filtered_df['rel_test_elements'].apply(
                lambda x: x[1])]
        return filtered_df

    def calculate_reverse_relation_proportion(self, df_train: DataFrame,
                                      df_test: DataFrame, df_val: DataFrame = None) -> tuple[float, float, float]:
        """
        Calculate the proportion of reverse relations added and not added in the dataset.

        This function computes the proportion of reverse relations that were added and not added in the dataset.
        It merges the training and testing DataFrames, counts the added and not added reverse relations,
        and calculates the proportions.

        Parameters:
        - df_train (DataFrame): The training DataFrame.
        - df_test (DataFrame): The testing DataFrame.
        - df_val (DataFrame, optional): The validation DataFrame. Defaults to None.

        Returns:
        - Tuple[float, float]: A tuple containing the proportion of added and not added reverse relations, respectively.
        """
        if df_val is not None:
            full_train = concat([df_train, df_val], ignore_index=True)
        else:
            full_train = df_train
        # Merged dataframes on the columns 'from' and 'to'
        merged_df = merge(full_train, df_test, left_on=['from', 'to'], right_on=['to', 'from'], suffixes=('_train', '_test'))
        # saved the merged dataframe for controls
        merged_df.to_csv('benchmark/data/merged_reverse_train_test_test_3.csv', index=False, header=True)

        # here you got the real number of reverse relations added by the prepare_graph.py script
        nb_rev_rel_added = self.count_reverse_relations_added(merged_df)
        df_reverse_relation_not_added = self.reverse_relations_not_added(merged_df)
        nb_rev_not_added = len(df_reverse_relation_not_added)
        df_false_rev_rel = self.count_false_reverse_relation(merged_df)
        nb_false_rev_rel = len(df_false_rev_rel)

        # Calculer la proportion
        total_reverse_relations = len(full_train) + len(df_test)
        print(f"Total number of reverse relations: {total_reverse_relations}")
        proportion_rev_added = round((nb_rev_rel_added / len(df_test)) * 100, 2)
        proportion_rev_not_added = round((nb_rev_not_added / len(df_test)) * 100, 2)
        proportion_false_rev = round((nb_false_rev_rel / len(df_test)) * 100, 2)
        return proportion_rev_added, proportion_rev_not_added, proportion_false_rev

    def split_train_test_val(self, graph: DataFrame, test_size=0.2, random_state=None) -> (
            tuple)[DataFrame, DataFrame]:
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
        return train_set, test_set

    def check_train_test_independence(self, train_set: DataFrame, test_set: DataFrame):
        merged_df = merge(train_set, test_set, left_on=['from', 'to'], right_on=['from', 'to'], suffixes=('_test', '_test'))
        print(merged_df)
        return None

    def organize_col(self, df: DataFrame):
        return df[['from', 'rel', 'to']]


if __name__ == "__main__":
    prepare_kg = PrepareKG(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_CREATED_WITH_NEW_CLASS.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_CREATED_WITH_NEW_CLASS.txt")
    prepare_kg.generate_edgelist()
