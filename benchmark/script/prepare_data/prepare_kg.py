
from pandas import concat, merge, read_csv, DataFrame
from igraph import Graph
from collections import Counter


class PrepareKG:

    def __init__(self, kg_path, output_nodes_map, output_kg_edge_list):
        self.kg = self.read_graph(kg_path)
        self.output_nodes_map = output_nodes_map
        self.output_kg_edge_list = output_kg_edge_list

    def read_graph(self, path):
        graph = read_csv(path, dtype={"x_id": str, "y_id": str})
        graph = graph[graph["x_name"] != "missing"]
        graph = graph[graph["y_name"] != "missing"]
        return graph

    def clean_edges(self, df):  # class1
        df = df.get(
            ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name', 'x_source', 'y_id', 'y_idx', 'y_type',
             'y_name', 'y_source'])
        assert len(df[df.isna().any(axis=1)]) == 0
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.query(
            'not ((x_id == y_id) and (x_idx == y_idx) and (x_type == y_type) and (x_source == y_source) and (x_name == y_name))')
        return df

    def get_node_df(self, graph):  # Assign nodes to IDs between 0 and N-1 #class1

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
        return full_graph

    def generate_edgelist(self):
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

        new_kg.to_csv("benchmark/output/new_kg.csv", sep="\t", index=False)
        full_graph = new_kg
        print(full_graph)
        ############################################################# WE CAN TRY TO REMOVE THIS LINE
        print("Starting to get reverse edges")
        #full_graph = add_reverse_edges(full_graph)  # en faisant ca elles rajoutent de nouveaux des duplicats
        #############################################################

        full_graph = self.expand_graph_relations(full_graph)

        print("Starting to save final dataframes")
        new_nodes = new_nodes.get(["new_node_idx", "node_id", "node_type", "node_name", "node_source"]).rename(
            columns={"new_node_idx": "node_idx"})
        new_nodes.to_csv(node_map_f, sep="\t", index=False)
        full_graph = full_graph[["x_idx", "y_idx", "full_relation"]]
        full_graph = full_graph.drop_duplicates(ignore_index=True).reset_index(drop=True)

        #full_graph = self.split_edges(full_graph)
        full_graph.to_csv(mask_f, sep="\t", index=False)

        print("Final Number of Edges:", len(full_graph["full_relation"].tolist()))
        for k, v in Counter(full_graph["full_relation"].tolist()).items():
            print(k, v)


if __name__ == "__main__":
    prepare_kg = PrepareKG(kg_path='benchmark/data/kg_giant_orphanet.csv',
                           output_nodes_map="benchmark/data/KG_node_map_CREATED_WITH_NEW_CLASS.txt",
                           output_kg_edge_list="benchmark/data/KG_edgelist_mask_CREATED_WITH_NEW_CLASS.txt")
    prepare_kg.generate_edgelist()