# Generate edge attribute dictionary

import pickle as pkl
import pandas as pd
import argparse
from collections import Counter
import random
from pathlib import Path
import numpy as np
import sys
import igraph as ig

sys.path.insert(0, '../..') # add config to path
#import project_config


def clean_edges(df): # class1
    df = df.get(['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name', 'x_source', 'y_id', 'y_idx', 'y_type', 'y_name', 'y_source'])
    assert len(df[df.isna().any(axis=1)]) == 0
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.query('not ((x_id == y_id) and (x_idx == y_idx) and (x_type == y_type) and (x_source == y_source) and (x_name == y_name))')
    return df


def get_node_df(graph): # Assign nodes to IDs between 0 and N-1 #class1
    
    # Get all nodes
    nodes = pd.concat([graph.get(['x_id','x_type', 'x_name','x_source']).rename(columns={'x_id':'node_id', 'x_type':'node_type', 'x_name':'node_name','x_source':'node_source'}), graph.get(['y_id','y_type', 'y_name','y_source']).rename(columns={'y_id':'node_id', 'y_type':'node_type', 'y_name':'node_name','y_source':'node_source'})]).drop_duplicates(ignore_index=True)
    
    # Assign them to 0 to N-1
    nodes = nodes.reset_index().drop('index',axis=1).reset_index().rename(columns={'index':'node_idx'})
    print(nodes)    

    print("Finished assigning all nodes to IDs between 0 to N-1")
    return nodes


def reindex_edges(graph, nodes): # Assign node indices to nodes in edge
    
    # Map source nodes
    edges = pd.merge(graph, nodes, 'left', left_on=['x_id', 'x_type', 'x_name','x_source'], right_on=['node_id','node_type','node_name','node_source'])
    edges = edges.rename(columns={'node_idx':'x_idx'})
    
    # Map target nodes
    edges = pd.merge(edges, nodes, 'left', left_on=['y_id','y_type', 'y_name','y_source'], right_on=['node_id','node_type','node_name','node_source'])
    edges = edges.rename(columns={'node_idx':'y_idx'})
    
    # Subset only node info
    edges = edges.get(['x_idx', 'x_type', 'y_idx', 'y_type', 'relation', 'display_relation']).drop_duplicates(ignore_index=True).reset_index()
    print(edges)
    
    print("Finished updating edge list with new node IDs")
    return edges


def split_edges(edges): # Generate data splits
    split_idx = list(range(len(edges)))
    random.shuffle(split_idx)
    train_idx = split_idx[ : int(len(split_idx) * 0.8)]
    val_idx = split_idx[int(len(split_idx) * 0.8) : int(len(split_idx) * 0.9)]
    test_idx = split_idx[int(len(split_idx) * 0.9) : ]
    assert len(set(train_idx).intersection(set(val_idx), set(test_idx))) == 0
    
    mask = np.zeros(len(split_idx))
    mask[train_idx] = 0
    mask[val_idx] = 1
    mask[test_idx] = 2
    
    edges["mask"] = pd.Series(mask).map({0: "train", 1: "val", 2: "test"})
    #print(edges)
    print("Finished train/val/test split")
    return edges


def get_LCC(full_graph, nodes):
    edge_index = full_graph.get(['x_idx', 'y_idx']).values.T
    graph = ig.Graph()
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


def map_to_LCC(full_graph, giant, nodes, giant_nodes):
    new_nodes = nodes.query('node_idx in @giant_nodes')
    new_nodes = new_nodes.reset_index().drop('index',axis=1).reset_index().rename(columns={'index':'new_node_idx'})
    assert new_nodes.shape[0] == giant.vcount()
    assert len(new_nodes["node_idx"].to_list()) == len(new_nodes["new_node_idx"].to_list())

    new_edges = full_graph.query('x_idx in @giant_nodes and y_idx in @giant_nodes').copy()
    new_edges = new_edges.reset_index(drop=True)
    assert new_edges.shape[0] == giant.ecount()

    new_kg = pd.merge(new_edges, new_nodes, 'left', left_on='x_idx', right_on='node_idx')
    new_kg = new_kg.rename(columns={'node_id':'new_x_id', 'node_type':'new_x_type', 'node_name':'new_x_name', 'node_source':'new_x_source', 'new_node_idx':'new_x_idx'})
    new_kg = pd.merge(new_kg, new_nodes, 'left', left_on='y_idx', right_on='node_idx')
    new_kg = new_kg.rename(columns={'node_id':'new_y_id', 'node_type':'new_y_type', 'node_name':'new_y_name', 'node_source':'new_y_source', 'new_node_idx':'new_y_idx'})
    new_kg = new_kg[[c for c in new_kg.columns if "new" in c or "relation" in c]]
    new_kg = new_kg.rename(columns={k: k.split("new_")[1] for k in new_kg.columns if "new" in k})

    new_kg = clean_edges(new_kg)

    assert max(new_kg["x_idx"].to_list() + new_kg["y_idx"].to_list()) == giant.vcount() - 1
    assert len(set(new_kg['x_idx'].tolist() + new_kg['y_idx'].tolist())) == len(giant_nodes)
    return new_kg, new_nodes


def triadic_closure(graph):
    '''
    'disease_phenotype_positive' & 'disease_protein' -> 'phenotype_protein'
    '''
    print(f'Before triadic closure - Nodes: {len(pd.concat([graph["x_idx"], graph["y_idx"]]).unique())}')
    print(f'Before triadic closure - Edges: {len(graph["relation"].tolist())}')
    d_phen_relations = graph.loc[graph['relation'] == 'disease_phenotype_positive']
    d_prot_relations = graph.loc[graph['relation'] == 'disease_protein']
    merged_relations = d_phen_relations.set_index(['x_id', 'x_idx', 'x_name', 'x_source', 'x_type']) \
        .join(d_prot_relations.set_index(['x_id', 'x_idx', 'x_name', 'x_source', 'x_type']), how='inner', rsuffix='_dp')
    new_relations = merged_relations.reset_index(drop=True)
    new_relations['relation'] = 'phenotype_protein'
    new_relations['display_relation'] = 'associated with'
    new_relations.drop(columns=['relation_dp', 'display_relation_dp'], inplace=True)
    new_relations.rename(columns={'y_id':'x_id', 'y_idx':'x_idx', 'y_type':'x_type', 'y_name':'x_name', 'y_source':'x_source'}, inplace=True)
    new_relations.rename(columns={'y_id_dp':'y_id', 'y_idx_dp':'y_idx', 'y_type_dp':'y_type', 'y_name_dp':'y_name','y_source_dp':'y_source' }, inplace=True)
    triadic_closure_graph = pd.concat([graph, new_relations], ignore_index=True)
    print(f'After triadic closure, pre-dedup - Edges: {len(triadic_closure_graph["relation"].tolist())}')

    triadic_closure_graph = clean_edges(triadic_closure_graph)

    print('cleaned graph\n', triadic_closure_graph.head())
    print(f'After triadic closure - Nodes: {len(pd.concat([triadic_closure_graph["x_idx"], triadic_closure_graph["y_idx"]]).unique())}')
    print(f'After triadic closure - Edges: {len(triadic_closure_graph["relation"].tolist())}')
    return triadic_closure_graph


def find_reciprocal_relationships(graph: pd.DataFrame) -> pd.DataFrame:
    """
    Find true reverse relationships in a graph DataFrame.

    This function takes a DataFrame representing a graph with relationships between nodes
    and finds true reverse relationships, where the relationship between two nodes A and B
    is reciprocal, meaning that if A is related to B with a certain relation, then B is also
    related to A with the same relation.

    Parameters:
    - graph (pd.DataFrame): A DataFrame representing the graph containing relationships
                            between nodes. It should have columns 'x_idx', 'y_idx', and 'relation'.

    Returns:
    - pd.DataFrame: A DataFrame containing the true reverse relationships found in the input graph.
                    It includes columns such as 'relation', 'display_relation', 'x_id', 'x_idx',
                    'x_type', 'x_name', 'x_source', 'y_id', 'y_idx', 'y_type', 'y_name', and 'y_source'.

    Note:
    - The function assumes that the input DataFrame 'graph' contains columns 'x_idx', 'y_idx',
      and 'relation' representing the indices of connected nodes and the relationship between them.
    - It performs a merge operation on the DataFrame to identify reciprocal relationships,
      removes duplicates, and restructures the DataFrame for clarity.
    """
    # Merge the graph with itself on 'x_idx' and 'y_idx'
    merged = pd.merge(graph, graph, left_on=['x_idx', 'y_idx', 'relation'], right_on=['y_idx', 'x_idx', 'relation'])
    # Keep only the rows where 'x_idx_x' equals 'y_idx_y' and 'y_idx_x' equals 'x_idx_y'
    true_reverse = merged[(merged['x_idx_x'] == merged['y_idx_y']) & (merged['y_idx_x'] == merged['x_idx_y'])]
    # Split the DataFrame into two based on the column names
    df_relation = true_reverse["relation"]
    df_x = true_reverse[[col for col in true_reverse.columns if col.endswith('_x')]]
    df_y = true_reverse[[col for col in true_reverse.columns if col.endswith('_y')]]
    # Rename the columns to remove the suffixes for concatenation
    df_x.columns = df_x.columns.str.rstrip('_x')
    df_y.columns = df_y.columns.str.rstrip('_y')
    df_x = pd.concat([df_relation, df_x], axis=1)
    df_y = pd.concat([df_relation, df_y], axis=1)
    df_x.columns = ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name',
                    'x_source', 'y_id', 'y_idx', 'y_type', 'y_name', 'y_source']
    df_y.columns = ['relation', 'display_relation', 'x_id', 'x_idx', 'x_type', 'x_name',
                    'x_source', 'y_id', 'y_idx', 'y_type', 'y_name', 'y_source']
    # Concatenate and reset the index
    result = pd.concat((df_x, df_y), ignore_index=True).drop_duplicates().reset_index(drop=True)
    return result


def add_reverse_edges(graph: pd.DataFrame) -> pd.DataFrame:
    """
    Adds reverse edges to a graph DataFrame.

    This function takes a DataFrame representing a graph and adds reverse edges to it. Reverse edges
    are created by identifying reciprocal relationships and appending corresponding reverse relations to
    the graph. Reverse relations are distinguished by adding a "_rev" suffix to the original relation.

    Parameters:
    - graph (pd.DataFrame): A DataFrame representing the graph containing relationships between nodes.
                            It should have columns 'x_idx', 'x_type', 'y_idx', 'y_type', and 'relation'.

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
    true_reverse = find_reciprocal_relationships(graph)

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
    rev_edges = pd.concat((rev_edge_eqtype, rev_edge_neqtype)).drop_duplicates(ignore_index=True).reset_index()

    full_graph = pd.concat((graph[["x_idx", "x_type", "y_idx", "y_type", "relation"]], rev_edges[["x_idx", "x_type", "y_idx", "y_type", "relation"]]))#.drop_duplicates(ignore_index=True).reset_index()


    full_graph = full_graph.drop_duplicates(ignore_index=True).reset_index()

    full_graph["full_relation"] = full_graph["x_type"] + ";" + full_graph["relation"] + ";" + full_graph["y_type"]

    return full_graph


def generate_edgelist(node_map_f, mask_f, graph, triad_closure):
    
    print("Starting to process the KG table")
    nodes = get_node_df(graph)
    edges = reindex_edges(graph, nodes)

    print("Starting to generate the connected KG")
    giant = get_LCC(edges, nodes)
    giant_nodes = giant.vs['name']
    new_kg, new_nodes = map_to_LCC(edges, giant, nodes, giant_nodes) # ici elles clean les duplicats

    if triad_closure:
        print('Performing triadic closure on P-G-D relationships.')
        new_kg = triadic_closure(new_kg)

    new_kg.to_csv("benchmark/output/new_kg.csv", sep="\t", index=False)
    full_graph = new_kg
    print(full_graph)
############################################################# WE CAN TRY TO REMOVE THIS LINE
    print("Starting to get reverse edges")
    full_graph = add_reverse_edges(full_graph) # en faisant ca elles rajoutent de nouveaux des duplicats
#############################################################
    print("Starting to save final dataframes")
    new_nodes = new_nodes.get(["new_node_idx", "node_id", "node_type", "node_name", "node_source"]).rename(columns={"new_node_idx": "node_idx"})
    new_nodes.to_csv(node_map_f, sep="\t", index=False)
    full_graph = full_graph[["x_idx", "y_idx", "full_relation"]]
    full_graph = full_graph.drop_duplicates(ignore_index=True).reset_index(drop=True)

    full_graph = split_edges(full_graph)
    full_graph.to_csv(mask_f, sep="\t", index=False)

    print("Final Number of Edges:", len(full_graph["full_relation"].tolist()))
    for k, v in Counter(full_graph["full_relation"].tolist()).items():
        print(k, v)


'''
python prepare_graph.py \
--triad_closure
'''

def main():
    parser = argparse.ArgumentParser(description="Prepare graph.")
    parser.add_argument('--triad_closure', action='store_true', \
        help='Whether to add edges between phenotypes & genes if edges exist between P-D and D-G')
    args = parser.parse_args()


    graph = pd.read_csv('benchmark/data/kg_giant_orphanet.csv', dtype={"x_id": str, "y_id": str})


    # filter_list = ["contraindication", "drug_drug", "side_effect", "drug_targets", "drug_protein", "drug_effect", "indication", "off-label use", "exposure_protein", "exposure_molfunc", "exposure_cellcomp", "exposure_bioprocess", "exposure_disease", "exposure_exposure", "anatomy_protein_present", "anatomy_protein_absent", "anatomy_anatomy", "protein_present_anatomy", "protein_absent_anatomy"]
    # graph = graph.loc[~graph['relation'].isin(filter_list)]
    # print(graph)

    graph = graph[graph["x_name"] != "missing"]
    graph = graph[graph["y_name"] != "missing"]
    print(graph)

    # Output
    node_map_f =  f"benchmark/data/KG_node_map_cor.txt"
    mask_f = f"benchmark/data/KG_edgelist_mask_cor.txt"
    generate_edgelist(node_map_f, mask_f, graph, triad_closure=args.triad_closure)


if __name__ == "__main__":
    main()
