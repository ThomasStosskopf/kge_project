from pandas import read_csv, concat
from torchkge.data_structures import KnowledgeGraph

def load_data():
    """
    Load CSV files into DataFrames and create a KnowledgeGraph.

    Returns:
    - Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    A tuple containing three KnowledgeGraph instances for training, validation, and test sets.
    """
    # Load CSV file into a DataFrame

    df1 = read_csv('benchmark/data/train_set.csv',
                   sep=',', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv('benchmark/data/val_set.csv',
                   sep=',', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv('benchmark/data/test_set.csv',
                   sep=',', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))