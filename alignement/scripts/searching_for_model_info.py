import pykeen.nn
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import pandas as pd





model = torch.load("/home/thomas/Documents/projects/kge_project/alignement/data/fedcoder_prep/drug/model.pt")

# Assuming your model has entity and relation embeddings
entity_embeddings = model.entity_embeddings.weight.data
relation_embeddings = model.relation_embeddings.weight.data

# Number of entities and relations
num_entities = entity_embeddings.size(0)
num_relations = relation_embeddings.size(0)

print("Number of Entities:", num_entities)
print("Number of Relations:", num_relations)


