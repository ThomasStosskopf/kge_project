import pykeen.nn
from typing import List
from pykeen.pipeline import pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pykeen.triples import TriplesFactory
import torch
import pandas as pd



model = torch.load("benchmark/output/output_first_method_test/DistMult_output_vrai50epochs/trained_model.pkl")

entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
relation_representation_modules: List['pykeen.nn.Representation'] = model.relation_representations

entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

print(entity_embeddings)

entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
relation_embedding_tensor: torch.FloatTensor = relation_embeddings()



entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=None)
relation_embedding_tensor: torch.FloatTensor = relation_embeddings(indices=None)

# ca prend l'embedding et ca le convertie en numpy array
entity_embedding = model.entity_representations[0](indices=None).detach().numpy()
plt.figure(figsize=(6, 6))
pca = PCA(n_components=2)
m = pca.fit(entity_embedding)

eu = m.transform(entity_embedding)
df = pd.read_csv("benchmark/output/output_first_method_test/DistMult_output_vrai50epochs/training_triples/entity_to_id.tsv.gz", sep="\t")
type_to_entities = pd.read_csv("benchmark/data/first_method/type_to_entities_first.csv", sep="\t")
x_df = type_to_entities[["x_name", "x_type"]].copy()

y_df = type_to_entities[["y_name", "y_type"]].copy()
x_df.rename(columns={"x_name": "label", "x_type": "type"}, inplace=True)
y_df.rename(columns={"y_name": "label", "y_type": "type"}, inplace=True)

entities_to_type = pd.concat([x_df, y_df], axis=0)
df_merged = pd.merge(df, entities_to_type, on=["label"], how='left').drop_duplicates(ignore_index=True).reset_index()
# Utilisation de la méthode value_counts() pour compter le nombre d'occurrences de chaque valeur
counts = df_merged["id"].value_counts()

# Sélection des valeurs qui ont un décompte supérieur à 1 (valeurs non uniques)
valeurs_non_uniques = counts[counts > 1].index.tolist()


df_merged.to_csv("benchmark/data/OUPSJECOMPRENDSPAS.csv", sep="\t", index=False)
#df_merged = df_merged.drop_duplicates(ignore_index=True).reset_index()


df_to_plot = df_merged[["id", "type"]].copy()


types = df_to_plot["type"].unique()
colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "brown", "lime"]
type_color_dict = dict(zip(types, colors))
print(type_color_dict)


def df_to_dict(graph: pd.DataFrame, col1, col2) -> dict:
    return graph.set_index(col1)[col2].to_dict()


dict_entity = df_to_dict(graph=df_to_plot, col1="id", col2="type")


#Define figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting
for i, entity in enumerate(dict_entity):
    ax.annotate(
        text="o",
        xy=(eu[i, 0], eu[i, 1]),
        color=type_color_dict[dict_entity[entity]],  # Utilisation de la couleur associée à chaque type
        ha="center", va="center",
        fontsize=4,  # Adjust the font size
        fontweight='bold',  # Make text bold
        bbox=dict(boxstyle="round,pad=0.3", fc=type_color_dict[dict_entity[entity]], ec=type_color_dict[dict_entity[entity]], lw=1),  # Add a rounded box around text
    )

# Add legend
legend_handles = []
for type_, color in type_color_dict.items():
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=type_))

ax.legend(handles=legend_handles, loc='upper right')

# Customize plot aesthetics
ax.set_aspect('equal', 'box')  # Ensure aspect ratio is equal
ax.grid(True, linestyle='--', alpha=0.6)  # Add grid lines
ax.set_title('Embedding representation', fontsize=14)  # Title of the plot

# Remove the axis spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the plot
plt.savefig("benchmark/output/distmult_50epochs_embedding.png")

# Show plot
plt.tight_layout()
plt.show()






#
# relation_embeddings = model.relation_representations[0](indices=None).detach().numpy()
# ru = pca.transform(relation_embeddings)
# for i, relation in enumerate(triples_factory.relation_id_to_label):
#     plt.annotate(
#         text=id_to_relation[relation],
#         xy=(0, 0), xytext=(ru[i, 0], ru[i, 1]),
#         arrowprops=dict(
#             arrowstyle="<-",
#             color="tab:red",
#             shrinkA=5,
#             shrinkB=5,
#             patchA=None,
#             patchB=None,
#             connectionstyle="arc3,rad=0."
#         )
#     )
#
# plt.xlim([-2, 2])
# plt.ylim([-1, 1])
