import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from pandas import DataFrame, read_csv, concat, merge


class EmbeddingVisualizer:

    def __init__(self, model_path, entities_to_id, entities_to_type, save_path):
        self.model = torch.load(model_path)
        self.entities_to_id = read_csv(entities_to_id, sep="\t")
        self.entities_to_type = read_csv(entities_to_type, sep="\t")
        self.entity_embedding = self.load_embedding_as_numpy_array()
        self.save_path = save_path


    def load_embedding_as_numpy_array(self):
        return self.model.entity_representations[0](indices=None).detach().numpy()

    def prepare_data_with_PCA(self, n_comp=2):
        pca = PCA(n_components=n_comp)
        pca_model = pca.fit(self.entity_embedding)
        entity_embedding_pca = pca_model.transform(self.entity_embedding)
        return entity_embedding_pca

    def prepare_type_entity_df(self):
        x_df = self.entities_to_type [["x_name", "x_type"]].copy()
        y_df = self.entities_to_type [["y_name", "y_type"]].copy()
        x_df.rename(columns={"x_name": "label", "x_type": "type"}, inplace=True)
        y_df.rename(columns={"y_name": "label", "y_type": "type"}, inplace=True)
        self.entities_to_type = concat([x_df, y_df], axis=0)

    def map_id_to_type(self) -> DataFrame:
        df_merged = merge(self.entities_to_id, self.entities_to_type, on=["label"], how='left').drop_duplicates(
            ignore_index=True).reset_index()
        return df_merged[["id", "type"]].copy()

    def attribute_color_to_type(self, df_to_plot: DataFrame) -> dict:
        types = df_to_plot["type"].unique()
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "brown", "lime"]
        return  dict(zip(types, colors))

    def df_to_dict(self, graph: DataFrame, col1: str, col2: str) -> dict:
        return graph.set_index(col1)[col2].to_dict()

    def plot_fig_embedding(self, dict_id_type: dict, type_to_color: dict, entity_embedding_pca):
        # Define figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plotting
        for i, entity in enumerate(dict_id_type):
            ax.annotate(
                text="o",
                xy=(entity_embedding_pca[i, 0], entity_embedding_pca[i, 1]),
                color=type_to_color[dict_id_type[entity]],  # Utilisation de la couleur associée à chaque type
                ha="center", va="center",
                fontsize=4,  # Adjust the font size
                fontweight='bold',  # Make text bold
                bbox=dict(boxstyle="round,pad=0.3", fc=type_to_color[dict_id_type[entity]],
                          ec=type_to_color[dict_id_type[entity]], lw=1),  # Add a rounded box around text
            )

        # Add legend
        legend_handles = []
        for type_, color in type_to_color.items():
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=type_))

        ax.legend(handles=legend_handles, loc='upper right')

        # Customize plot aesthetics
        ax.set_aspect('equal', 'box')  # Ensure aspect ratio is equal
        ax.grid(True, linestyle='--', alpha=0.6)  # Add grid lines
        ax.set_title('Embedding representation', fontsize=14)  # Title of the plot

        # Remove the axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save the plot
        plt.savefig(self.save_path)

        # Show plot
        plt.tight_layout()
        plt.show()

    def main(self):
        entity_embedding_pca = self.prepare_data_with_PCA()
        self.prepare_type_entity_df()
        df_to_plot = self.map_id_to_type()
        dict_type_color = self.attribute_color_to_type(df_to_plot)
        dict_id_type = self.df_to_dict(graph=df_to_plot, col1="id", col2="type")
        self.plot_fig_embedding(dict_id_type=dict_id_type, type_to_color=dict_type_color, entity_embedding_pca=entity_embedding_pca)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Script to visualize enbedding")

    parser.add_argument("--model", type=str, help="Path to the trained model.")
    parser.add_argument("--mapping", type=str, help="Path to the file mapping the entities and their id.")
    parser.add_argument("--type", type=str, help="Path to the file mapping entities to their type.")
    parser.add_argument("--output", type=str, help="Path the file where the image will be saved.")

    args = parser.parse_args()

    visualizer = EmbeddingVisualizer(model_path=args.model,
                                     entities_to_id=args.mapping,
                                     entities_to_type=args.type,
                                     save_path=args.output)
    visualizer.main()