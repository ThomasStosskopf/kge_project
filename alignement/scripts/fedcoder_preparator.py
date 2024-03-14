# imports
import argparse
import autoalign
import numpy as np
import os
import torch
from typing import List
from pathlib import Path
import pykeen.nn
from pandas import read_csv

class FedcoderPreparer:

    def __init__(self, input_path: str, output_path: str):
        self.path = Path(input_path)
        self.output_path = output_path
        self.model = torch.load(self.path.joinpath("trained_model.pkl"))
        self.entities_tensor = self.make_entities_tensor()
        self.relations_tensor = self.make_relations_tensor()
        self.entities_ids = read_csv(self.path.joinpath("training_triples/entity_to_id.tsv.gz"), sep="\t")
        self.relations_ids = read_csv(self.path.joinpath("training_triples/relation_to_id.tsv.gz"), sep="\t")

    def get_model(self):
        return self.model

    def get_entities_tensor(self):
        return self.entities_tensor

    def get_relations_tensor(self):
        return self.relations_tensor

    def get_entities_ids(self):
        return self.entities_ids
    def make_entities_tensor(self):
        entity_representation_modules: List['pykeen.nn.Representation'] = self.model.entity_representations
        entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
        entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=None)
        return entity_embedding_tensor

    def make_relations_tensor(self):
        relation_representation_modules: List['pykeen.nn.Representation'] = self.model.relation_representations
        relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]
        relation_embeddings_tensor: torch.FloatTensor = relation_embeddings(indices=None)
        return relation_embeddings_tensor

    def create_model(self, pretrained_ents: torch.Tensor, pretrained_rels: torch.Tensor
                     ) -> autoalign.embedder.TripleEmbedder:
        """Create a autoalign TripleEmbedder model from pre-trained embeddings."""
        # initialize embeddings with the dimensions from
        # the pre-trained embedding matrices
        ents = autoalign.lookup.Embedding(
            pretrained_ents.shape[0],
            pretrained_ents.shape[1],
        )
        rels = autoalign.lookup.Embedding(
            pretrained_rels.shape[0],
            pretrained_rels.shape[1],
        )

        # copy the pre-trained embedding matrices
        ents._embedding.weight.data = pretrained_ents
        rels._embedding.weight.data = pretrained_rels

        # combine embeddings in a triple embedder
        model = autoalign.embedder.TripleEmbedder(
            head_embedding=ents,
            pred_embedding=rels,
            tail_embedding=ents,
        )

        # return the embedding model
        return model

    def main(self):
        # ensure output directory
        os.makedirs(self.output_path, exist_ok=True)
        print(f"folder {self.output_path} created")
        model = self.create_model(self.entities_tensor, self.relations_tensor)

        # save pre-trained embeddings
        np.save(os.path.join(self.output_path, "PRETRAINED_ENTS.npy"), self.entities_tensor.detach().numpy())
        np.save(os.path.join(self.output_path, "PRETRAINED_RELS.npy"), self.relations_tensor.detach().numpy())

        # save ID mapping for entities and relations
        self.entities_ids.to_csv(os.path.join(self.output_path, "entity_ids.del"), index=False, header=False, sep="\t")
        self.relations_ids.to_csv(os.path.join(self.output_path, "relation_ids.del"), index=False, header=False, sep="\t")
        self.entities_ids["id"].to_csv(os.path.join(self.output_path, "vocab.json"), index=False, header=False)

        # save the triple embedder state
        torch.save(model.state_dict(), os.path.join(self.output_path, "model.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process data so it can be used by Fedcoder for graph alignment")
    parser.add_argument("--input", type=str, help="Path to the trained model folder")
    parser.add_argument("--output", type=str, help="Path to the output folder")

    args = parser.parse_args()

    # input_path = "/home/thomas/Documents/projects/kge_project/alignement/output/drug_kge" # exemple
    # output_path = "/home/thomas/Documents/projects/kge_project/fedcoder/examples/alignement1/drug_align" # exemple

    preparatory = FedcoderPreparer(input_path=args.input, output_path=args.output)

    preparatory.main()


