import csv

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import argparse
import torch
from pathlib import Path
from pykeen.sampling import PseudoTypedNegativeSampler

parser = argparse.ArgumentParser(description="PyKEEN Pipeline Argument Parser")
parser.add_argument("--input_folder", type=str, help="Path to train and test files")
parser.add_argument("--train", type=str, help="Path to the training set TSV file")
parser.add_argument("--test", type=str, help="Path to the testing set TSV file")
parser.add_argument("--output", type=str, help="Path to the output directory")
parser.add_argument("--model", type=str, help="Model to use in the pipeline for the embedding")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--emb_dim", type=int, help="Dimension of the embedding")

args = parser.parse_args()

output_path = Path(args.output)

input_path = Path(args.input_folder)

train = input_path.joinpath("train.csv")
test = input_path.joinpath("test.csv")


class ExtendedBasicNegativeSampler(PseudoTypedNegativeSampler):
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        # Extract head, relation, and tail from negatives
        negatives = super().corrupt_batch(positive_batch=positive_batch)
        neg_file = output_path.joinpath("negative_sample.txt")
        # Extract head, relation, and tail from negatives
        negative_triplets = []
        for neg in negatives:
            neg_list = neg.tolist()
            negative_triplets.append(neg_list)

        # Write negatives to CSV file
        with open(neg_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['head', 'relation', 'tail'])
            for triplet in negative_triplets:
                row_data = [triplet[0][0], triplet[0][1], triplet[0][2]]
                writer.writerow(row_data)
        return negatives


training = TriplesFactory.from_path(train)

testing = TriplesFactory.from_path(
    path=test,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id, )

testing.to_path_binary(output_path.joinpath("testing_triples"))

result = pipeline(
    training=training,
    testing=testing,
    model=args.model,
    random_seed=666,
    negative_sampler=ExtendedBasicNegativeSampler,
    training_kwargs=dict(num_epochs=args.epochs),
    negative_sampler_kwargs=dict(filtered=True),
    model_kwargs=dict(embedding_dim=args.emb_dim),
    evaluation_kwargs=dict(batch_size=100),

)

result.save_to_directory(args.output)
