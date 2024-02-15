from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="PyKEEN Pipeline Argument Parser")

parser.add_argument("--train", type=str, help="Path to the training set TSV file")
parser.add_argument("--test", type=str, help="Path to the testing set TSV file")
parser.add_argument("--output", type=str, help="Path to the output directory")
parser.add_argument("--model", type=str, help="Model to use in the pipeline for the embedding")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="batch size")
args = parser.parse_args()

path = Path(args.output)

training = TriplesFactory.from_path(args.train)

testing = TriplesFactory.from_path(
    path=args.test,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,)

testing.to_path_binary(path.joinpath("testing_triples"))

result = pipeline(
    training=training,
    testing=testing,
    model=args.model,
    epochs=args.epochs,
    random_seed=666,
    negative_sampler="PseudoTypedNegativeSampler",
    training_kwargs=dict(batch_size=args.batch_size)
)

result.save_to_directory(args.output)


