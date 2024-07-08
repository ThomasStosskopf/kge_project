import pykeen
print(pykeen.get_version())


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
parser.add_argument("--emb_size", type=int, help="Embedding dimension")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

args = parser.parse_args()

path = Path(args.output)
final_output = Path(path.joinpath(f"{args.model}_{args.epochs}epochs_{args.emb_size}emb_{args.batch_size}batchSize_{args.learning_rate}lr"))


training = TriplesFactory.from_path(args.train)

testing = TriplesFactory.from_path(
    path=args.test,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,)


training.to_path_binary(final_output.joinpath("training_triples"))
testing.to_path_binary(final_output.joinpath("testing_triples"))

result = pipeline(
    training=training,
    testing=testing,
    model=args.model,
    epochs=args.epochs,
    random_seed=666,
    negative_sampler="PseudoTypedNegativeSampler",
    training_kwargs=dict(
        num_epochs=args.epochs,
        checkpoint_name='my_checkpoint.pt',
        checkpoint_directory=final_output.joinpath('doctests/checkpoint_dir'),
        checkpoint_frequency=60,),
    negative_sampler_kwargs=dict(filtered=True),
    model_kwargs=dict(embedding_dim=args.emb_size),
    optimizer_kwargs=dict(lr=args.learning_rate),


)



result.save_to_directory(final_output)

