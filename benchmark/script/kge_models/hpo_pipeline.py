import argparse
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline


def main(args):
    result = hpo_pipeline(
        training=args.train_path,
        testing=args.test_path,
        validation=args.val_path,
        model=args.model,
        n_trials=args.n_trials,
    )
    result.save_to_directory(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyKEEN HPOPipeline Argument Parser")

    parser.add_argument("--train_path", type=str,
                        help="Path to the training set TSV file")
    parser.add_argument("--test_path", type=str,
                        help="Path to the testing set TSV file")
    parser.add_argument("--val_path", type=str,
                        help="Path to the validation set TSV file")
    parser.add_argument("--model", type=str,
                        help="Name of the model to use")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Output directory to save the results")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of trials to optimize the model")

    args = parser.parse_args()
    main(args)
