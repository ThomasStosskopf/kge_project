from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
def split_train_test_val(graph: DataFrame, test_size=0.1, random_state=None) -> (
        tuple)[DataFrame, DataFrame]:

    train_set, test_set = train_test_split(graph, test_size=test_size, random_state=random_state)
    return train_set, test_set


if __name__=="__main__":

    parser = ArgumentParser(description="Split a dataset into train and val set")
    parser.add_argument("--input_file", type=str, help="path to the csv file to split.")
    parser.add_argument("--output_train", type=str, help="path of the output train.")
    parser.add_argument("--output_val", type=str, help="path of the output val.")
    args = parser.parse_args()
    input_path=args.input_file
    output_train = args.output_train
    output_val = args.output_val

    df = read_csv(input_path, sep="\t")

    train, val = split_train_test_val(graph=df)

    train.to_csv(output_train, sep="\t", header=False, index=False)
    val.to_csv(output_val, sep="\t", header=False, index=False)
