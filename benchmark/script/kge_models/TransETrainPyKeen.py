
from pykeen.pipeline import pipeline


import pandas as pd

def convert_csv_to_tsv(csv_file_path, tsv_file_path):
    df = pd.read_csv(csv_file_path, sep=',')
    df.to_csv(tsv_file_path, sep='\t', index=False)


# Convert your files
convert_csv_to_tsv("benchmark/data/train_set.csv", "benchmark/data/train_set.tsv")
convert_csv_to_tsv("benchmark/data/test_set.csv", "benchmark/data/test_set.tsv")
convert_csv_to_tsv("benchmark/data/val_set.csv", "benchmark/data/val_set.tsv")


result = pipeline(

    training="benchmark/data/train_set.tsv",

    testing="benchmark/data/test_set.tsv",

    validation="benchmark/data/val_set.tsv",

    model='TransE',

    epochs=5,  # short epochs for testing - you should go higher


)

result.save_to_directory('./')