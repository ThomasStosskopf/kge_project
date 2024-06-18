from pandas import DataFrame, read_csv, concat
import argparse
from pathlib import Path
import shutil
import os

class RelationSelector:

    def __init__(self, file_path, output, id_rel):
        self.path = Path(file_path)
        self.test_df = read_csv(self.path.joinpath("numeric_triples.tsv.gz"), sep="\t")
        self.output = Path(output)
        self.id_rel = id_rel
        self.filtered_df = self.filter_df()

    def get_test_file(self):
        return self.test_df

    def get_filter_df(self):
        return self.filtered_df

    def filter_df(self):
        # Filter the DataFrame
        filtered_df = self.test_df[self.test_df['relation'] == self.id_rel]
        return filtered_df

    def save_df(self):
        # Construct the output file path
        output_file_path = self.output / "numeric_triples.tsv.gz"
        # Save the filtered DataFrame to a gzip-compressed TSV file
        self.filtered_df.to_csv(output_file_path, sep="\t", index=False, header=False, compression='gzip')

    def copy_files_to_output(self):

        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.save_df()
        # Browse all files in source folder
        for file in os.listdir(self.path):
            path_source_folder = os.path.join(self.path, file)

            if os.path.isfile(path_source_folder) and file != 'numeric_triples.tsv.gz':

                shutil.copy(path_source_folder, self.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a relation to keep in test file")

    parser.add_argument("--test_file", type=str, help="Path to the test file")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--rel_id", type=int, help="relation id to evaluate on")
    args = parser.parse_args()
    path = args.test_file
    output = args.output
    rel_id = args.rel_id

    relationSelector = RelationSelector(file_path=path, output=output, id_rel=rel_id)

    print(relationSelector.get_test_file())
    print(relationSelector.get_filter_df())

    relationSelector.copy_files_to_output()


