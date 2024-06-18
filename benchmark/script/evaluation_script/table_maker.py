from pathlib import Path
from metrics_finder import MetricsFinder
from pandas import DataFrame
import argparse

class TableMaker:
    """
    A class to create tables from evaluation metrics obtained from multiple models.

    Attributes:
        path_to_models (Path): The path to the directory containing model directories.
    """

    def __init__(self, path_to_models):
        """
        Initialize the TableMaker object.

        Args:
            path_to_models (str): The path to the directory containing model directories.
        """
        self.path_to_models = Path(path_to_models)

    def get_name_of_data_preparation_method(self):
        """
        Extract the name of the data preparation method from the path to the models directory.

        Returns:
            str: Name of the data preparation method.
        """
        path = str(self.path_to_models)
        return path.split('/')[-1]

    def iter_in_folder(self) -> list:
        """
        Iterate through the directories in the models directory and identify valid model directories.

        Returns:
            list: List of paths to valid model directories.
        """
        list_path_models = []
        for element in self.path_to_models.iterdir():
            if element.is_dir():
                list_down_elem = []
                for down_elem in element.iterdir():
                    list_down_elem.append(down_elem.name)
                if all(item in list_down_elem for item in ["trained_model.pkl", "testing_triples", "training_triples"]):
                    list_path_models.append(str(element.resolve()))
        return list_path_models

    def find_names_folder(self, path: str) -> str:
        """
        Extract the name of a folder from its path.

        Args:
            path (str): The path of the folder.

        Returns:
            str: The name of the folder.
        """
        name = path.split('/')
        return name[-1]

    def find_metrics(self, list_of_paths: list) -> dict:
        """
        Find evaluation metrics for models located at specified paths.

        Args:
            list_of_paths (list): List of paths to model directories.

        Returns:
            dict: Dictionary containing evaluation metrics for each model.
        """
        dict_all_metrics = {}
        for path in list_of_paths:
            metrics_finder = MetricsFinder(path)
            dict_rel_score = metrics_finder.create_dict_of_evaluation(metrics_finder.relation_to_id,
                                                                      metrics_finder.get_list_of_id_in_mapped_testing_triples())
            name = self.find_names_folder(path=path)
            print(f"NAME {name}")
            dict_all_metrics[name] = dict_rel_score
        return dict_all_metrics

    def create_df(self, dict_metrics: dict) -> DataFrame:
        """
        Create a pandas DataFrame from the evaluation metrics.

        Args:
            dict_metrics (dict): Dictionary containing evaluation metrics for each model.

        Returns:
            DataFrame: Pandas DataFrame containing the evaluation metrics.
        """
        records = []
        for key, value in dict_metrics.items():
            for subkey, subvalue in value.items():
                record = {'Model': key, 'Rel': subkey}
                record.update(subvalue)
                records.append(record)
        df = DataFrame(records)
        return df

    def make_specific_table(self, df_all_data: DataFrame, metric: str) -> DataFrame:
        """
        Create a specific table for a given metric from the DataFrame.

        Args:
            df_all_data (DataFrame): Pandas DataFrame containing all evaluation metrics.
            metric (str): The metric for which the table is to be created.

        Returns:
            DataFrame: Specific table for the given metric.
        """
        pivot_df = df_all_data.pivot_table(index='Rel', columns='Model', values=metric, aggfunc='first')
        pivot_df = pivot_df.reset_index()
        return pivot_df

    def get_list_metrics_in_df(self, df_all_data: DataFrame) -> list:
        """
        Get a list of metrics present in the DataFrame.

        Args:
            df_all_data (DataFrame): Pandas DataFrame containing all evaluation metrics.

        Returns:
            list: List of metrics present in the DataFrame.
        """
        columns = df_all_data.columns.tolist()
        columns.remove('Model')
        columns.remove('Rel')
        return columns

    def make_dict_df_for_each_metric(self, df_all_data: DataFrame) -> dict:
        """
        Create a dictionary of DataFrames, each for a specific metric.

        Args:
            df_all_data (DataFrame): Pandas DataFrame containing all evaluation metrics.

        Returns:
            dict: Dictionary containing DataFrames for each metric.
        """
        dict_df = {}
        list_of_metrics = self.get_list_metrics_in_df(df_all_data)
        for metric in list_of_metrics:
            dict_df[metric] = self.make_specific_table(df_all_data=df_all_data, metric=metric)
        return dict_df

    def save_df_from_dict(self, dict_of_df: dict) -> None:
        """
        Save DataFrames to files.

        Args:
            dict_of_df (dict): Dictionary containing DataFrames to be saved.
        """
        path_to_save = self.path_to_models.joinpath("eval")
        if not path_to_save.exists():
            path_to_save.mkdir()
        for metric, dataframe in dict_of_df.items():
            dataframe.to_csv(path_to_save.joinpath(f"{metric}.tsv"), sep='\t')

    def main(self):
        folder_list = self.iter_in_folder()
        print(folder_list)
        dict_all_metrics = self.find_metrics(folder_list)
        df_all_metrics = self.create_df(dict_all_metrics)
        dict_df = self.make_dict_df_for_each_metric(df_all_data=df_all_metrics)
        self.save_df_from_dict(dict_of_df=dict_df)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create multiple tables about the differents metrics to evaliuate "
                                                 "graph embedding")
    parser.add_argument("--input_path", type=str, help="path to the folder where the different trained models are.")
    args = parser.parse_args()

    table_maker = TableMaker(args.input_path)
    table_maker.main()
