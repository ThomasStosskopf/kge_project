from pandas import DataFrame, read_csv, concat
import random
from sklearn.model_selection import train_test_split
from pathlib import Path

class RandomKG:

    def __init__(self, kg_path, output_path):
        self.path = Path(output_path)
        self.kg = read_csv(kg_path, sep=",", low_memory=False)
        self.output_train = self.path.joinpath("train.csv")
        self.output_test = self.path.joinpath("test.csv")
        self.output_kg = self.path.joinpath("kg.csv")

    def random_number(self, lower_nb: int, upper_nb: int) -> int:
        return random.randint(lower_nb, upper_nb)

    def generate_random_graph(self) -> DataFrame:
        new_kg = []
        list_nb = []
        graph = self.kg
        for i in range(100000):
            index_to_copy = self.random_number(lower_nb=0, upper_nb=len(graph) - 2)
            list_nb.append(index_to_copy)
            row_to_copy = graph.iloc[index_to_copy:index_to_copy + 1]
            new_kg.append(row_to_copy)
        print(new_kg[0])
        print(list_nb[0])
        return concat(new_kg, ignore_index=True)

    def split_train_test_val(self, graph: DataFrame, test_size=0.2, random_state=None) -> (
            tuple)[DataFrame, DataFrame]:
        """
        Split the input graph DataFrame into training, testing, and validation sets.

        This function divides the input graph DataFrame into three subsets: training set, testing set, and validation set.
        The data splitting is performed based on the specified proportions for the test and validation sets.

        Parameters:
        - graph (DataFrame): The DataFrame representing the input graph.
        - test_size (float, optional): The proportion of the graph to include in the test set. Defaults to 0.2.
        - val_size (float, optional): The proportion of the training set to include in the validation set.
                                      Defaults to 0.1.
        - random_state (int or None, optional): Controls the randomness of the splitting.
                                                 If specified, it ensures reproducibility of the splitting.
                                                 Defaults to None.

        Returns:
        - tuple[DataFrame, DataFrame, DataFrame]: A tuple containing DataFrames representing the training, testing,
                                                   and validation sets, respectively.

        Note:
        - The sum of test_size and val_size should be less than 1.0 to ensure that there is data left for the training set.
        - If random_state is set, the data splitting will be reproducible across multiple function calls.
        - The function utilizes the train_test_split function from scikit-learn to perform the data splitting.
        """
        train_set, test_set = train_test_split(graph, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_train_test_val(self, train: DataFrame, test: DataFrame) -> None:
        train.to_csv(self.output_train, sep="\t", index=False, header=False)
        test.to_csv(self.output_test, sep="\t", index=False, header=False)

    def main(self):
        new_kg = self.generate_random_graph()
        print(new_kg.head())
        print(new_kg.shape[0])

        new_kg.to_csv(self.output_kg, sep=",", index=False)

        #train, test = self.split_train_test_val(new_kg)
        #self.save_train_test_val(train=train, test=test)


if __name__ == "__main__":
    random_KG_creator = RandomKG("/home/thomas/Documents/projects/kge_project/benchmark/data/kg_giant_orphanet.csv",
                                 output_path="/home/thomas/Documents/projects/kge_project/benchmark/data/random_100m")
    random_KG_creator.main()
