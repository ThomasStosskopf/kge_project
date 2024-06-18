from pathlib import Path
from pandas import DataFrame, read_csv, concat, merge
import argparse
import matplotlib.pyplot as plt

class DataPropCalculator:

    def __init__(self, path_folder):
        self.path = Path(path_folder)
        col_names = ["from", "rel", "to"]
        self.train = read_csv(self.path.joinpath("train.csv"), sep="\t", names=col_names)
        self.test = read_csv(self.path.joinpath("test.csv"), sep="\t", names=col_names)
        self.list_proportion = list(self.calculate_reverse_relation_proportion(df_train=self.train, df_test=self.test))

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def reverse_relations_not_added(self, merged_df: DataFrame) -> DataFrame:
        """
        Calculate the number of reverse relations not added in the merged DataFrame.

        This function takes a DataFrame containing merged data and calculates the number of reverse relations
        that have not been added to the merged DataFrame. It removes lines where '_rev' is found in either 'rel_train'
        or 'rel_test' columns, compares the elements between the semicolons in 'rel_train' and 'rel_test', and filters
        out lines where these elements differ.

        Parameters:
        - merged_df (DataFrame): A DataFrame containing merged data with columns 'rel_train' and 'rel_test'.

        Returns:
        - int: The number of lines in the DataFrame after filtering, representing the count of reverse relations
               not added to the merged DataFrame.

        Note:
        - The function modifies the DataFrame by filtering out lines where '_rev' is found in 'rel_train' or 'rel_test'
          columns and comparing elements between semicolons in these columns.
        """
        # remove all lines where you can find _rev
        filtered_df = merged_df[
            ~(merged_df['rel_train'].str.contains('_rev') | merged_df['rel_test'].str.contains('_rev'))]

        # Separate elements between semicolons in rel_train and rel_test
        filtered_df['rel_train_elements'] = filtered_df['rel_train'].str.split(';')
        filtered_df['rel_test_elements'] = filtered_df['rel_test'].str.split(';')

        # Compare elements between semicolons in rel_train and rel_test and filter lines
        filtered_df = filtered_df[
            filtered_df['rel_train_elements'].apply(lambda x: x[1]) == filtered_df['rel_test_elements'].apply(
                lambda x: x[1])]

        return filtered_df

    def count_reverse_relations_added(self, merged_df: DataFrame) -> int:
        """
        Count the number of reverse relations added to the merged DataFrame.

        This function calculates the number of reverse relations added to the merged DataFrame by inspecting
        the 'rel_train' and 'rel_test' columns. We simply count the number of relation with '_rev' in the
        rel_train column and in the rel_test column. We sum the two and get the total number of relation that
        where added. We then return this sum that can be used to calculate the proportion of reverse relations in
        the test data.

        Parameters:
        - merged_df (DataFrame): A DataFrame containing merged data with columns 'rel_train' and 'rel_test'.

        Returns:
        - int: The total count of reverse relations added to the merged DataFrame.
        """
        relation_in_test = merged_df["rel_test"]
        relation_in_train = merged_df["rel_train"]
        results_filtered = [relation for relation in relation_in_test if "_rev" in relation]
        filter_rev_in_train = [relation for relation in relation_in_train if "_rev" in relation]
        # here you got the real number of reverse relations added by the prepare_graph.py script
        nb_rev_rel_added = len(results_filtered) + len(filter_rev_in_train)
        return nb_rev_rel_added

    def count_false_reverse_relation(self, merged_df: DataFrame) -> DataFrame:
        """
        Count the number of false reverse relations in the merged DataFrame.

        This count the number of triplets where the head in train and the tail in test are the same and the tail of
        train and the tail from test are also the same AND the rel part is different.
        ex :        train                                       test
           head: A   rel: knows   tail: B               head: B   rel: loves    tail: A


        Parameters:
        - merged_df (DataFrame): A DataFrame containing merged data with columns 'rel_train' and 'rel_test'.

        Returns:
        - int: The number of lines in the DataFrame after filtering, representing the count of false reverse relations.

        Note:
        - The function modifies the DataFrame by removing lines where '_rev' is found in 'rel_train' or 'rel_test'
          columns and comparing elements between semicolons in these columns.
        """
        # remove all lines where you can find _rev
        filtered_df = merged_df[
            ~(merged_df['rel_train'].str.contains('_rev') | merged_df['rel_test'].str.contains('_rev'))]

        # Separate elements between semicolons in rel_train and rel_test
        filtered_df['rel_train_elements'] = filtered_df['rel_train'].str.split(';')
        filtered_df['rel_test_elements'] = filtered_df['rel_test'].str.split(';')
        # Compare elements between semicolons in rel_train and rel_test and filter lines
        filtered_df = filtered_df[
            filtered_df['rel_train_elements'].apply(lambda x: x[1]) != filtered_df['rel_test_elements'].apply(
                lambda x: x[1])]
        return filtered_df

    def calculate_reverse_relation_proportion(self, df_train: DataFrame,
                                              df_test: DataFrame, df_val: DataFrame = None) -> tuple[
        float, float, float, float]:
        """
        Calculate the proportion of reverse relations added and not added in the dataset.

        This function computes the proportion of reverse relations that were added and not added in the dataset.
        It merges the training and testing DataFrames, counts the added and not added reverse relations,
        and calculates the proportions.

        Parameters:
        - df_train (DataFrame): The training DataFrame.
        - df_test (DataFrame): The testing DataFrame.
        - df_val (DataFrame, optional): The validation DataFrame. Defaults to None.

        Returns:
        - Tuple[float, float]: A tuple containing the proportion of added and not added reverse relations, respectively.
        """
        if df_val is not None:
            full_train = concat([df_train, df_val], ignore_index=True)
        else:
            full_train = df_train
        # Merged dataframes on the columns 'from' and 'to'
        merged_df = merge(full_train, df_test, left_on=['from', 'to'], right_on=['to', 'from'],
                          suffixes=('_train', '_test'))
        # saved the merged dataframe for controls
        merged_df.to_csv('benchmark/data/merged_reverse_train_test_test_3.csv', index=False, header=True)

        # here you got the real number of reverse relations added by the prepare_graph.py script
        nb_rev_rel_added = self.count_reverse_relations_added(merged_df)
        df_reverse_relation_not_added = self.reverse_relations_not_added(merged_df)
        nb_rev_not_added = len(df_reverse_relation_not_added)
        df_false_rev_rel = self.count_false_reverse_relation(merged_df)
        nb_false_rev_rel = len(df_false_rev_rel)

        # Calculer la proportion
        total_reverse_relations = len(full_train) + len(df_test)
        print(f"Total number of reverse relations: {total_reverse_relations}")
        proportion_rev_added = round((nb_rev_rel_added / len(df_test)) * 100, 2)
        proportion_rev_not_added = round((nb_rev_not_added / len(df_test)) * 100, 2)
        proportion_false_rev = round((nb_false_rev_rel / len(df_test)) * 100, 2)
        proportion_not_rev = round(100 - (proportion_false_rev+proportion_rev_added+proportion_rev_not_added), 2)
        return proportion_rev_added, proportion_rev_not_added, proportion_false_rev, proportion_not_rev

    def make_pie_chart(self):
        # Data to plot
        fig, ax1 = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        labels = ['known relation (added)', 'known relation', 'known heterogeneous reverse relations',
                  'unknown relation']
        sizes = self.list_proportion  # sizes in percentage
        explode = (0, 0, 0, 0.1)
        custom_colors = ['#E8C517', '#E89317', 'grey', '#00A6F0']
        # Plot
        wedges, texts, autotexts = ax1.pie(sizes, autopct='%1.2f%%',
                                           shadow=True, startangle=90, explode=explode, colors=custom_colors)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Proportion of reverse relations in test file compare with train relation', fontweight='bold')
        # Move the legend closer to the pie chart
        ax1.legend(wedges, labels,
                   title="Categories",
                   loc="center left",
                   )
        plt.savefig('benchmark/output/img/pie_chart.png', bbox_inches='tight', transparent=True)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the proportion of reverse in the test data and train data")
    parser.add_argument("--input", type=str, help="Path to the folder where test and train file are.")
    parser.add_argument("--output", type=str, help="Path to output folder.")
    args = parser.parse_args()
    folder = args.input

    calculator = DataPropCalculator(path_folder=folder)
    print(calculator.make_pie_chart())
    print(calculator.list_proportion)

