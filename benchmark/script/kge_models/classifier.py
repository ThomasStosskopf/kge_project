from pathlib import Path
import argparse
from pandas import read_csv, DataFrame
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt


class Classifier:


    def __init__(self, data_folder):
        self.input_path = Path(data_folder)
        self.input_path_data = self.input_path.joinpath("data_preprocess_classifier")
        self.train = read_csv(self.input_path_data.joinpath("train.tsv"), sep="\t", index_col=False, converters={'head': self.convert_string_to_array, 'tail': self.convert_string_to_array})
        self.test = read_csv(self.input_path_data.joinpath("test.tsv"), sep="\t", index_col=False, converters={'head': self.convert_string_to_array, 'tail': self.convert_string_to_array})
        self.saving_directory = self.input_path.joinpath("output_classifier")
        self.saving_directory.mkdir(parents=True, exist_ok=True)
        self.accuracy, self.confusion_mat, self.roc_auc, self.f1 = self.train2(train_df=self.train,
                                                                                   test_df=self.test)

    def convert_string_to_array(self, s):
        return np.array(list(map(float, s.split(','))))

    def get_train(self):
        return self.train

    def train2(self, train_df, test_df):
        # Extract features and labels from train DataFrame
        X_from_train = np.vstack(train_df['head'].values)
        X_to_train = np.vstack(train_df['tail'].values)
        X_train = np.hstack((X_from_train, X_to_train))
        y_train = train_df['mark'].values

        # Extract features and labels from test DataFrame
        X_from_test = np.vstack(test_df['head'].values)
        X_to_test = np.vstack(test_df['tail'].values)
        X_test = np.hstack((X_from_test, X_to_test))
        y_test = test_df['mark'].values

        # Train the RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=100000, min_samples_split=2, random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = round(accuracy_score(y_test, y_pred), 2)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate ROC curve and AUROC
        y_prob = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        np.savetxt(self.saving_directory.joinpath("fpr.txt"), fpr)
        np.savetxt(self.saving_directory.joinpath("tpr.txt"), tpr)
        roc_auc = round(auc(fpr, tpr), 2)

        # Calculate F1 score
        f1 = round(f1_score(y_test, y_pred), 2)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(self.saving_directory.joinpath("roc_curve.png"))
        plt.show()


        return accuracy, cm, roc_auc, f1

    def create_table(self) -> DataFrame:
        dict_metric = {
            "Metric": ["Accuracy", "AUROC", "F1_score"],
            "Value": [self.accuracy, self.roc_auc, self.f1]
        }
        df = DataFrame(dict_metric)
        return df


    def save_file(self) -> str:
        df_to_save = self.create_table()
        df_to_save.to_csv(self.saving_directory.joinpath("metrics.tsv"), sep="\t")
        return "files saved"



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Add description here")

    parser.add_argument("--folder", type=str, help="Path to the trained model' folder.")
    parser.add_argument("--mapping", type=str, help="Path to the file mapping the entities and their id.")
    parser.add_argument("--type", type=str, help="Path to the file mapping entities to their type.")
    parser.add_argument("--output", type=str, help="Path the file where the image will be saved.")

    args = parser.parse_args()

    an_input = args.folder

    classifier = Classifier(data_folder=an_input)

    print(classifier.save_file())

