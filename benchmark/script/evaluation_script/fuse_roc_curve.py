from pathlib import Path
import argparse
from pandas import read_csv, DataFrame
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt


class FuseRocCurve:

    def __init__(self, path1, path2, path3, path4, output):
        self.path1 = Path(path1)
        self.path2 = Path(path2)
        self.path3 = Path(path3)
        self.path4 = Path(path4)
        self.output_path = Path(output)
        self.fpr_m1 = np.loadtxt(self.path1.joinpath('output_classifier/fpr.txt'))
        self.fpr_m2 = np.loadtxt(self.path2.joinpath('output_classifier/fpr.txt'))
        self.fpr_m3 = np.loadtxt(self.path3.joinpath('output_classifier/fpr.txt'))
        self.fpr_m4 = np.loadtxt(self.path4.joinpath('output_classifier/fpr.txt'))
        self.tpr_m1 = np.loadtxt(self.path1.joinpath('output_classifier/tpr.txt'))
        self.tpr_m2 = np.loadtxt(self.path2.joinpath('output_classifier/tpr.txt'))
        self.tpr_m3 = np.loadtxt(self.path3.joinpath('output_classifier/tpr.txt'))
        self.tpr_m4 = np.loadtxt(self.path4.joinpath('output_classifier/tpr.txt'))

    def plot_roc_curves(self):
        plt.figure()
        plt.plot(self.fpr_m1, self.tpr_m1, color='#f36f1c', lw=2, label='Method 1 ROC curve')  # Neon pink
        plt.plot(self.fpr_m2, self.tpr_m2, color='#2ca02c', lw=2, label='Method 2 ROC curve')  # Neon green
        plt.plot(self.fpr_m3, self.tpr_m3, color='#d62728', lw=2, label='Method 3 ROC curve')  # Neon red
        plt.plot(self.fpr_m4, self.tpr_m4, color='#46a9be', lw=2, label='Method 4 ROC curve')  # Neon cyan
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right", fontsize='x-large')
        plt.savefig(self.output_path)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add description here")

    parser.add_argument("--folder1", type=str, help="Path to the trained model' folder 1.")
    parser.add_argument("--folder2", type=str, help="Path to the trained model' folder 2.")
    parser.add_argument("--folder3", type=str, help="Path to the trained model' folder 3.")
    parser.add_argument("--folder4", type=str, help="Path to the trained model' folder 4.")
    parser.add_argument("--output", type=str, help="Path the file where the image will be saved.")

    args = parser.parse_args()

    fuser = FuseRocCurve(path1=args.folder1,
                         path2=args.folder2,
                         path3=args.folder3,
                         path4=args.folder4,
                         output=args.output)

    print(fuser.plot_roc_curves())
