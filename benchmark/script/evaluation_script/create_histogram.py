import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from pandas import DataFrame, read_csv, concat, merge


# Read the CSV file
data = read_csv("benchmark/output/eval/barplot/transe_hit@10.csv")

# Extract the column names and values
columns = data.columns
values = data.values.flatten()

# Define colors for each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Using Tableau Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
colors = ['#8A2BE2', '#39FF14', '#FF073A', '#00FFFF']
# Create a bar plot with custom colors
plt.bar(range(len(values)), values, color=colors)
plt.xlabel('Methods')
plt.ylabel('hit@10 score')
plt.title('hit@10 results for TransE model with different pre-process data method')

plt.ylim(0, 0.3)
# Add column names below each bar
plt.xticks(range(len(values)), columns)

plt.show()

