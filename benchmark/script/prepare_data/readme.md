# Prepare data

This folder contain the scripts where I defined the different approaches to pre-process data for KGE models.
The PrepareKG class is designed to process and prepare knowledge graph data for machine learning and data analysis tasks. This class reads a CSV file containing knowledge graph data, cleans the data, assigns unique identifiers to nodes, reindexes edges, extracts the largest connected component (LCC), and saves the processed data in various formats. Additionally, the class can split the data into training, testing, and validation sets, and handle reciprocal relationships in the graph.

## first appraoch

### Usage

 #### Initializing the Class

To initialize the PrepareKG class, provide the path to the knowledge graph CSV file and the output folder where the processed data will be saved:

'''from prepare_kg import PrepareKG

kg_path = 'path/to/your/kg_data.csv'
output_folder = 'path/to/output_folder'

prepare_kg = PrepareKG(kg_path=kg_path, output_folder=output_folder)'''

