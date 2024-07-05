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

### Output Files

The processed data is saved in the following files:

    KG_node_map.txt: Mapping of nodes with their unique indices.
    KG_edge_list.txt: Edge list of the knowledge graph.
    train.txt: Training set.
    test.txt: Testing set.
    valid.txt: Validation set (if applicable).
    type_to_entities.csv: Mapping of node types to entities.

The prepare_kg_first_method.py used all function defined by the SHEPHERD. 

## Second approach

The second approach do not add reverse relation in the graph. But it is an other extension of the PrepareKG class. 

## Third approach

The Third approach is an extension of the PrepareKGSecondMethod class. 
We preprocess the data the same until we have to split the data. WIth the third approach we split the data 80/20 per type of relation in the KG. 

## Fourth approach

The Fourth approach is an extension of the PrepareKGThirdMethod class.
We preprocess data same as before but hide a chosen relation in the training dataset. 

