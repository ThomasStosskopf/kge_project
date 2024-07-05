# kge_project

The folder named "Benchmark" is where I put the scripts used in the benchmark to compare 4 different data pre-processing approaches. 

## Benchmark

In the "scripts" folder there is three folders:
- kge_models: the "pipeline_embedding.py" script is there. It is used to train KGE models and can be used with commands line in the terminal with the followed argument: "--input_folder", "--output", "--model", "--epochs", "--batch_size", "--emb_dim".

- prepare_data: there is severa script to used to pre-process the data for KGE embedding models. It is where I defined the different approaches used.

- evaluation_script: in this folder there is 4 scripts. metric_finder.py and table_maker.py are used together to evaluate a KGE models on all the relation type of the KG and then make two tables one with the hits@10 table and one with the MRR table.

## Usage

### run a model: 

    python benchmark/script/kge_models/pipeline_embedding.py --input_folder alignement/data/essaie_05_avril/split_kg/drug_prep_data/ --output alignement/data/essaie_05_avril/output_drug/ --model distmult epochs 4 emb_dim 128
