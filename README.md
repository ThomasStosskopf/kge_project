# kge_project

The folder named "Benchmark" is where I put the scripts used in the benchmark to compare 4 different data pre-processing approaches. 

## Benchmark

In the "scripts" folder there is three folders:
- kge_models: the "pipeline_embedding.py" script is there. It is used to train KGE models and can be used with commands line in the terminal with the followed argument: "--input_folder", "--output", "--model", "--epochs", "--batch_size", "--emb_dim".

- prepare_data: there is severa script to used to pre-process the data for KGE embedding models. It is where I defined the different approaches used.

- evaluation_script: in this folder there is 4 scripts. metric_finder.py and table_maker.py are used together to evaluate a KGE models on all the relation type of the KG and then make two tables one with the hits@10 table and one with the MRR table.

## Usage

### Prepare data:

    python3 benchmark/script/prepare_data/prepare_kg_third_method.py --input kg_giant_orphanet.csv  --output /benchmark/data/third_approach

### run a model: 

    python3 benchmark/script/pipeline_embedding.py --train benchmark/data/third_method/train_set_third_method.csv --test benchmark/data/third_method/test_set_third_method.csv --output benchmark/output/output_third_method/ --model distmult --epochs 100 --batch_size 256 --emb_size 128 --learning_rate 0.0001

### Evaluate on specific relation

    python3 
