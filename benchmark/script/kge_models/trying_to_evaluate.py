import torch
import pykeen.nn
from typing import List
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
from pathlib import Path


model = torch.load("benchmark/output/testingsomething/trained_model.pkl")

path = Path('benchmark/output/testingsomething')
# Charger les triples de test et de train
triples_factory_test = TriplesFactory.from_path_binary(path.joinpath("testing_triples"))
triples_factory_train = TriplesFactory.from_path_binary(path.joinpath("training_triples"))

# Mapp triples
mapped_testing_triples = triples_factory_test.mapped_triples
mapped_training_triples = triples_factory_train.mapped_triples

# Find the id to relation to choose the relation on which we want to evaluate
relation_to_id = triples_factory_test.relation_to_id
id = relation_to_id["drug;drug_drug;drug"]
print(id)

from pykeen.evaluation import RankBasedEvaluator

evaluator = RankBasedEvaluator()
# Evaluate
results = evaluator.evaluate(
    model=model,
    mapped_triples=mapped_testing_triples,
    batch_size=1024,
    additional_filter_triples=[mapped_training_triples],
    restrict_relations_to=[21],
    pre_filtered_triples = False
    )
print(results)
print(results.get_metric('Hits@10'))

