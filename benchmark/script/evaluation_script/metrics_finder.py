from pykeen.triples import TriplesFactory
from pathlib import Path
from pykeen.evaluation import RankBasedEvaluator
import torch


class MetricsFinder:
    """
    A class to find metrics for evaluating a knowledge graph model.

    Attributes:
        path (Path): The path to the directory containing the model and data files.
        model (torch.nn.Module): The trained knowledge graph embedding model.
        mapped_testing_triples (torch.Tensor): Mapped testing triples used for evaluation.
        mapped_training_triples (torch.Tensor): Mapped training triples used for filtering during evaluation.
        relation_to_id (dict): Mapping of relation names to their corresponding IDs.
        list_metrics (list): List of metrics to evaluate the model on.
    """

    def __init__(self, path_to_model_dir):
        """
        Initialize the MetricsFinder object.

        Args:
            path_to_model_dir (str): The path to the directory containing the model and data files.
        """
        self.path = Path(path_to_model_dir)
        self.model = torch.load(self.path.joinpath("trained_model.pkl"))
        self.mapped_testing_triples = TriplesFactory.from_path_binary(
            self.path.joinpath("testing_triples")).mapped_triples
        self.mapped_training_triples = TriplesFactory.from_path_binary(
            self.path.joinpath("training_triples")).mapped_triples
        self.relation_to_id = TriplesFactory.from_path_binary(self.path.joinpath("testing_triples")).relation_to_id
        self.list_metrics = ['mrr', 'hits@10']

    def get_mapped_testing_triples(self):
        """
        Get the mapped testing triples.

        Returns:
            torch.Tensor: Mapped testing triples.
        """
        return self.mapped_testing_triples

    def get_list_of_id_in_mapped_testing_triples(self):
        """
        Get the list of IDs in mapped testing triples.

        Returns:
            set: Set of IDs present in mapped testing triples.
        """
        return set(self.mapped_testing_triples[:, 1].tolist())

    def evaluate_a_relation(self, rel_id: int, mapped_testing_triples, mapped_training_triples):
        """
        Evaluate a specific relation based on the given ID.

        Args:
            id (int): The ID of the relation to evaluate.
            mapped_testing_triples (Tensor): Mapped testing triples for evaluation.
            mapped_training_triples (Tensor): Mapped training triples for filtering during evaluation.

        Returns:
            dict: Evaluation results containing various metrics.
        """
        evaluator = RankBasedEvaluator()
        # Evaluate
        results = evaluator.evaluate(
            model=self.model,
            mapped_triples=mapped_testing_triples,
            batch_size=1024,
            additional_filter_triples=[mapped_training_triples],
            restrict_relations_to=[rel_id],

        )
        return results

    def create_dict_of_evaluation(self, relation_to_id_dict, list_id_in_test) -> dict:
        dict_rel_score = {}
        for rel in relation_to_id_dict:
            if relation_to_id_dict[rel] in list_id_in_test:
                print(f"relation: {rel}, id: {self.relation_to_id[rel]}")
                results = self.evaluate_a_relation(relation_to_id_dict[rel], self.mapped_testing_triples,
                                                   self.mapped_training_triples)
                dict_metrics = self.get_metrics_listed(eval_results=results, metrics_list=self.list_metrics)
                dict_rel_score[rel] = dict_metrics
        return dict_rel_score

    def get_metrics_listed(self, eval_results, metrics_list):

        dict_metrics = {}
        for metric in metrics_list:
            print(metric)
            dict_metrics[metric] = eval_results.get_metric(metric)
        return dict_metrics


if __name__ == "__main__":
    metrics_finder = MetricsFinder("benchmark/output/output_first_method/transe_75epochs_dfltbatch_size")
    dict_rel_score = metrics_finder.create_dict_of_evaluation(metrics_finder.relation_to_id,
                                                              metrics_finder.get_list_of_id_in_mapped_testing_triples())
    print(dict_rel_score)
