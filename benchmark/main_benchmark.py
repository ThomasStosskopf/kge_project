from script.prepare_data import prepare_file as pf
from script.prepare_data import prepare_graph
from script.kge_models.TransETrain import TransETrain
from script.kge_models.DistMultTrain import DistMultTrain


class Main:

    def __init__(self) -> None:
        pass

    def main(self) -> None:
        prepare_graph.main()
        prepare_data = pf.PrepareData('benchmark/data/KG_edgelist_mask.txt')
        prepare_data.main()
        print("Starting TransE training .....")
        transe_model_trainer = TransETrain(100, 0.0004, 1000, 32768, 0.5)
        transe_model_trainer.train()
        transe_model_trainer.eval_link_prediction()
        transe_model_trainer.eval_triplet_classification()
        print("Starting DistMult training .....")
        distmult_model_trainer = DistMultTrain(100, 0.0004, 1000, 32768, 0.5)
        distmult_model_trainer.train()
        distmult_model_trainer.eval_link_prediction()
        distmult_model_trainer.eval_triplet_classification()


if __name__ == "__main__":
    print("Starting now .....")
    launch = Main()
    launch.main()
    print("END")
