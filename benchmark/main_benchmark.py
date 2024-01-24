import argparse
from script.prepare_data import prepare_file as pf
from script.prepare_data import prepare_graph
from script.kge_models.TransETrain import TransETrain
from script.kge_models.DistMultTrain import DistMultTrain
# from script.kge_models.ConvKBTrain import ConvKBTrain


class Main:

    def __init__(self, args) -> None:
        self.args = args

    def main(self) -> None:
        prepare_graph.main()

        prepare_data = pf.PrepareData('benchmark/data/KG_edgelist_mask.txt')
        prepare_data.main()
        # if self.args.model == 'transe':
        # #     print("Starting TransE training .....")
        #     transe_model_trainer = TransETrain(
        #         self.args.emb_dim, self.args.lr,
        #         self.args.n_epochs, self.args.b_size, self.args.margin
        #     )
        #     print("emb_dim: ", transe_model_trainer.emb_dim)
        #     print("lr: ", transe_model_trainer.lr)
        #     print("n_epochs: ", transe_model_trainer.n_epochs)
        #     print("b_size: ", transe_model_trainer.b_size)
        #     print("margin: ", transe_model_trainer.margin)
        #     transe_model_trainer.train()
        #     transe_model_trainer.eval_link_prediction()
        #     transe_model_trainer.eval_triplet_classification()
        # elif self.args.model == 'distmult':
        #     print("Starting DistMult training .....")
        #     distmult_model_trainer = DistMultTrain(
        #         self.args.emb_dim, self.args.lr,
        #         self.args.n_epochs, self.args.b_size, self.args.margin
        #     )
        #     distmult_model_trainer.train()
        #     distmult_model_trainer.eval_link_prediction()
        #     distmult_model_trainer.eval_triplet_classification()
        # print("Starting ConvKB training .....")
        # convkb_model_trainer = ConvKBTrain(emb_dim=100, n_filters=64, lr=0.0004, n_epochs=1000, b_size=32768, margin=0.5)
        # convkb_model_trainer.train()
        # convkb_model_trainer.eval_link_prediction()
        # convkb_model_trainer.eval_triplet_classification()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Embedding Training")
    parser.add_argument('--model', choices=['transe', 'distmult', 'convkb'], help="Choose the KGE model")
    parser.add_argument('--emb_dim', type=int, help="Embedding dimension")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--n_epochs', type=int, help="Number of epochs")
    parser.add_argument('--b_size', type=int, help="Batch size")
    parser.add_argument('--margin', type=float, help="Margin for the margin-based ranking loss")

    args = parser.parse_args()

    print("Starting now .....")
    launch = Main(args)
    launch.main()
    print("END")