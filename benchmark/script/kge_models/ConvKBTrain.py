from torch.optim import Adam
from torchkge.models import ConvKBModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from tqdm.autonotebook import tqdm
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.evaluation import TripletClassificationEvaluator
from torchkge.data_structures import KnowledgeGraph
from pandas import read_csv, concat


def load_data():
    """
    Load CSV files into DataFrames and create a KnowledgeGraph.

    Returns:
    - Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    A tuple containing three KnowledgeGraph instances for training, validation, and test sets.
    """
    # Load CSV file into a DataFrame

    df1 = read_csv('benchmark/data/train_set.csv',
                   sep=',', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv('benchmark/data/val_set.csv',
                   sep=',', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv('benchmark/data/test_set.csv',
                   sep=',', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


class ConvKBTrain:
    def __init__(self, emb_dim, n_filters, lr, n_epochs, b_size, margin):
        self.emb_dim = emb_dim
        self.n_filters = n_filters
        self.lr = lr
        self.n_epochs = n_epochs
        self.b_size = b_size
        self.margin = margin
        self.kg_train, self.kg_val, self.kg_test = load_data()
        print("Work in progress ....")
        self.model = ConvKBModel(emb_dim=emb_dim, n_filters=n_filters,
                                 n_entities=self.kg_train.n_ent, n_relations=self.kg_train.n_rel)
        print("ConvKB Model created with the requested parameters.")
        self.criterion = MarginLoss(margin)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.sampler = BernoulliNegativeSampler(self.kg_train)
        self.dataloader = DataLoader(self.kg_train, batch_size=b_size)

    def train(self):
        print("Training begins...")
        iterator = tqdm(range(self.n_epochs), unit='epoch')
        for epoch in iterator:
            running_loss = 0.0
            for i, batch in enumerate(self.dataloader):
                h, t, r = batch[0], batch[1], batch[2]
                n_h, n_t = self.sampler.corrupt_batch(h, t, r)

                self.optimizer.zero_grad()

                # forward + backward + optimize
                pos, neg = self.model(h, t, r, n_h, n_t)
                loss = self.criterion(pos, neg)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(self.dataloader)))
        print("done")
        self.model.normalize_parameters()

    def eval_link_prediction(self):
        evaluator = LinkPredictionEvaluator(self.model, self.kg_test)
        evaluator.evaluate(b_size=32)
        evaluator.print_results()

    def eval_triplet_classification(self):
        evaluator = TripletClassificationEvaluator(self.model, self.kg_val, self.kg_test)
        evaluator.evaluate(b_size=128)

        print('Accuracy on test set: {}'.format(evaluator.accuracy(b_size=128)))


if __name__ == "__main__":
    print("Starting ConvKB training .....")
    convkb_trainer = ConvKBTrain(emb_dim=100, n_filters=32, lr=0.0004, n_epochs=1000, b_size=32768, margin=0.5)
    convkb_trainer.train()
    convkb_trainer.eval_link_prediction()
    convkb_trainer.eval_triplet_classification()
