from torch.optim import Adam
from pandas import read_csv, concat
from torchkge.data_structures import KnowledgeGraph
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.evaluation import TripletClassificationEvaluator
from tqdm.autonotebook import tqdm


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


class BaseTrain:
    """
    BaseTrain is a base class for training knowledge graph embedding models.

    Parameters:
        model_class (torch.nn.Module): The class of the knowledge graph embedding model.
        emb_dim (int): The dimensionality of the entity and relation embeddings.
        lr (float): The learning rate for the Adam optimizer.
        n_epochs (int): The number of training epochs.
        b_size (int): The batch size for training.
        margin (float): The margin parameter used in the margin loss function.

    Attributes:
        emb_dim (int): The dimensionality of the entity and relation embeddings.
        lr (float): The learning rate for the Adam optimizer.
        n_epochs (int): The number of training epochs.
        b_size (int): The batch size for training.
        margin (float): The margin parameter used in the margin loss function.
        kg_train (torchkge.data.KnowledgeGraph): The training knowledge graph.
        kg_val (torchkge.data.KnowledgeGraph): The validation knowledge graph.
        kg_test (torchkge.data.KnowledgeGraph): The testing knowledge graph.
        model (torchkge.models): The knowledge graph embedding model.
        criterion (torch.nn.Module): The margin loss function.
        optimizer (torch.optim.Adam): The Adam optimizer for model training.
        sampler (torchkge.sampling.BernoulliNegativeSampler): The negative sampler for generating negative samples.
        dataloader (torchkge.utils.DataLoader): The data loader for batching training data.

    Methods:
        train():
            Train the knowledge graph embedding model using the specified parameters.

        eval_link_prediction():
            Evaluate the link prediction performance of the trained model on the test set.

        eval_triplet_classification():
            Evaluate the triplet classification performance of the trained model on the test set.
    """
    def __init__(self, model_class, emb_dim, lr, n_epochs, b_size, margin):
        self.emb_dim = emb_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.b_size = b_size
        self.margin = margin
        self.kg_train, self.kg_val, self.kg_test = load_data()
        print("Work in progress ....")
        self.model = model_class(emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
        print("Model created with the requested parameters.")
        self.criterion = MarginLoss(margin)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.sampler = BernoulliNegativeSampler(self.kg_train)
        self.dataloader = DataLoader(self.kg_train, batch_size=b_size)

    def train(self):
        """
        Train the knowledge graph embedding model using the specified parameters.
        """
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
        """
        Evaluate the link prediction performance of the trained TransE model on the test set.
        """
        evaluator = LinkPredictionEvaluator(self.model, self.kg_test)
        evaluator.evaluate(b_size=32)
        evaluator.print_results()

    def eval_triplet_classification(self):
        """
        Evaluate the triplet classification performance of the trained TransE model on the test set.
        """
        # Triplet classification evaluation on test set by learning thresholds on validation set
        evaluator = TripletClassificationEvaluator(self.model, self.kg_val, self.kg_test)
        evaluator.evaluate(b_size=128)

        print('Accuracy on test set: {}'.format(evaluator.accuracy(b_size=128)))
