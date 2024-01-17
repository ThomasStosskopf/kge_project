from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader

from tqdm.autonotebook import tqdm
import sys
sys.path.append("/home/thomas/Documents/projects/kge_project")
from benchmark.script.prepare_data.data_loader import load_data

class TransETrain:
    def __init__(self, emb_dim, lr, n_epochs, b_size, margin):
        self.em_dim = emb_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.b_size = b_size
        self.margin = margin
        self.kg_train, self.kg_val, self.kg_test = load_data()
        print("Work in progress ....")
        self.model = TransEModel(emb_dim, self.kg_train.n_ent, self.kg_train.n_rel, dissimilarity_type='L2')
        print("Model created with the requested parameters.")
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
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                      running_loss / len(self.dataloader)))
        print("done")
        self.model.normalize_parameters()





if __name__ == "__main__":
    print("Starting TransE training .....")
    model_trainer = TransETrain(100, 0.0004, 1000, 32768, 0.5)
    model_trainer.train()