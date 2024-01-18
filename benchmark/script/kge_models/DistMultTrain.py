from .BaseTrain import BaseTrain
from torchkge.models import DistMultModel


class DistMultTrain(BaseTrain):
    """
    TransETrain is a class for training the TransE knowledge graph embedding model.

    Parameters:
        emb_dim (int): The dimensionality of the entity and relation embeddings.
        lr (float): The learning rate for the Adam optimizer.
        n_epochs (int): The number of training epochs.
        b_size (int): The batch size for training.
        margin (float): The margin parameter used in the margin loss function.

    Attributes:
        Inherits attributes from BaseTrain.

    Methods:
        Inherits methods from BaseTrain.

    """
    def __init__(self, emb_dim, lr, n_epochs, b_size, margin):
        model_class = DistMultModel
        super().__init__(model_class, emb_dim, lr, n_epochs, b_size, margin)


if __name__ == "__main__":
    print("Starting TransE training .....")
    model_trainer = DistMultTrain(100, 0.0004, 1000, 32768, 0.5)
    model_trainer.train()