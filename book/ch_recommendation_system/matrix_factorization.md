---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## Download ml-1m dataset

```{code-cell} ipython3
import pandas as pd
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

import tabml.datasets


GLOBAL_SEED = 42  # number of life
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
```

```{code-cell} ipython3
df_dict = tabml.datasets.download_movielen_1m()
users, movies, ratings = df_dict["users"], df_dict["movies"], df_dict["ratings"]
ratings["Rating"] = ratings["Rating"] - 3
train_ratings, validation_ratings = train_test_split(ratings, test_size=0.1, random_state=42)
```

```{code-cell} ipython3
movie_index_by_id = {id: idx for idx, id in enumerate(movies["MovieID"])}
user_index_by_id = {id: idx for idx, id in enumerate(users["UserID"]) }
```

```{code-cell} ipython3
class MLDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        user_id = self.ratings["UserID"].iloc[index]
        movie_id = self.ratings["MovieID"].iloc[index]
        rating = self.ratings["Rating"].iloc[index]
        user_index = user_index_by_id[user_id]
        movie_index = movie_index_by_id[movie_id]
        return user_index, movie_index, rating
    
training_data = MLDataset(train_ratings)
validation_data = MLDataset(validation_ratings)
batch_size = 1024
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, num_workers=10
)
validation_dataloader = DataLoader(
    validation_data, batch_size=batch_size, shuffle=False, num_workers=10
)

for batch in train_dataloader:
    print(batch)
    break
```

```{code-cell} ipython3
from pytorch_lightning.loggers import TensorBoardLogger
import jdc

LR = 1
WEIGHT_DECAY = 5e-5


class MatrixFactorization(pl.LightningModule):
    """Pytorch lighting class for Matrix Factorization training.

    Attributes:
        n_users: number of users.
        n_items: number of items.
        n_factors: number of latent factors (or embedding size)
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int = 40):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.bias = nn.Parameter(data=torch.rand(1))
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        bias + user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Arguments:
            users: Array of user indices
            items : Array of item indices
        Returns:
            preds: Predicted ratings.
        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users) + self.bias
        preds += self.item_biases(items)
        preds += torch.reshape(
            torch.diag(torch.matmul(ues, torch.transpose(uis, 0, 1))),
            (-1, 1),
        )

        return torch.clip(preds.squeeze(), min=-2, max=2)

    def training_step(self, batch, batch_idx):
        users, items, rating = batch
        rating = rating.to(torch.float32)
        output = self.forward(users, items)
        loss = F.mse_loss(rating, output)
        self.log("batch_loss", loss)
        return {"loss": loss}  # for computing avg_loss in training_epoch_end

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        return optimizer
```

```{code-cell} ipython3
%%add_to MatrixFactorization
def validation_step(self, batch, batch_idx):
    users, items, rating = batch
    rating = rating.to(torch.float32)
    output = self.forward(users, items)
    loss = F.mse_loss(rating, output)
    self.log("batch_loss", loss)
    return {"loss": loss}  # for computing avg_loss in training_epoch_end

def training_epoch_end(self, outputs):
    avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    self.logger.experiment.add_scalars(
        "Loss", {"Train": avg_loss}, self.current_epoch
    )
    self.logger.experiment.add_scalars(
        "RMSE", {"Train": avg_loss ** 0.5}, self.current_epoch
    )
    epoch_dict = {"loss": avg_loss}

def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    self.logger.experiment.add_scalars(
        "Loss", {"Val": avg_loss}, self.current_epoch
    )
    self.logger.experiment.add_scalars(
        "RMSE", {"Val": avg_loss ** 0.5}, self.current_epoch
    )
    epoch_dict = {"loss": avg_loss}
```

```{code-cell} ipython3
logger = TensorBoardLogger("mf_tb_logs", name=f"lr{LR}_wd{WEIGHT_DECAY}")

n_users = len(user_index_by_id)
n_movies = len(movie_index_by_id)
n_factors = 40
model = MatrixFactorization(n_users=n_users, n_items=n_movies, n_factors=n_factors)
trainer = pl.Trainer(gpus=1, max_epochs=100, logger=logger)
trainer.fit(model, train_dataloader, validation_dataloader)
```

```{code-cell} ipython3
def eval_model(model, train_dataloader):
    loss = 0
    for users, items, rating in train_dataloader:
        pred = model(users, items)
        loss += F.mse_loss(pred, rating)
    RMSE = (loss / len(train_dataloader))**.5
    return RMSE
    
print("Train RMSE: {:.3f}".format(eval_model(model, train_dataloader)))
print("Validation RMSE: {:.3f}".format(eval_model(model, validation_dataloader)))
```

```{code-cell} ipython3

```
