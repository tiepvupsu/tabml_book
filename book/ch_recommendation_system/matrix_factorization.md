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

# build dataset
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


GLOBAL_SEED = 42  # number of life
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
```

```{code-cell} ipython3
ratings = pd.read_csv(
    "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/movielens/ml-1m/ratings.dat",
    delimiter="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    usecols=["UserID", "MovieID", "Rating"]
)
users = pd.read_csv(
    "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/movielens/ml-1m/users.dat",
    delimiter="::",
    engine="python",
    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
)
movies = pd.read_csv(
    "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/movielens/ml-1m/movies.dat",
    delimiter="::",
    encoding="ISO-8859-1",
    engine="python",
    names=["MovieID", "Title", "Genres"]
)
```

```{code-cell} ipython3
# Split train, val for ratings

ratings["Rating"] = ratings["Rating"] - 3
train_ratings, validation_ratings = train_test_split(ratings, test_size=0.1, random_state=42)
```

```{code-cell} ipython3
validation_ratings
```

```{code-cell} ipython3
users_in_validation = validation_ratings["UserID"].unique()
all_users = users["UserID"].unique()

print(f"There are {len(users_in_validation)} users in validation set.")
print(f"Total number of users: {len(all_users)}")
```

```{code-cell} ipython3
# Number of movies each user rated in train_ratings
train_ratings["UserID"].value_counts()
```

Mỗi người dùng trong tập huấn luyện đã đánh giá it nhất 14 bộ phim. Cá biệt, người dùng có ID 4169 đã đánh giá tới 2074 bộ phim.

```{code-cell} ipython3
movie_index_by_id = {id: idx for idx, id in enumerate(movies["MovieID"])}
user_index_by_id = {id: idx for idx, id in enumerate(users["UserID"]) }
```

```{code-cell} ipython3
train_ratings
```

```{code-cell} ipython3
class MlDataset(Dataset):
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
    
training_data = MlDataset(train_ratings)
validation_data = MlDataset(validation_ratings)
batch_size = 1024
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, num_workers=10
)
validation_dataloader = DataLoader(
    validation_data, batch_size=batch_size, shuffle=False, num_workers=10
)

def eval_model(model, train_dataloader):
    loss = 0
    for users, items, rating in train_dataloader:
        pred = model(users, items)
        loss += F.mse_loss(pred, rating) ** 0.5
    avg_loss = loss / len(train_dataloader)
    print(f"avg rmse: {avg_loss}")
```

```{code-cell} ipython3
from pytorch_lightning.loggers import TensorBoardLogger


class MatrixFactorization(pl.LightningModule):
    def __init__(self, n_users, n_items, n_factors=40, dropout_p=0, sparse=False):
        """
        # TODO: move docstring to class level
        Attributes:
        ----------
        n_users : int Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.bias = nn.Parameter(data=torch.rand(1))
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        user_bias + item_bias + user_embeddings.dot(item_embeddings)
        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices
        Returns
        -------
        preds : np.ndarray
            Predicted ratings.
        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users) + self.bias
        preds += self.item_biases(items)
        preds += torch.reshape(
            torch.diag(
                torch.matmul(
                    self.dropout(ues), torch.transpose(self.dropout(uis), 0, 1)
                )
            ),
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
    
    def validation_step(self, batch, batch_idx):
        users, items, rating = batch
        rating = rating.to(torch.float32)
        output = self.forward(users, items)
        loss = F.mse_loss(rating, output)
        self.log("batch_loss", loss)
        return {"loss": loss}  # for computing avg_loss in training_epoch_end   

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Loss", {"Train": avg_loss}, self.current_epoch)
        epoch_dict = {"loss": avg_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Loss", {"Val": avg_loss}, self.current_epoch)
        epoch_dict = {"loss": avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1, weight_decay=5e-4)
        return optimizer


logger = TensorBoardLogger("tb_logs", name="exp3")
# self.logger.experiment.add_scalars("losses", {"val_loss": loss})

n_users = len(user_index_by_id)
n_movies = len(movie_index_by_id)
n_factors = 40
model = MatrixFactorization(n_users=n_users, n_items=n_movies, n_factors=n_factors)
trainer = pl.Trainer(gpus=1, max_epochs=40, logger=logger)
trainer.fit(model, train_dataloader, validation_dataloader)
print("Train loss")
eval_model(model, train_dataloader)
print("Validation loss")
eval_model(model, validation_dataloader)
```

```{code-cell} ipython3

```
