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

ratings["Rating"] = ratings["Rating"]
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
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_MOVIES = len(movies)
NUM_USERS = len(users)


class FactorizationMachineDataset(Dataset):
    def __init__(self, user_df, movie_df, rating_df):
        self.user_df = user_df
        self.movie_df = movie_df
        self.rating_df = rating_df
        self.age_vocab = Vocab("ages.txt")
        self.gender_vocab = Vocab("genders.txt")
        self.occupation_vocab = Vocab("occupations.txt")
        self.genre_vocab = Vocab("genres.txt")
        self.dims = [
            NUM_MOVIES,
            NUM_USERS,
            self.gender_vocab.len,
            self.age_vocab.len,
            self.occupation_vocab.len,
            self.genre_vocab.len,
        ]
        self.input_dim = sum(self.dims)

    def __len__(self):
        return len(self.rating_df)

    def __getitem__(self, ind):
        """
        movie_id, movie_genre, user_id, user_gender, user_age, user_occupation
        """
        user_ind = self.rating_df["UserID"].iloc[ind]
        movie_true_ind = movie_index_by_id[self.rating_df["MovieID"].iloc[ind]]
        rating = self.rating_df["Rating"].iloc[ind]
        user_ind = user_index_by_id[user_ind]
        gender_ind = self.gender_vocab.find_index(
            self.user_df["Gender"][user_ind]
        )
        assert gender_ind < 2
        age_ind = self.age_vocab.find_index(            str(self.user_df["Age"][user_ind])        )
        assert age_ind < self.age_vocab.len
        # occupation is given as index already
        occupation_ind = self.user_df["Occupation"][user_ind]
        assert occupation_ind < self.occupation_vocab.len
        genre_inds = []
        genres = self.movie_df["Genres"][movie_true_ind].split("|")
        if genres:
            genre_inds = [self.genre_vocab.find_index(genre) for genre in genres]
        inputs = self.gen_multihot_input_tensor(
            [
                [movie_true_ind],
                [user_ind],
                [gender_ind],
                [age_ind],
                [occupation_ind],
                genre_inds,
            ]
        )
        return inputs, rating - 3

    def gen_multihot_input_tensor(self, list_inds: List[List[int]]):
        """Generates a 1-d tensor as input of factorization machine model.

        Args:
            list_inds: a list of list of indices for each field defined in self.dims

        Example:
            If self.dims = [3, 5] and list_inds = [[2], [1, 3]] then return a tensor
            with values: [0, 0, 1, 0, 1, 0, 1, 0].
        """
        assert len(list_inds) == len(
            self.dims
        ), f"len(list_inds) ({len(list_inds)})  != len(self.dims) ({len(self.dims)})."

        offset = 0
        sparse = torch.zeros((self.input_dim), dtype=torch.float)
        one_inds = []
        for i, inds in enumerate(list_inds):
            assert not inds or all([ind < self.dims[i] for ind in inds])
            one_inds.extend([offset + ind for ind in inds])
            offset += self.dims[i]
        indices = torch.LongTensor([one_inds])
        values = torch.ones_like(torch.tensor((len(one_inds),)), dtype=torch.float)
        sparse[indices] = values
        return sparse


class Vocab:
    def __init__(self, vocab_file: str):
        """
        Args:
            vocab_file: path to file containing dictionary, each line is a word.
        """
        self.vocab_file = vocab_file
        self.vocab_list = self.read_vocab_file()
        self.len = len(self.vocab_list)

    def read_vocab_file(self):
        """Returns a list of words."""
        with open(self.vocab_file) as f:
            words = f.read().splitlines()  # avoid newline at the end
        return words

    def find_index(self, word):
        return self.vocab_list.index(word)


def get_ml_1m_dataset():
    return (
        FactorizationMachineDataset(users, movies, train_ratings),
        FactorizationMachineDataset(users, movies, validation_ratings),
    )
```

```{code-cell} ipython3

```

```{code-cell} ipython3
class FactorizationMachine(pl.LightningModule):
    def __init__(self, num_inputs, num_factors):
        super(FactorizationMachine, self).__init__()
        self.embedding = nn.Parameter(
            torch.randn(num_inputs, num_factors), requires_grad=True
        )
        self.linear_layer = nn.Linear(num_inputs, 1, bias=True)

    def forward(self, x):
        out_1 = torch.matmul(x, self.embedding).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.embedding.pow(2)).sum(1, keepdim=True)

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.linear_layer(x)
        out = out_inter + out_lin

        return torch.clip(out.squeeze(), min=-2, max=2)

    def training_step(self, batch, batch_idx):
        inputs, rating = batch
        rating = rating.to(torch.float32)
        output = self.forward(inputs)
        loss = F.mse_loss(rating, output)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.2, weight_decay=1e-3
        )  # learning rate
        return optimizer
```

```{code-cell} ipython3
training_data, validation_data = get_ml_1m_dataset()
batch_size = 1024
n_factors = 30
num_workers = min(batch_size, 14)
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
validation_dataloader = DataLoader(
    validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

model = FactorizationMachine(
    num_inputs=training_data.input_dim, num_factors=n_factors
)
trainer = pl.Trainer(gpus=1, max_epochs=30)
trainer.fit(model, train_dataloader)
print("Validation loss")
def eval_model(model, train_dataloader):
    loss = 0
    for inputs, rating in train_dataloader:
        pred = model(inputs)
        loss += F.mse_loss(pred, rating) ** 0.5
    avg_loss = loss / len(train_dataloader)
    print(f"avg rmse: {avg_loss}")
    
print("Validation dataset")
eval_model(model, validation_dataloader)
```

```{code-cell} ipython3

```
