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

import tabml.datasets


GLOBAL_SEED = 42  # number of life
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
```

```{code-cell} ipython3
# Split train, val for ratings

users, movies, ratings = tabml.datasets.download_movielen_1m()
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
# user_featuers
gender_index_by_name = {"M":0, "F": 1}
age_index_by_name = {1: 0, 18: 1, 25: 2, 35:3, 45: 4, 50: 5, 56:6}
occupations = [
"other",
"academic/educator",
"artist",
"clerical/admin",
"college/grad student",
"customer service",
"doctor/health care",
"executive/managerial",
"farmer",
"homemaker",
"K-12 student",
"lawyer",
"programmer",
"retired",
"sales/marketing",
"scientist",
"self-employed",
"technician/engineer",
"tradesman/craftsman",
"unemployed",
"writer",
]
occupation_index_by_name = {name: index for index, name in enumerate(occupations)}

num_users = len(users)
gender_offset = num_users
age_offset = gender_offset + len(gender_index_by_name)
occupation_offset = age_offset + len(age_index_by_name)

# user_features = np.zeros((num_users, occupation_offset + len(occupations)))
# for index in range(num_users):
#     user_features[index, index] = 1
#     # gender
#     gender_index = gender_index_by_name[users["Gender"][index]]
#     user_features[index, gender_offset + gender_index] = 1
    
#     # age
#     age_index = age_index_by_name[users["Age"][index]]
#     user_features[index, age_offset + age_index] = 1

#     # occupation
#     occupation_index = users["Occupation"][index]
#     user_features[index, occupation_offset + occupation_index] = 1


user_features = []
for index in range(num_users):
    gender_index = gender_index_by_name[users["Gender"][index]] + gender_offset
    age_index = age_index_by_name[users["Age"][index]] + age_offset
    occupation_index = users["Occupation"][index] + occupation_offset
    user_features.append([index, gender_index, age_index, occupation_index])
user_features[0]    
```

```{code-cell} ipython3
# build moive_features
# a (len(movie), len(movie) + len(genres)) binary matrix
movie_offset = occupation_offset + len(occupation_index_by_name)

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
genre_index_by_name = {name:i for i, name in enumerate(genres)}
num_movies = len(movies)

movie_features = []
for i, movie_genres in enumerate(movies["Genres"]):
    movie_feature = [movie_offset + i]
    for genre in movie_genres.split("|"):
        genre_index = genre_index_by_name[genre] + num_movies + movie_offset
        movie_feature.append(genre_index)
    movie_features.append(movie_feature)
print(movie_features[0])
total_inputs = movie_offset + num_movies + len(genres)
print(total_inputs)
```

```{code-cell} ipython3
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_MOVIES = len(movies)
NUM_USERS = len(users)
padding_idx = total_inputs

class FactorizationMachineDataset(Dataset):
    def __init__(self,  rating_df):
        self.rating_df = rating_df
        self.max_size = 5 + len(genres)  # 4 for user feature + movie index + genres

    def __len__(self):
        return len(self.rating_df)
    
    def __getitem__(self, index):
        user_index = user_index_by_id[self.rating_df["UserID"].iloc[index]]
        movie_index = movie_index_by_id[self.rating_df["MovieID"].iloc[index]]
        rating = self.rating_df["Rating"].iloc[index]
        user_feature = user_features[user_index]
        movie_feature = movie_features[movie_index]
        padding_size = self.max_size - len(user_feature) - len(movie_feature)
        feature = user_feature + movie_feature + [padding_idx]*padding_size
        return torch.IntTensor(feature), rating

def get_ml_1m_dataset():
    return (
        FactorizationMachineDataset(train_ratings),
        FactorizationMachineDataset(validation_ratings),
    )
```

```{code-cell} ipython3
from pytorch_lightning.loggers import TensorBoardLogger

LR = 5e-4
WEIGHT_DECAY = 5e-5


class FactorizationMachine(pl.LightningModule):
    def __init__(self, num_inputs, num_factors):
        super(FactorizationMachine, self).__init__()
        self.embedding = nn.Embedding(num_inputs + 1, num_factors, padding_idx=padding_idx)
#         self.embedding.weight.data.uniform_(-.1, .1)
        torch.nn.init.xavier_normal_(self.embedding.weight.data, gain=1e-3)
        self.linear_layer = nn.Embedding(num_inputs+1, 1, padding_idx=padding_idx)
        self.bias = nn.Parameter(data=torch.rand(1))
    

    def forward(self, x):
        emb = self.embedding(x)
        pow_of_sum = emb.sum(dim=1, keepdim=True).pow(2).sum(dim=2)
        sum_of_pow = emb.pow(2).sum(dim=1, keepdim=True).sum(dim=2)
        out_inter = 0.5 * (pow_of_sum - sum_of_pow)
        out_lin = self.linear_layer(x).sum(1)
        out = out_inter + out_lin + self.bias

#         return torch.clip(out.squeeze(), min=-2, max=2)
#         return torch.sigmoid(out.squeeze())

        return out.squeeze()

    def training_step(self, batch, batch_idx):
        inputs, rating = batch
        rating = rating.to(torch.float32)
        if self.current_epoch < 30:
            output = self.forward(inputs)
        else:
            output = torch.clip(self.forward(inputs), min=-2, max=2)
        loss = F.mse_loss(rating, output)
        self.log("batch_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, rating = batch
        rating = rating.to(torch.float32)
        output = torch.clip(self.forward(inputs), min=-2, max=2)
        loss = F.mse_loss(rating, output)
        self.log("batch_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        #         self.logger.experiment.add_scalars("Loss", {"Train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars(
            "RMSE", {"Train": avg_loss ** 0.5}, self.current_epoch
        )
        epoch_dict = {"loss": avg_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        #         self.logger.experiment.add_scalars("Loss", {"Val": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars(
            "RMSE", {"Val": avg_loss ** 0.5}, self.current_epoch
        )
        epoch_dict = {"loss": avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        return optimizer


n_factors = 100
batch_size = 1024
logger = TensorBoardLogger(
    "fm_2_tb_logs", name=f"ilr{LR}_wd{WEIGHT_DECAY}_emb{n_factors}_b{batch_size}"
)

training_data, validation_data = get_ml_1m_dataset()


num_workers = min(batch_size, 14)
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
validation_dataloader = DataLoader(
    validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

num_inputs = total_inputs

# model = FactorizationMachine(num_inputs=training_data.input_dim, num_factors=n_factors)
model = FactorizationMachine(num_inputs=num_inputs, num_factors=n_factors)
trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)

trainer.fit(model, train_dataloader, validation_dataloader)
# print("Validation loss")
```

```{code-cell} ipython3
embs_arr = model.state_dict()['embedding.weight'].detach().numpy()
```

```{code-cell} ipython3
embs_arr.shape
```

```{code-cell} ipython3
# get movie genres embedding
num_movies + movie_offset
```

```{code-cell} ipython3
genre_embs = embs_arr[num_movies + movie_offset: num_movies + movie_offset + len(genres), :]
```

```{code-cell} ipython3

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def vis(arr, labels):
    assert arr.shape[0] == len(labels), arr.shape
    # pca to 2d 
#     X2 = PCA(n_components=2).fit_transform(arr)
    X2 = TSNE(n_components=2).fit_transform(arr)
    plt.figure(figsize=(10, 10))
    plt.scatter(X2[:, 0], X2[:, 1])
    for i in range(arr.shape[0]):
        plt.text(X2[i, 0] + .01, X2[i, 1] + 0.01, labels[i])
    
vis(genre_embs, genres)
```

```{code-cell} ipython3
len(genres)
```

```{code-cell} ipython3
# find movies with single genres

movie_genres = movies["Genres"]
movie_inds_one_genre = [i for i, gs in enumerate(movie_genres) if gs in ('Drama', 'Comedy', 'Horror')]
# movie_inds_one_genre = [i for i, gs in enumerate(movie_genres) if '|' not in gs]
```

```{code-cell} ipython3
movie_embs = embs_arr[movie_offset: movie_offset + num_movies, :]
movie_subset_embs = movie_embs[movie_inds_one_genre, :]
```

```{code-cell} ipython3
movie_subset_embs.shape
# movie_2d = PCA(n_components=2).fit_transform(movie_subset_embs)
movie_2d = TSNE(n_components=2).fit_transform(movie_subset_embs)
```

```{code-cell} ipython3
movie_df = pd.DataFrame(
    data={
        "x": movie_2d[:, 0],
        "y": movie_2d[:, 1],
        "genre": movie_genres[movie_inds_one_genre],
    }
)
import seaborn as sns
plt.figure(figsize=(20, 20))
ax = sns.scatterplot(x="x", y="y", hue='genre',data=movie_df)
```

```{code-cell} ipython3
from collections import Counter
Counter(movie_genres[movie_inds_one_genre])
```

```{code-cell} ipython3
model.state_dict()['linear_layer.weight'].detach().numpy()
```

```{code-cell} ipython3
model.state_dict()['bias'].detach().numpy()
```

```{code-cell} ipython3

```
