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
from sklearn.model_selection import train_test_split

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
# Generate genre vector for each movie
movies
```

```{code-cell} ipython3
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

import numpy as np
# build binary array for movie genres
movie_features = np.zeros((len(movies), len(genres)))
for i, movie_genres in enumerate(movies["Genres"]):
    for genre in movie_genres.split("|"):        
        genre_index = genre_index_by_name[genre]
        movie_features[i, genre_index] = 1
        
movie_features
```

```{code-cell} ipython3
import jdc
```

```{code-cell} ipython3
movie_index_by_id = {id: i for i, id in enumerate(movies["MovieID"])}
movie_index_by_id
```

```{code-cell} ipython3
from sklearn.linear_model import Ridge
from tqdm import tqdm


def train_user_model(user_id):
    user_ratings = train_ratings[train_ratings["UserID"] == user_id]
    movie_indexes = [
        movie_index_by_id[movie_id] for movie_id in user_ratings["MovieID"]
    ]
    train_data = movie_features[movie_indexes]
    train_label = user_ratings["Rating"]
    model = Ridge(alpha=0.1)
    model.fit(train_data, train_label)
    return model


# build model for each user
user_model_dict = {}
for user_id in tqdm(users["UserID"].unique()):
    user_model_dict[user_id] = train_user_model(user_id)
```

```{code-cell} ipython3
def predict(user_id, movie_id):
    movie_feature = movie_features[movie_index_by_id[movie_id]].reshape((1, -1))
    pred = user_model_dict[user_id].predict(movie_feature)
    return min(max(pred, 1), 5)
```

```{code-cell} ipython3
from sklearn.metrics import mean_squared_error

def eval_rmse(ratings: pd.DataFrame) -> float:
    predictions = np.zeros(len(ratings))
    for index, row in tqdm(enumerate(ratings.itertuples(index=False))):
        predictions[index] = predict(row[0], row[1])
    rmse = mean_squared_error(ratings["Rating"], predictions, squared=False)
    return rmse
    
print(f"RMSE train: {eval_rmse(train_ratings)}")
print(f"RMSE validation: {eval_rmse(validation_ratings)}")
    
```

```{code-cell} ipython3
# Sanity check one user

# user_id = users["UserID"].unique()[0]
user_id = 4547  # rated 15 movies
```

```{code-cell} ipython3
for genre, coef in zip(genres, user_model_dict[160].coef_):
    print("{:15s}: {:.3f}".format(genre, coef))
```

```{code-cell} ipython3
user_model_dict[user_id].intercept_
```

```{code-cell} ipython3
user_ratings = train_ratings[train_ratings["UserID"]==user_id]
user_ratings
```

```{code-cell} ipython3
user_ratings.reset_index().join(movie, on="MovieID", how="inner",rsuffix="_")
```

```{code-cell} ipython3
movies[movies["MovieID"].isin(user_ratings["MovieID"])]
```

```{code-cell} ipython3
movies
```

```{code-cell} ipython3

```
