---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# EDA cho dữ liệu California Housing

_Nội dung trong site này được tham khảo rất nhiều từ chương "End-to-End Machine Learning Project" của cuốn [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)._

Chúng ta cùng làm quen với bộ dữ liệu California Housing.

Bộ dữ liệu này chỉ có một file:

```{code-cell} ipython3
!ls ../data/titanic
```

```{code-cell} ipython3
import pandas as pd
df_train = pd.read_csv("../data/titanic/train.csv")
df_test = pd.read_csv("../data/titanic/test.csv")

```

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

df_train0, df_val = train_test_split(df_train, test_size=.1)
```

```{code-cell} ipython3
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV


cat_cols = ['Embarked', 'Sex', 'Pclass']
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
])
```

```{code-cell} ipython3
num_cols = ['Age', 'Fare']
num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])
```

```{code-cell} ipython3
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
```

```{code-cell} ipython3
X_train = df_train0.drop(columns="Survived")
X_train.drop(columns=["Cabin"])
y_train = df_train0["Survived"]

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()
clf.fit(X_train, y_train);
```

```{code-cell} ipython3
X_val = df_val.copy()
X_val.drop(columns=["Cabin"])
y_val = X_val.pop("Survived")

y_pred = clf.predict(X_val)
```

```{code-cell} ipython3
from sklearn.metrics import accuracy_score

accuracy_score(list(y_val), list(y_pred))
```
