import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def pipeline():
    # load data
    df_train = pd.read_csv("../data/titanic/train.csv")

    # remove row with many missing values
    df_train.drop(columns=["Cabin"])

    # split data
    df_train0, df_val = train_test_split(df_train, test_size=0.1)
    X_train = df_train0.copy()
    y_train = X_train.pop("Survived")

    X_val = df_val.copy()
    y_val = X_val.pop("Survived")

    # feature engineering for categorical features
    cat_cols = ["Embarked", "Sex", "Pclass"]
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # feature engineering for numeric features
    num_cols = ["Age", "Fare"]
    num_transformer = Pipeline(
        steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())]
    )

    # Full data engineering
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    # Full training pipeline
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
    )

    # training
    clf.fit(X_train, y_train)

    # validation
    y_pred = clf.predict(X_val)
    print(f"Accuracy score {accuracy_score(list(y_val), list(y_pred))}")


if __name__ == "__main__":
    pipeline()
