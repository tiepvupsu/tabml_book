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

# Pipeline đơn giản cho cuộc thi Titanic

Trong trang này, tôi xin giới thiệu một pipeline hoàn thiện rất đơn giản để có thể tạo ra một bài nộp lên Kaggle và tính điểm. Tôi xin không đi sâu vào từng dòng lệnh mà muốn dùng ví dụ này để giúp các bạn có cái nhìn bao quát về một pipeline hoàn thiện.

Toàn bộ mã nguồn của pipeline này có thể được tìm thấy [tại đây](https://github.com/tiepvupsu/tabml_book/tree/main/book/ch_intro/titanic_pipeline.py).


Bước đầu tiên luôn luôn là import những thư viện cần thiết.

```{code-cell} ipython3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
```

Load dữ liệu:

```{code-cell} ipython3
titanic_path = "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/titanic/"
df_train_full = pd.read_csv(titanic_path + "train.csv")
df_test = pd.read_csv(titanic_path + "test.csv")

# data_dir = Path("../data/titanic")
# df_train_full = pd.read_csv(data_dir / "train.csv")
# df_test = pd.read_csv(data_dir / "test.csv")
```

Tiếp theo, ta cần bỏ đi những cột có quá nhiều giá trị bị khuyết. Sử dụng các giá trị này thường không mang lại sự cải thiện cho mô hình.

```{code-cell} ipython3
df_train_full.drop(columns=["Cabin"])
df_test.drop(columns=["Cabin"]);
```

Trước khi đi vào bước xây dựng đặc trưng, ta cần phân chia dữ liệu huấn luyện/kiểm định. Ở đây, 10% ngẫu nhiên của dữ liệu có nhãn ban đầu được tách ra làm dữ liệu kiểm định (_validation data_), 90% còn lại được giữ làm dữ liệu huấn luyện (_training data_). Cột `Survived` là cột nhãn được tách ra làm một biến riêng chứa nhãn:

```{code-cell} ipython3
df_train, df_val = train_test_split(df_train_full, test_size=0.1)
X_train = df_train.copy()
y_train = X_train.pop("Survived")

X_val = df_val.copy()
y_val = X_val.pop("Survived")
```

Sau khi đã phân chia dữ liệu, ta cần xử lý tạo các đặc trưng cho mô hình. Các đặc trưng hạng mục và đặc trưng số cần có những cách xử lý khác nhau. Với mỗi loại đặc trưng, ta cần hai bước nhỏ: (i) làm sạch dữ liệu và (ii) biến dữ liệu về dạng số phù hợp với đầu vào của mô hình. Trước hết là với dữ liệu dạng hạng mục, ở đây, `cat_transformer` được áp dụng lên cả ba đặc trưng hạng mục:

```{code-cell} ipython3
cat_cols = ["Embarked", "Sex", "Pclass"]
cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)
```

Tiếp theo, ta áp dụng `num_transformer` lên hai đặc trưng số:

```{code-cell} ipython3
num_cols = ["Age", "Fare"]
num_transformer = Pipeline(
    steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler())]
)
```

Kết hợp hai bộ xử lý đặc trưng lại để có một bộ xử lý đặc trưng hoàn thiện. Lớp `ColumnTransformer` trong scikit-learn giúp kết hợp các _transformers_ lại:

```{code-cell} ipython3
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)
```

Cuối cùng, ta kết hợp bộ xử lý đặc trưng `preprocessor` với một bộ phân loại đơn giản hay được sử dụng với dữ liệu dạng bảng là `RandomForestClassifier` để được một pipeline `full_pp` hoàn chỉnh bao gồm cả xử lý dữ liệu và mô hình. `full_pp` được _fit_ với dữ liệu huấn luyện `(X_train, y_train)` sau đó được dùng để áp dụng lên dữ liệu kiểm định:

```{code-cell} ipython3
# Full training pipeline
full_pp = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)

# training
full_pp.fit(X_train, y_train)

# training metric
y_train_pred = full_pp.predict(X_train)
print(f"Accuracy score on train data: {accuracy_score(list(y_train), list(y_train_pred)):.2f}")

# validation metric
y_pred = full_pp.predict(X_val)
print(f"Accuracy score on validation data: {accuracy_score(list(y_val), list(y_pred)):.2f}")
```

Như vậy, cả _hệ thống_ này cho độ chính xác 98% trên tập huấn luyện và 83% trên tập kiểm định. Sự chênh lệch này chứng tỏ đã xảy ra hiện tượng [_overfitting_](https://machinelearningcoban.com/2017/03/04/overfitting/). Tạm gác vấn đề này sang một bên, chúng ta sử dụng hệ thống vừa thu được để đưa ra dự đoán cho dữ liệu của cuộc thi.

```{code-cell} ipython3
# make submission
preds = full_pp.predict(df_test)
sample_submission = pd.read_csv(titanic_path + "gender_submission.csv")
sample_submission["Survived"] = preds
sample_submission.to_csv("titanic_submission.csv", index=False)
```

Sau khi file nộp bài `titanic_submission.cssv` được tạo, ta có thể thử xem kết quả trên Leadboard của Kaggle.

```{code-cell} ipython3
%%capture
!kaggle competitions submit -c titanic -f titanic_submission.csv -m "simple submission"
```

Kết quả trên Leaderboard của cuộc thi cho bài nộp này là `0.74641`, không quá tệ cho một pipeline đơn giản.

```{code-cell} ipython3
:tags: [hide-input]

!rm titanic_submission.csv
```
