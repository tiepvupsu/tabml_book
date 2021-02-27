---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Kỹ thuật xử lý đặc trưng (WIP)

Với dữ liệu dạng bảng, việc tiền xử lý dữ liệu và xây dựng đặc trưng thường tốn nhiều thời gian hơn việc xây dựng mô hình.
Có một [thống kê](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/?sh=3ce816956f63) chỉ ra rằng,
80% thời gian của các nhà khoa học dữ liệu là dành cho việc chuẩn bị dữ liệu. Vì vậy, cuốn sách này sẽ bắt đầu với các bài viết về việc chuẩn bị dữ liệu.

## Dữ liệu dạng CSV

Việc lưu trữ dữ liệu dạng bảng trong công nghiệp đòi hỏi nhiều chức năng quan trọng như tốc độ ghi đọc, tổ chức bộ nhớ.
Trong phạm vi cuốn sách này, chúng ta sẽ làm việc với các cơ sở dữ liệu ở dạng [CSV](https://en.wikipedia.org/wiki/Comma-separated_values).
Ghi và đọc dữ liệu từ các file csv sẽ bị hạn chế về tốc độ; tuy nhiên, việc trình bày những kỹ thuật xử lý đặc trưng cũng như minh họa kết quả dễ dàng hơn rất nhiều nhờ nhiều thư viện có sẵn.

```{code-cell} ipython3
%%capture
!rm -rf nb_data/titanic; mkdir -p nb_data/titanic
!pip install kaggle;
!kaggle competitions download -c titanic -p nb_data/titanic;

```

Unzip `titanic.zip`


```{code-cell} ipython3
!cd nb_data/titanic; unzip titanic.zip; cd ../../
!cat nb_data/titanic/train.csv | head -10
```


Sau khi giải nén, thư mục `nb_data/titanic` có ba file `.csv` như trên. Trong ba file này, `train.csv` là dữ liệu dược dùng để huấn luyện, `test.csv` là dữ liệu cần dự đoán, và `gender_submision.csv` là file nộp kết quả mẫu.


```{code-cell} ipython3
import pandas as pd
df_train = pd.read_csv("nb_data/titanic/train.csv")
df_train.head()
```

(sec_clean_data)=
## Làm sạch dữ liệu

Tại sao cần làm sạch dữ liệu?
Như đã đề cập, dữ liệu dạng bảng thường có nhiều trường thông tin bị khuyết và bị nhiễu.
Việc đầu tiên trước khi bắt tay vào xây dựng các đặc trưng là việc xử lý nhưng thông tin bị khuyết và nhiễu này.

### Xử lý thông tin bị khuyết



### Xử lý các cột bị nhiễu

## Đặc trưng hạng mục

(sec_one_hot)=
### Mã hóa one-hot

### Hashing

### Crossing

(sec_embedding)=
### Embedding


## Đặc trưng dạng số

### Chuẩn hóa

### Bucketizing

### Dữ liệu dạng chuỗi thời gian

## Pipeline dữ liệu

