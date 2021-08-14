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

(sec_crossing)=
# Crossing

Chúng ta thường quen với việc xử lý các đặc trưng một cách riêng biệt từ các trường dữ liệu tương ứng. Thực tế thì các trường dữ liệu có thể có mối quan hệ với nhau nhưng các mô hình machine learning *đơn giản* khó có thể *hình dung* ra được. Những mối quan hệ đó thường được khám phá ra bởi các nhà khoa học dữ liệu hoặc dựa trên kiến thức miền (_domain knowledge_).

Lấy một ví dụ nhỏ với bài toán dự đoán giá nhà California, ở đây kinh độ và vĩ độ là hai trường dữ liệu độc lập. Nếu tách biệt hai đặc trưng được tạo bởi hai trường dữ liệu này, mô hình có thể học được một tính chất rằng những vùng có cùng kinh độ hoặc vĩ độ sẽ có giá nhà gần với nhau. Điều này rõ ràng không đúng. Tuy nhiên, nếu có các thông tin về cả kinh độ và vĩ độ trong cùng một giá trị, mô hình sẽ học được những thông tin hữu ích hơn.

## Đặc trưng chéo

Đặc trưng chéo (_feature crossing_) có thể giải quyết được vấn đề này. Đặc trưng chéo thể hiện những sự kiện xảy ra đồng thời ở các đặc trưng khác và là một đặc trưng hạng mục.

Xét ví dụ dưới đây với một tập dữ liệu có ba đặc trưng `col1`, `col2` và `col3`:

```{code-cell} ipython3
import typing
import pandas as pd

df = pd.DataFrame(
    data={
        "col1": ["A", "B", "C", "A", "A"],
        "col2": ["x", "x", "y", "x", "z"],
        "col3": [1, 3, 2, 1, 2],
    }
)
df
```

Dưới đây là ví dụ về cách tạo đặc trưng chéo dựa trên: (i) hai cột đầu tiên và (ii) cả ba cột của DataFrame `df`:

```{code-cell} ipython3
from functools import partial


def add_cross(df: pd.DataFrame, cols: typing.List[str]) -> pd.DataFrame:
    """Add an column to the original dataframe as a cross feature.

    Args:
        df: input dataframe.
        cols: a list of columns in df that are used to create the new cross feature.

    Returns:
        A new dataframe with the new cross feature.
    """
    cross_col = "_X_".join(cols)

    def cross_value(x):
        return "_X_".join(str(x[col]) for col in cols)

    df[cross_col] = df.apply(cross_value, axis=1)
    return df


first_cross = ["col1", "col2"]
second_cross = ["col1", "col2", "col3"]
df = add_cross(df, first_cross)
df = add_cross(df, second_cross)
df
```

Bạn có thể đặt tên bất kỳ cho đặc trưng chéo, miễn là nó không trùng tên với các đặc trưng khác. Như một quy ước, tên của cột đặc trưng chéo có thể được tạo bằng cách nối tên của các đặc trưng thành phần bởi chuỗi `"_X_"`, dấu `X` thể hiện cho việc _cross_ các đặc trưng.

Tương tự, giá trị của các đặc trưng chéo có thể được quy ước là các chuỗi được tạo bởi cách nối các chuỗi thể hiện giá trị của các cột thành phần. Bạn có thể có cách nối các giá trị này khác; tuy nhiên, cần đảm bảo rằng nếu các cột thành phần có giá trị giống nhau.

## Thảo luận

1. Về lý thuyết, các đặc trưng cấu thành đặc trưng chéo có thể là đặc trưng số. Tuy nhiên, việc này cần tránh vì có thể đặc trưng số có rất nhiều giá trị khác nhau và cả những giá trị chưa biết không xuất hiện trong tập huấn luyện. Nếu muốn sử dụng các đặc trưng số để tạo đặc trưng chéo, ta có thể chia đặc trưng số vào một lượng nhỏ các _bin_ được định nghĩa trước. Việc này giúp đảm bảo đặc trưng chéo (dạng hạng mục) không có quá nhiều giá trị khác nhau.

2. Số lượng giá trị phân biệt của đặc trưng chéo có thể sẽ rất lớn nếu các đặc trưng thành phần cũng có số phần tử phân biệt lớn. Khi đó, đặc trực chéo sẽ được kết hợp với kỹ thuật {ref}`sec_hashing` để giảm số lượng phần tử riêng biệt. Cách làm này có thể gây ra xung đột hash nhưng vẫn mang lại hiệu quả cao trong nhiều trường hợp.

3. Việc chọn ra các tập đặc trưng nào để làm đặc trưng chéo có thể sẽ mất nhiều thời gian và công sức trong việc thí nghiệm. Năm 2017, một nhóm nghiên cứu ở Google đã đề xuất một phương pháp có tên "Deep and Cross Network" giúp tạo ra các đặc trưng chéo một cách tự động. Bạn đọc có thể tìm hiểu thêm trong mục Tài liệu tham khảo.


Các bạn có thể sẽ gặp kỹ thuật tạo đặc trưng chéo trong những bài toán cụ thể ở phần sau của cuốn sách.

## Tài liệu tham khảo

[Feature Crosses - Machine Learning crash course](https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture)

[Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)

```{code-cell} ipython3

```
