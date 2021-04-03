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

(sec_outlier_processing)=
# Xử lý các giá trị ngoại lệ

Giá trị ngoại lệ [^1] (_outliers_) trong dữ liệu là gì?

Với dạng số, dữ liệu ngoại lệ có thể là một giá trị phi thực tế như số tuổi âm, hoặc một giá trị khác xa với phần còn lại của các giá trị trong trường đó.
Với dạng hạng mục, dữ liệu ngoại lệ có thể là một giá trị phi thực tế như một hạng mục nằm ngoài những khả năng có thể xảy ra như một địa danh không có trên bản đồ.
Các giá trị có tần xuất xảy ra vô cùng thấp trong một cột dữ liệu cũng có _khả năng_ [^2] là một giá trị ngoại lệ.

(sec_numeric_outliers)=
## Dữ liệu số

### Ảnh hưởng lên chất lượng mô hình
Các phép biến đổi số học tương đối nhạy cảm với các giá trị ngoại lệ (quá lớn hoặc quá nhỏ). Đặc biệt, nếu ta muốn xây dựng đặc trưng dựa trên trung bình của một cột, các giá trị ngoại lệ có thể làm thay đổi trung bình đáng kể. Ví dụ, ngôi làng A có 100 ngôi nhà, trong đó 99 ngôi nhà có thu nhập 1 triệu/tháng. Ngôi nhà còn lại của một anh đại gia có thu nhập 3 tỉ/tháng. Như vậy "thu nhập bình quân" của ngôi làng là gần 33 triệu/tháng. Một ngôi làng B khác có mọi nhà đều thu nhập vào khoảng 5-10 triệu/tháng. Nếu một công ty muốn mở cửa hàng tạp hóa dựa trên thu nhập bình quân đầu người của mỗi làng thì rõ ràng ngôi làng A được đánh giá cao hơn mặc dù trên thực tế, ngôi làng B có mức sống cao hơn.

Các giá trị ngoại lệ cũng ảnh hưởng lớn đến chất lượng mô hình machine learning. Xét ví dụ đơn giản dưới đây.

Có một bảng dữ liệu với chiều cao được lưu trong cột `height` và cân nặng được lưu trong `weight`. Giả sử cột `height_2` là một phiên bản của `height` với chỉ một sự khác biệt ở chiều cao của người đầu tiên là 110cm thay vì 147cm. Cột `weight_2` chỉ khác cột `weight` ở dòng thứ hai với cân nặng 90kg thay vì 50 kg. Dòng đầu tiên trong `height_2` và dòng thứ hai trong `weight_2` có thể coi là các giá trị ngoại lệ. Các giá trị này có thể do sai số ghi chép hoặc thực sự đó là dữ liệu thật.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd
df = pd.DataFrame(
    data={
        "height": [147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183],
        "weight": [49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68],
        "height_2": [110, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183],
        "weight_2": [49, 90, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68],
    }
)
df
```

Giả sử ta cần dùng bộ dữ liệu này để xây dựng một mô hình dự đoán cân nặng theo chiều cao. Ta có thể thấy rằng cân nặng _thường_ tỉ lệ thuận với chiều cao nên mô hình hồi quy tuyến tính sẽ phù hợp cho công việc này. Hình vẽ dưới đây thể hiện kết quả mà mô hình hồi quy tuyến tính học được trong ba trường hợp:

* TH1 (trái): dùng dữ liệu trong cột `height` làm đầu vào, trong cột `weight` làm nhãn.
* TH2 (giữa): dùng dữ liệu trong cột `height_2` làm đầu vào, trong cột `weight` làm nhãn.
* TH3 (phải): dùng dữ liệu trong cột `height` làm đầu vào, trong cột `weight_2` làm nhãn.

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression


def fit_linear_regression_and_visualize(
    df: pd.DataFrame, input_col: str, label_col: str
):
    # fit the model by Linear Regression
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(df[[input_col]], df[label_col])
    w1 = lin_reg.coef_
    w0 = lin_reg.intercept_

    # visualize
    plt.plot(df[input_col], df[label_col], "ro", label="data")
    plt.axis([105, 190, 45, 75])
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.ylim(45, 95)
    plt.plot([105, 190], [w1 * 105 + w0, w1 * 190 + w0], label="fitted line")
    plt.legend()


plt.figure(figsize=(17, 6))
plt.subplot(1, 3, 1)
fit_linear_regression_and_visualize(df, input_col="height", label_col="weight")

plt.subplot(1, 3, 2)
fit_linear_regression_and_visualize(df, input_col="height_2", label_col="weight")

plt.subplot(1, 3, 3)
fit_linear_regression_and_visualize(df, input_col="height", label_col="weight_2")
```

Các điểm màu đỏ thể hiện các điểm dữ liệu với trục hoành là cân nặng và trục tung là chiều cao. Đường thẳng màu xanh là đường thằng mà mô hình hồi quy tuyến tính học được. Ta có thể thấy rằng đường màu xanh trong hình bên trái khá khớp dữ liệu, trong khi hai đường thẳng ở hai trường hợp còn lại bị lệch đi khá nhiều dù chỉ có một điểm dữ liệu ngoại lệ trong mỗi trường hợp.

Như vậy, với dữ liệu rất đơn giản này, dữ liệu ngoại lệ dù ở đầu vào mô hình hay nhãn đều mang lại kết quả không tốt.

### Xác định và xử lý các điểm ngoại lệ

Có hai nhóm các giá trị ngoại lệ:

* Các giá trị không nằm trong miền xác định của dữ liệu. Ví dụ, tuổi, thu nhập hay khoảng cách không thể là số âm.

* Các giá trị có khả năng xảy ra nhưng xác suất rất thấp. Ví dụ, 120 tuổi, thu nhập 1 triệu đô la/tháng. Những giá trị này có khả năng xảy ra nhưng thực sự hiếm có.

Nhìn chung, chúng ta luôn có thể xóa bỏ cột hoặc hàng có dữ liệu ngoại lệ. Nếu xóa bỏ cột, ta có thể lãng phí rất nhiều các giá trị không phải ngoại lệ ở các hàng khác. Nếu xóa bỏ hàng, chúng ta cần lưu ý tới cách xử lý với dữ liệu mới. Tức là nếu một điểm dữ liệu mới cũng có giá trị ngoại lai thì sao? Ta không thể bỏ không dự đoán điểm đó mà phải có cách biến đổi dữ liệu ngoại lai này về những giá trị hợp lý hơn.

Với dữ liệu thuộc nhóm thứ nhất, ta có thể thay nó bằng `nan` và coi như một giá trị bị khuyết. Đôi khi những giá trị bị khuyết được mã hóa bằng một giá trị đặc biệt không nằm trong miền giá trị khả dĩ của dữ liệu. Khi coi chúng là giá trị bị khuyết, ta có thể xử lý tiếp như trong `ref{sec_missing_data}`.

Với dữ liệu thuộc nhóm thứ hai, người ta thường dùng phương pháp chặn trên hoặc chặn dưới (_clipping_ hay _capping_). Tức là khi một giá trị quá lớn hoặc quá nhỏ, ta đưa nó về giá trị lớn nhất/nhỏ nhất được coi là những điểm bình thường. Ví dụ với một giá trị của tuổi là 120, ta có thể đưa nó về 70 và giả sử như điểm dữ liệu này có những đặc tính chung của "người cao tuổi". Một điểm đáng lưu ý là việc chọn giá trị lớn nhất/nhỏ nhất cũng tùy thuộc vào dữ liệu. Nếu dữ liệu chỉ toàn bao gồm người cao tuổi tử 65 trở lên thì rõ ràng chặn trên bởi 70 là không hợp lý vì 70 vẫn là quá trẻ trong bộ dữ liệu này.

Vậy làm thế nào để chọn những giá trị lớn nhất, nhỏ nhất đó?

Cách phổ biến nhất là sử dụng {ref}`sec_boxplot`. Box plot vừa giúp xác định xem dữ liệu có điểm ngoại lệ không, vừa giúp tìm ra ngưỡng lớn nhất và nhỏ nhất để làm điểm cắt.

**Box plot**

Để minh họa cho cách sử dụng box plot, ta sẽ sử dụng bộ dữ liệu California Housing

```{code-cell} ipython3
import pandas as pd

df = pd.read_csv("../data/california_housing/housing.csv")
df.head()
```

Dưới đây là histogram và box plot của cột `total_rooms`. Ở đâ, box plot được vẽ ở dạng nằm ngang để so sánh với histogram.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
df[["total_rooms"]].hist(bins=50, ax=axes[0]);
df[["total_rooms"]].boxplot(ax=axes[1], vert=False);
```

Từ histogram ta thấy dữ liệu bị lệch phải (có điểm ngoại lệ lệch nhiều về bên phải, hoặc "đuôi" của histogram nằm ở bên phải). Từ boxplot ta thấy có khá nhiều điểm được coi là ngoại lệ.
Các điểm ngoại lệ có thể được xử lý bằng cách _clip_ về giá trị cực tiểu và cực đại của Box plot. Bộ xử lý này có thể được triển khai dưới dạng sklearn API như sau:

```{code-cell} ipython3
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin


def find_boxplot_boundaries(
    col: pd.Series, whisker_coeff: float = 1.5
) -> Tuple[float, float]:
    """Findx minimum and maximum in boxplot.

    Args:
        col: a pandas serires of input.
        whisker_coeff: whisker coefficient in box plot
    """
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - whisker_coeff * IQR
    upper = Q3 + whisker_coeff * IQR
    return lower, upper


class BoxplotOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, whisker_coeff: int = 1.5):
        self.whisker = whisker_coeff
        self.lower = None
        self.upper = None

    def fit(self, X: pd.Series):
        self.lower, self.upper = find_boxplot_boundaries(X, self.whisker)
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)
```

Áp dụng lại vào dữ liệu của cột `total_rooms` ta có histogram và boxplot mới như sau:

```{code-cell} ipython3
clipped_total_rooms = BoxplotOutlierRemover().fit_transform(df["total_rooms"])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
clipped_total_rooms.hist(bins=50, ax=axes[0])
clipped_total_rooms.to_frame().boxplot(ax=axes[1], vert=False);
```

Sau khi clip dữ liệu theo cực tiểu và cực đại của box plot, ta thấy rằng dữ liệu _đỡ_ bị lệch đi. Box plot cũng chó thấy không còn điểm dữ liệu ngoại lệ nào.

```{margin}
Sau khi clip dữ liệu bằng cực đại và cực tiểu của boxplot, dữ liệu mới luôn luôn không có điểm ngoại lệ. Điều này đạt được vì phép biến đổi clip không làm thay đổi tứ phân vị của dữ liệu. Khoảng "hợp lệ" của boxplot trước và sau clip không thay đổi.
```


## Dữ liêu hạng mục

[^1]: Đôi khi được gọi là "ngoại lai".

[^2]: Giá trị đặc biệt này cũng có thể mang lại nhiều thông tin cho việc dự đoán. Cần kiểm tra kỹ mối tương quan giữa cột dữ liệu tương ứng và cột nhãn.
