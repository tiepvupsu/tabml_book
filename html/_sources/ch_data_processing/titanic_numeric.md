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

# Thông tin dạng số
Để có cái nhìn nhanh về thống kê của mỗi trường thông tin dạng số, phương thức `describe()` có thể được sử dụng:

```{code-cell} ipython3
import pandas as pd
df_train = pd.read_csv("../data/titanic/train.csv")
df_train.describe()
```

Một vài quan sát với **tập huấn luyện** này:

* `"PassengerID", "Pclass"` mặc dù là các thông tin dạng hạng mục, chúng vẫn được liệt kê ở đây vì khi không chỉ định cụ thể, các trường thông tin mà toàn bộ các giá trị có thể chuyển đổi về số được coi là thông tin dạng số.

* Ở mỗi trường thông tin, các thống kê được chỉ ra cho các giá trị trong trường đó là:
    * `count`: số lượng phần tử _không bị khuyết_,
    * `mean`: giá trị trung bình,
    * `std`: phương sai của,
    * `min`: giá trị nhỏ nhất,
    * `max`: giá trị lớn nhất,
    * `50%`: trung vị -- giá trị mà ở đó có đúng một nửa số phần tử trong cột có giá trị nhỏ hơn hoặc bằng nó.
    * `25%`: trung vị của các giá trị từ `min` tới `50%`, tức có đúng 25% số phần tử trong cột có giá trị nhỏ hơn hoặc bằng nó,
    * `75%`: trung vị của các giá trị từ `50%` tới `max`, tức có đúng 75% số phần tử trong cột có giá trị nhỏ hơn hoặc bằng nó,
    
* Với cột `Survived`, giá trị trung bình trong cột là `0.384`. Đây là cột _nhãn_ mà mô hình cần dự đoán. Cột này chỉ mang các giá trị 0 và 1 nên ta có thể nói rằng 38.4% giá trị trong cột bằng 1. Việc này chứng tỏ dữ liệu tương đối cân bằng giữa hai lớp 0 và 1.

* Với cột `Age`, ta thấy rằng `count = 714` và nhỏ hơn số lượng phần từ ở các cột còn lại (891). Việc này chứng tỏ có tới 891 - 714 = 177 mẫu dữ liệu có `Age` bị khuyết. Người nhỏ nhất trên tàu mới chỉ 0.42 tuổi, trong khi người nhiều tuổi nhất đã 80.

* Với cột `Sibsp`, số lượng anh chị em hoặc vợ/chồng nhiều nhất với một hành khách là 8, nhưng có tới 75% số hành khách có nhiều nhất 1 anh chị em hoặc vợ/chồng đi cùng. Việc này chứng tỏ phân bố của dữ liệu này khá lệch (_skewed_).

* Cột `Parch` cũng bị lệch tương tự khi có một hành khách có tới 6 con/bố mẹ trong khi 75% số hành khách không có con/bố mẹ đi cùng.

* Cột `Fare` cũng khá lệch khi trung binh là 32 trong khi trung vị chỉ là 14 và giá tri lớn nhất lên tới 512. Những hành khách với giá vé bằng 0 khả năng nằm trong thủy thủ đoàn.

```{note}
Khi một cột có những giá trị bị khuyết, thống kê của cột được tính dựa trên các giá trị còn lại.
```

Với **tập kiểm tra**:

```{code-cell} ipython3
df_test = pd.read_csv("../data/titanic/test.csv")
df_test.describe()
```

Một vài quan sát:

* Số lượng phần tử trong tập này là 418 (bằng `count` trong cột `PassengerID`).

* Các cột `Age, Fare` có nhiều giá trị bị khuyết. Như vậy, mặc dù tập huấn luyện không có giá trị `Fare` nào bị khuyết, tập kiểm tra có một hàng bị khuyết giá trị này.

* Các thống kê trong các cột `Age, SibSp, Parch` và `Fare` tương đối nhất quán với tập huấn luyện.
