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

# Đặc trưng hạng mục (WIP)

Các thư viện hỗ trợ machine learning hầu hết chỉ chấp nhận dữ liệu đầu vào ở dạng số. Tuy nhiên, chúng ta đã thấy rằng với dữ liệu dạng bảng, rất thường xuyên các trường dữ liệu được lưu ở dạng hạng mục. Thậm chí, nhiều trường dữ liệu được lưu ở dạng số nhưng vẫn được coi là dạng hạng mục. Ví dụ, mã số người dùng có thể là bất cứ giá trị nào, miễn là không trùng lặp. Chúng có thể là các giá trị số 1, 2, 3, ... nhưng các giá trị này không nên được đưa trực tiếp vào mô hình.

Một điều cần được nhấn mạnh là một mô hình machine learning tốt là mô hình trả về kết quả đầu ra gần nhau khi dữ liệu đầu vào (ở dạng số) gần nhau. Mã người dùng, mã sản phẩm hay bất kỳ loại mã nào được đánh số theo thứ tự một cách ngẫu nhiên không thể coi là có độ tương đồng cao khi hai mã ở gần nhau. Ngay cả khi các mã được đánh theo một cách có chủ đích, chúng cũng chỉ gần nhau trong không gian một chiều. Thông tin có thể được định nghĩa là "gần nhau" trong không gian nhiều chiều hơn. Thêm một ví dụ khác, các ngày trong tuần giả sử được đánh số là 1 (chủ nhật), (thứ) 2, ..., (thứ) 7; thì ngày 1 và 2 là gần nhau nhưng ngày 1 và ngày 7 có ý nghĩa gần hơn vì cùng là cuối tuần. Xếp các ngày vào các điểm trên một hình tròn trong không gian hai chiều có thể mang lại nhiều giá trị hơn vì 1 gần với cả 7 và 2.

Như vậy, với dữ liệu dạng hạng mục, chúng ta không những cần phải đưa chúng về dạng số để các thuật toán có thể xử lý được mà còn phải đưa về những giá trị hợp lý trong không gian nhiều chiều để mang lại kết quả tốt. Những kỹ thuật này sẽ được đề cập trong các mục con dưới đây và các ví dụ về bài toán cụ thể trong các chương sau của cuốn sách.

(sec_one_hot)=
## Mã hóa one-hot

Cách truyền thống nhất để đưa dữ liệu hạng mục về dạng số là mã hóa one-hot. Trong cách mã hóa này, một "từ điển" cần được xây dựng chứa tất cả các giá trị khả dĩ của từng dữ liệu hạng mục. Sau đó mỗi giá trị hạng mục sẽ được mã hóa bằng một vector nhị phân với toàn bộ các phần tử bằng 0 trừ một phần tử bằng 1 tương ứng với vị trí của giá trị hạng mục đó trong từ điển.

Ví dụ, nếu ta có dữ liệu một cột là `"Sài Gòn", "Huế", "Hà Nội"` thì ta thực hiện các bước sau:

1. Xây dựng từ điển. Trong trường hợp này ta có thể xây dựng từ điển là `["Hà Nội", "Huế", "Sài Gòn"]` (thứ tự không quan trọng) hoặc nếu biết rằng trong tương lai có thể có thêm các địa danh khác thì từ điển có thể là `["Hà Nội", "Huế", "Sài Gòn", "Khác"]`.

2. Sau khi xây dựng được từ điển ta cần lưu lại chỉ số của từng hạng mục trong từ điển. Với từ điển thứ nhất có ba phần tử, chỉ số tương ứng là `"Hà Nội": 0, "Huế": 1, "Sài Gòn": 2`, với từ điển thứ hai, ta có thêm chỉ số `"Khác" :3`.

3. Cuối cùng, ta mã hóa các giá trị ban đầu như sau:

Với từ điền thứ nhất:

| Giá trị | Mã one-hot |
| --- | --- |
| `"Sài Gòn"` | `[0, 0, 1]` |
| `"Huế"` | `[0, 1, 0]`|
|`"Hà Nội"` | `[1, 0, 0]`|

Với từ điền thứ hai:

| Giá trị | Mã one-hot |
| --- | --- |
| `"Sài Gòn"` | `[0, 0, 1, 0]` |
| `"Huế"` | `[0, 1, 0, 0]`|
|`"Hà Nội"` | `[1, 0, 0, 0]`|
|`"Thái Bình"` | `[0, 0, 0, 1]`|
|`"Đồng Nai"` | `[0, 0, 0, 1]`|


Vì mỗi giá trị hạng mục được mã hóa bằng một vector với chỉ một phần tử bằng 1 tại vị trí tương ứng của nó trong từ điển nên vector này được gọi là "one-hot vector". Số chiều của vector này đúng bằng số từ trong từ điển. Diễn giải theo một cách khác, mỗi giá trị nhị phân trong vector này thể hiện việc giá trị hạng mục đang xét "có phải là" giá trị tương ứng trong từ điển không.

Ta thấy rằng cách mã hóa thứ nhất gọn hơn với chỉ ba phần tử nhưng không có cách mã hóa cụ thể cho các giá trị mới có thể xuất hiện trong tương lai. Cách thứ hai đã có một cách mã hóa cụ thể cho các giá trị khác là `"Thái Bình"` và `"Đồng Nai"`, hai giá trị này được mã hóa như nhau.

Việc mã hóa các giá trị chưa biết bằng cùng một vector có thể gây cho mô hình nhầm lẫn rằng đây là hai giá trị giống nhau. Nếu bằng một cách nào đó, bạn biết các giá trị này xuất hiện nhiều trong tương lai, bạn nên đưa chúng vào trong từ điển để có cách mã hóa riêng, tránh trùng lặp với các giá trị khác. Nếu các giá trị này hiếm khi xảy ra, ta có thể cho chung vào một mã và coi như chúng có tính chất giống nhau là "hiếm". Cố gắng mã hóa cho từng giá trị hiếm sẽ dẫn đến tình trạng phải dùng nhiều bộ nhớ và mô hình cũng phức tạp hơn để cố gắng học những trường hợp cá biệt, khi đó overfitting dễ xảy ra.

### Ví dụ với sklearn

Dưới đây là một ví dụ về việc mã hóa one-hot sử dụng
[`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn-preprocessing-onehotencoder). Trước tiên,

```{code-cell} ipython3
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df_train = pd.DataFrame(
    data={"location": ["Hà Nội", "Huế", "Sài Gòn"], "population (M)": [7, 9, 0.5]}
)
df_train
```

Tiếp theo, ta áp dụng mã hóa one-hot lên cột `location`:

```{code-cell} ipython3
onehot = OneHotEncoder()

onehot_encoded_location = onehot.fit_transform(df_train[["location"]])
print(type(onehot_encoded_location))
print(onehot_encoded_location)
```

Có một vài điểm cần lưu ý ở đây. Thứ nhất, kết quả trả về `onehot_encoded_location` mặc định được lưu ở kiểu [`scipy.sparse.csr.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) là một kiểu đặc biệt để lưu các mảng hai chiều có phần lớn các phần tử bằng 0. Cách lưu này rất thuận lợi về mặt bộ nhớ trong trường hợp này vì mỗi vector chỉ có đúng một phần tử khác 0. Nếu kích thước từ điển tăng lên tới hàng triệu và ta lưu ma trận ở dạng thông thường thì sẽ rất lãng phí tài nguyên khi phải lưu rất nhiều giá trị 0 không mang nhiều thông tin.

Khi in `onehot_encoded_location` ra ta sẽ thấy hay cột. Cột thứ nhất là tọa độ của các điểm khác 0, cột thứ hai là giá trị của phần tử ở tọa độ đó -- luôn luôn bằng 1 trong trường hợp này.

Để trả về kết quả ở dạng ma trận thông thường, ta có thể thêm `sparse=False` khi khởi tạo:

```{code-cell} ipython3
onehot = OneHotEncoder(sparse=False)

onehot_encoded_location = onehot.fit_transform(df_train[["location"]])
print(onehot_encoded_location)
```

```{code-cell} ipython3
onehot.categories_
```

```{code-cell} ipython3
onehot2 = OneHotEncoder(categories=["Hà Nội", "Huế", "Sài Gòn"])
onehot_encoded_location1 = onehot2.fit_transform(df_train[["location"]])
print(onehot_encoded_location2)
```

```{code-cell} ipython3
df_test = pd.DataFrame(data = {"location": ["Hà Nội", "Hải Phòng", "Đồng Nai"]})

onehot = OneHotEncoder()

print(onehot.fit_transform(df_train[["location"]]))
```



```{code-cell} ipython3
print(onehot.transform(df_test[["location"]]))
```

Mã hóa one-hot có ưu điểm là đơn giản nhưng
Về mặt toán học, hai vector one-hot khác nhau bất kỳ có khoảng cách trong không gian Euclid bằng $\sqrt{2}$. Như vậy, mã hóa one-hot không thể hiện được sự tương đồng giữa các giá trị hạng mục mà chỉ thể hiện được các giá trị khác nhau.

+++

(sec_embedding)=
## Embedding

(sec_hashing)=
## Hashing

(sec_crossing)=
## Crossing

```{code-cell} ipython3

```
