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

(sec_one_hot)=
# Mã hóa one-hot

Cách truyền thống nhất để đưa dữ liệu hạng mục về dạng số là mã hóa one-hot. Trong cách mã hóa này, một "từ điển" cần được xây dựng chứa tất cả các giá trị khả dĩ của từng dữ liệu hạng mục. Sau đó mỗi giá trị hạng mục sẽ được mã hóa bằng một vector nhị phân với toàn bộ các phần tử bằng 0 trừ một phần tử bằng 1 tương ứng với vị trí của giá trị hạng mục đó trong từ điển.

Ví dụ, nếu ta có dữ liệu một cột là `"Sài Gòn", "Huế", "Hà Nội"` thì ta thực hiện các bước sau:

1. Xây dựng từ điển. Trong trường hợp này ta có thể xây dựng từ điển là `["Hà Nội", "Huế", "Sài Gòn"]`

2. Sau khi xây dựng được từ điển ta cần lưu lại chỉ số của từng hạng mục trong từ điển. Với từ điển như trên, chỉ số tương ứng là `"Hà Nội": 0, "Huế": 1, "Sài Gòn": 2`.

3. Cuối cùng, ta mã hóa các giá trị ban đầu như sau:

Với từ điền thứ nhất:

| Giá trị | Mã one-hot |
| --- | --- |
| `"Sài Gòn"` | `[0, 0, 1]` |
| `"Huế"` | `[0, 1, 0]`|
|`"Hà Nội"` | `[1, 0, 0]`|


Vì mỗi giá trị hạng mục được mã hóa bằng một vector với chỉ một phần tử bằng 1 tại vị trí tương ứng của nó trong từ điển nên vector này được gọi là "one-hot vector". Số chiều của vector này đúng bằng số từ trong từ điển. Diễn giải theo một cách khác, mỗi giá trị nhị phân trong vector này thể hiện việc giá trị hạng mục đang xét "có phải là" giá trị tương ứng trong từ điển không. Với các giá trị mới không nằm trong từ điển (_out-of-vocabolary hay OOV_), ta có thể mã hóa chúng thành `[0, 0, 0]` theo nghĩa chúng không phải là bất cứ một giá trị nào trong từ điển.

Có một cách khác phổ biến để mã hóa các giá trị không có trong từ điển là thêm từ `"unknown"` vào trong từ điển và tất cả những giá trị mới được xếp vào mục `"unknown"` này. Cần lưu ý khi `"unknown"` cũng là một giá trị khả dĩ trong tập dữ liệu. Việc mã hóa các giá trị chưa biết bằng cùng một vector có thể gây cho mô hình nhầm lẫn rằng đây là hai giá trị giống nhau. Nếu bằng một cách nào đó, bạn biết các giá trị này xuất hiện nhiều trong tương lai, bạn nên đưa chúng vào trong từ điển một cách cụ thể để có cách mã hóa riêng, tránh trùng lặp với các giá trị khác. Nếu các giá trị này hiếm khi xảy ra, ta có thể cho chung vào một mã và coi như chúng có tính chất giống nhau là "hiếm". Cố gắng mã hóa cho từng giá trị hiếm sẽ dẫn đến tình trạng phải dùng nhiều bộ nhớ và mô hình cũng phức tạp hơn để cố gắng học những trường hợp cá biệt, khi đó overfitting dễ xảy ra.

## Ví dụ với sklearn

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

Như vậy, thứ tự các hạng mục trong từ điển đã được đảo đi:

```{code-cell} ipython3
onehot.categories_
```

Chúng ta cần lưu lại thứ tự này để có sự nhất quán khi mã hóa các dữ liệu về sau.

Với các giá trị không có trong từ điển, sklearn cung cấp hai cách xử lý thông qua biến `handle_unknown` (xem thêm [tại liệu chính thức](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn-preprocessing-onehotencoder) để biết thêm chi tiết). Biến này có thể nhận một trong hai giá trị `'error'` (mặc định) hoặc `'ignore'`. Với `'error'`, chương trình sẽ dừng chạy và báo lỗi khi gặp một giá trị không nằm trong từ điển. Với `'ignore'`, bộ mã hóa này sẽ biến đổi các giá trị lạ về vector toàn 0. Rất tiếc, bộ mã hóa này không hỗ trợ trường hợp gộp các giá trị mới vào một hạng mục riêng. Việc sử dụng `'error'` và `'ignore'` tùy thuộc vào ngữ cảnh. Nếu bạn biết chắc chắn toàn bộ các giá trị khả dĩ của dữ liệu hạng mục đó thì nên dùng `'error'` để bắt được các trường hợp đầu vào bị lỗi. Ngược lại, bạn nên dùng `'ignore'`; tuy nhiên, cần lưu ý với các trường hợp viết sai chính tả!


## Thảo luận

1. Ngoại trừ trường hợp một giá trị chưa biết được mã hóa thành vector 0, mỗi vector đều có khoảng cách Euclid tới một vector khác bằng $\sqrt{2}`. Việc này không thể hiện được việc các hạng mục có nét tương đồng với nhau.

2. Mã hóa one-hot là một cách biến đổi nhanh chóng từ dữ liệu dạng hạng mục sang dạng số. Với cách mã hóa này, ta có thể xây dựng nhanh chóng các mô hình đơn giản như hồi quy tuyến tính hay SVM, các mô hình này bắt buộc giá trị đầu vào là ở dạng số. Với các mô hình dạng cây quyết định (Random Forest, LightGBM, XGBoost, v.v.) -- rất phổ biến với dữ liệu dạng bảng, ta không cần đưa về dạng onehot mà chỉ cần đưa về dạng số thứ tự trong từ điển và báo với mô hình rằng đó là đặc trưng hạng mục, các mô hình sẽ có cách xử lý phù hợp (Xem thêm [`sklearn.preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)).

3. Với mã hóa onehot, ta cần lưu lại từ điển để tính toán vector cho các giá trị khác trong tương lai. Việc này có hạn chế lớn là cần thêm bộ nhớ và cần biết chính xác số lượng giá trị khả dĩ của dữ liệu. Một kỹ thuật có chức năng gần tương tự được sử dụng nhiều hơn là {ref}`sec_hashing` sẽ được trình bày ở mục tiếp theo.

4. Mã hóa onehot đặc biệt thiếu hiệu quả khi từ điển lớn lên, số chiều của dữ liệu đầu vào lớn sẽ khiến các mô hình machine learning _khó_ học hơn với những đầu vào có số chiều thấp. Một cách cực kỳ hiệu quả và phổ biến khác là xây dựng các embedding biến từng hạng mục thành một vector dày (_dense vector_) có số chiều thấp hơn thay vì ở dạng onehot là vector thưa (_sparse vector_). Các embedding vector cũng sẽ thể hiện được tự tương đồng giữa các hạng mục tốt hơn so với onehot. Chúng ta sẽ thảo luận tới kỹ thuật này trong mục {ref}`sec_embedding`.

```{code-cell} ipython3

```
