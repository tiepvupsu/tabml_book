(sec_hashing)=
# Hashing

One-hot encoding có một hạn chế lớn là cần biết trước từ điển và kích thước của nó. Từ điển này cũng cần được lưu lại để mã hóa các hạng mục mới trong tương lai. Thử tưởng tượng một cửa hàng thương mại điện tử có 1000 mặt hàng trong hôm nay nhưng trong một tháng tới, cửa hàng đó có thêm 1000 mặt hàng mới. Vậy các mặt hàng mới đó nên được mã hóa như thế nào? Rõ ràng việc mã hóa chúng bằng vector 0 hay bằng one-hot vector tương ứng với `"unknown"` sẽ làm giảm chất lượng mô hình vì không có sự phân biệt rạch ròi giữa 1000 mặt hàng mới. Lúc này, để có thể tiếp tục sử dụng mã hóa one-hot, ta cần cập nhật từ điển và mã hóa lại toàn bộ các giá trị hạng mục. Điều này đồng nghĩa với việc đầu vào của mô hình sẽ thay đổi và khả năng cao ta cũng cần thay đổi kiến trúc của mô hình để thích ứng với sự thay đổi kích thước đầu vào.

Một kỹ thuật được sử dụng rất nhiều để giải quyết vấn đề này là hashing (hàm băm). Hashing là một phép biến đổi giá trị đầu vào bất kỳ thành một số nguyên. Một hàm hash tốt là hàm có đặc điểm biến những giá trị đầu vào khác nhau thành các điểm phân bố đều trong khoảng giá trị khả dĩ (các số nguyên 32 bit hoặc nhiều hơn tùy thuộc vào hàm hash). Một đặc điểm khác là các giá trị đầu vào khác nhau sẽ được biến thành các số nguyên có khả năng cao khác nhau, đặc biệt khi dùng số lượng bit lớn.

Để sử dụng hashing như một cách biến đổi các giá trị hạng mục về một số tự nhiên được sử dụng trong các mô hình machine learning, ta có thể thực hiện các bước sau:

1. Biến đổi các giá trị hạng mục về dạng `string` (một số hàm hash chỉ chấp nhận đầu vào là dạng `string`).

2. Lựa chọn một hàm hash "tất định", tức một hàm luôn trả về một số cố định ở tất cả các lần chạy nếu đầu vào không đổi. Việc này rất quan trọng vì nếu với một giá trị mà hàm hash trả về đầu ra khác nhau thì mô hình machine learning không thể biết được các đầu vào là một. Lưu ý rằng một số hàm hash có khả năng trả về các giá trị khác nhau, có thể vì lý do bảo mật, ta cần tránh sử dụng các hàm hash này.

3. Ước lượng số lượng các phần tử khác nhau của dữ liệu hạng mục rồi chọn một số tự nhiên $K$ làm mod. Lấy phần dư của kết quả ở bước hai khi chia cho số $K$ này làm chỉ số cho hạng mục tương ứng.

## Ví dụ

Xét ví dụ về dữ liệu các sản phẩm trong bộ dữ liệu [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview).

import pandas as pd

df_items = pd.read_csv("../data/sales/items.csv")
print(f"Number of items: {len(df_items)}")
print(f"Number of category: {len(df_items['item_category_id'].unique())}")
df_items.head(5)

Như vậy, có 22170 sản phẩm khác nhau được chia vào 84 hạng mục khác nhau. Các hạng mục này có thể được trực tiếp sử dụng để xây dựng one-hot vector. Ta cũng có thể thấy rằng sẽ có nhiều sản phẩm được chia vào cùng một hạng mục. Nếu không có đặc trưng khác để phân biệt các sản phẩm, ta sẽ xây dựng được một mô hình mà mọi sản phẩm trong cùng một hạng mục có cùng một tính chất. Để có thể tách biệt được các sản phẩm, ta có thể xử lý thêm thông tin về tên sản phẩm ở cột thứ nhất. Việc này sẽ tương đối khó khăn vì không phải kỹ sư nào cũng biết tiếng Nga. Một cách khác là sử dụng `item_id` làm đặc trưng hạng mục và xây dựng one-hot vector cho cột này với 22170 phần tử. Đây là số lượng phần tử tương đối lớn, ngoài ra trong dữ liệu huấn luyện (file `sales_train.csv`), rất nhiều `item_id` chỉ xuất hiện một lần. Nếu xây dựng one-hot với 22170 phần tử, khả năng cao mô hình sẽ bị overfitting khi có quá nhiều hạng mục có ít dữ liệu.

Hashing là một kỹ thuật khả dĩ có thể áp dụng lên `item_name`. Dưới đây là một triển khai đơn giản của kỹ thuật hashing được viết theo sklearn API với số lượng hash bucket là 1000:

import hashlib
from typing import Tuple

from sklearn.base import BaseEstimator, TransformerMixin


def hash_modulo(val, mod):
    md5 = hashlib.md5()  # can be other deterministic hash functions
    md5.update(str(val).encode())
    return int(md5.hexdigest(), 16) % mod


class FeatureHasher(BaseEstimator, TransformerMixin):
    def __init__(self, num_buckets: int):
        self.num_buckets = num_buckets

    def fit(self, X: pd.Series):
        return self

    def transform(self, X: pd.Series):
        return X.apply(lambda x: hash_modulo(x, self.num_buckets))


fh = FeatureHasher(num_buckets=1000)

df_items["hashed_item"] = fh.transform(df_items["item_name"])
df_items.head(5)

Lúc này, ta có thêm cột `"hashed_item"` với các giá trị không vượt quá 1000. Cột này cũng giúp phân biệt các sản phẩm trong hạng mục 40.

df_items["hashed_item"].value_counts()

## Thảo luận

1. Hashing đặc biệt hữu ích khi ta không biết từ điển của một trường hạng mục. Nếu có giá trị OOV (ngoài từ điển), hashing vẫn tạo ra được một số tự nhiên và có thể đảm bảo được các giá trị chưa có trong từ điển nhận các mã khác nhau. Từ điển cùng với vị trí của từng từ trong đó không cần được lưu trong bộ nhớ. Kỹ thuật hash đặc biệt hữu ích với online learning khi ta không biết trước từ điển.

2. Với những hạng mục có lượng giá trị phân biệt lớn (_high cardinality_) nhưng có nhiều hạng mục có tần suất thấp, ta có thể chọn $K$ nhỏ để hạn chế việc tốn tài nguyên bộ nhớ và tránh overfitting. Điều này sẽ dẫn đến việc có nhiều giá trị đầu vào khác nhau cùng có một đầu ra (cùng _hash bucket_). Mặc dù vậy, các thuật toán machine learning sẽ vẫn có thể phân biệt được chúng dựa trên các đặc trưng khác.

3. Với những hạng mục mà việc xung đột hash (_hash collision_ -- các giá trị đầu vào khác nhau mà đầu ra của hàm hash cho cùng một giá trị) trở nên nghiêm trọng, ta có thể chọn số lượng bucket lớn để hạn chế việc xung đột. Việc chọn số lượng bucket như thế nào để xác suất xung đột thấp, bạn đọc có thể đọc thêm phân tích chi tiết trong [Don't be tricked by the Hashing Trick](https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087). Khi lượng bucket lớn, sẽ có nhiều bucket không có phần tử nào. Nếu sử dụng deep learning, các hệ số tương ứng với bucket này sẽ không được cập nhật mà sẽ là các giá trị ngẫu nhiên được xác định trong quá trình khởi tạo. Trong tương lai, nếu có một phần tử rơi vào bucket này, mô hình có thể sẽ đưa ra kết quả không tốt. Để giảm thiểu tình trạng này, ta có thể sử dụng các kỹ thuật điều chuẩn (_regularization_) như $\ell_1$ hoặc $\ell_2$ để đưa các hệ số tương ứng về gần với 0.


## Tài liệu tham khảo

[1] [Feature Hashing for Large Scale Multitask Learning](https://alex.smola.org/papers/2009/Weinbergeretal09.pdf)

[2] [Don't be tricked by the Hashing Trick](https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087)

[3] [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)