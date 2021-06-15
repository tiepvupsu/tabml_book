(sec_missing_data)=
# Xử lý dữ liệu bị khuyết

Với dữ liệu bảng, việc các trường thông tin bị khuyết (giá trị `nan`) là thường xuyên xảy ra. Việc này đến từ quá trình thu thập dữ liệu. Người dùng có thể thực sự không có thông tin đó hoặc họ không muốn tiết lộ vì lý do riên tư cá nhân.

Vậy, chúng ta cần làm gì với dữ liệu bị khuyết?

Một cách đơn giản là bỏ cả cột thông tin có dữ liệu bị khuyết ra khỏi quá trình xây dựng mô hình. Cách làm này đơn giản nhưng có những hạn chế nhất định. Nếu có quá nhiều cột có dữ liệu bị khuyết và ta đều bỏ các cột này đi thì sẽ không còn thông tin gì cho việc xây dựng mô hình.

Cách thứ hai là bỏ đi hàng có giá trị bị khuyết đó khỏi tập huấn luyện. Việc này cũng có hạn chế tương tự như trên nên mỗi hàng mất một chút dữ liệu. Hạn chế thứ hai là khi gặp một điểm dữ liệu mới mà mô hình cần dự đoán, ta không thể đơn giản bỏ điểm dữ liệu đó đi mà vẫn phải dự đoán ra một giá trị nào đó.

Có một quy tắc chung là nếu một cột có _quá nhiều_ dữ liệu bị khuyết thì ta có thể bỏ cột đó. Việc tìm ngưỡng thế nào là _quá nhiều_ tùy thuộc vào tính chất của dữ liệu và kinh nghiệm của các kỹ sư. Nếu dữ liệu có quá ít thông tin và lại bỏ bớt đi có thể làm giảm chất lượng mô hình.

Nếu muốn giữ một cột có thông tin bị khuyết, có hai hướng chính.

Thứ nhất là tạo một cột mới `is_nan` mang thông tin dữ liệu _có bị khuyết hay không_. Đôi khi việc khuyết thông tin cũng chính là một thông tin quý giá. Một người không khai báo số điện thoại có thể vì họ không có điện thoại, một người giấu địa chỉ IP của máy chứng tỏ họ đề cao quyền riêng tư và có kỹ năng nhất định về máy tính. Như vậy việc thiếu thông tin cũng chính là một thông tin có thể khai thác.

Cách thứ hai giúp ta có thể giải quyết vấn đề dữ liệu bị khuyết là "điền" (_impute_) các giá trị bị khuyết một giá trị nào đó rồi dùng giá trị đó để xây dựng mô hình.

## Dữ liệu dạng số

Với dữ liệu dạng số, hai cách phổ biến và đơn giản nhất là điền các giá trị bị khuyết bằng trung bình cộng hoặc trung vị của các giá trị không bị khuyết. Đây là các lựa chọn an toàn vì trung bình cộng hoặc trung vị là các giá trị có khả năng cao xảy ra. Một điểm đáng lưu ý là việc lấy trung bình hay trung vị này nên được cân nhắc dựa trên dữ liệu trước hoặc sau khi xử lý các điểm ngoại lệ.

Thư viện scikit-learn với lớp [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) thường được sử dụng cho tác vụ này. Lấy ví dụ với cột `Age` trong dữ liệu Titanic. Trong bộ dữ liệu này, tập `train.csv` có $891 - 714 = 177$ điểm bị khuyết, tập `test.csv` có $418 - 332 = 86$ điểm bị khuyết.

import pandas as pd

titanic_path = "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/titanic/"

df_train = pd.read_csv(titanic_path + "train.csv")
df_test = pd.read_csv(titanic_path + "test.csv")
df_train[["Age"]].info()

df_test[["Age"]].info()

Một điểm đáng lưu ý khác là việc tính toán giá trị để điền chỉ được dựa trên dữ liệu huấn luyện, trong trường hợp này là tập `train.csv`. Khi điền các giá trị bị khuyết trên tập `test.csv` ta cần sử dụng kết quả thu được ở tập `train.csv`. Dưới đây là ví dụ cụ thể với việc sử dụng `sklearn.impute.SimpleImputer` và cách điền là `'median'` (trung vị).

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(df_train[["Age"]])
df_train[["ImputedAge"]] = imputer.transform(df_train[["Age"]])
df_test[["ImputedAge"]] = imputer.transform(df_test[["Age"]])

df_train[["Age", "ImputedAge"]].tail(3)

df_test[["Age", "ImputedAge"]].tail(3)

Đoạn code trên đây tính toán giá trị trung vị (`strategy='median'`) dựa trên các điểm _không_ bị khuyết ở tập huấn luyện rồi điền vào cả hai tập đó. Ta thấy rằng các giá trị `NaN` ở cột `Age` đã được điền một giá trị gần bằng $28.0$ ở cột `ImputedAge`. Bạn cũng có thể thử với các `strategy` khác để xem cách nào mang lại kết quả tốt nhất. Nên nhớ rằng không có một cách điền giá trị nào đúng cho mọi loại dữ liệu, bạn cần hiểu dữ liệu để đề ra phương án mà bạn nghĩ là có kết quả tốt nhất.

Nếu có thêm thời gian, bạn có thể điền các giá trị một cách tỉ mỉ hơn. Ví dụ, điền các giá trị về tuổi bị khuyết khác nhau cho mỗi loại giới tính.

## Dữ liệu hạng mục

Với dữ liệu hạng mục, vì ta không tính được giá trị trung bình nên cách thường dùng là điền vào giá trị xuất hiện nhiều nhất (`strategy='mode'`) hoặc coi chính việc bị khuyết là một giá trị đặc biệt (`strategy='constant'`) với giá trị đặc biệt được truyền qua biến `fill_value` (Xem thêm tại [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)).