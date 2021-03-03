# EDA (WIP)

EDA (Exploratory Data Analysis -- Phân tích Khám phá Dữ liệu) là một bước quan trọng trước khi làm bất kỳ một bài toán ML với dữ liệu dạng bảng nào.

*Trước khi xây dựng mô hình, bạn cần xây dựng đặc trưng. Trước khi xây dựng đặc trưng, bạn phải làm bước khám phá dữ liệu.*

## Mục đích của EDA

Bước EDA này giúp chúng ta có cái nhìn đầu tiên về dữ liệu.
Bạn cần có một cảm giác nhất định về những gì mình có trong tay trước khi có những chiến lược xây dựng mô hình.
EDA giúp bạn mường tượng được độ phức tạp của bài toán và vạch ra những bước đầu tiên cần làm.

```{note}
Để phân biệt dữ liệu trước và sau bước tiền xử lý, các cột trong bảng dữ liệu ban đầu được gọi là *trường dữ liệu*. Các cột đã được xử lý và sẵn sàng cho việc huấn luyện mô hình được gọi là *đặc trưng*.
```

Vậy, làm EDA là làm những gì?

Dưới đây là danh sách những câu hỏi bạn cần trả lời ở bước EDA:

1. Có bao nhiêu mẫu dữ liệu tổng cộng?

2. Có bao nhiêu trường thông tin trong dữ liệu?

3. Bạn có thông tin về ý nghĩa của từng trường dữ liệu không?

4. Có bao nhiêu trường dữ liệu ở dạng số, bao nhiêu ở dạng hạng mục.

5. Phân bố của từng trường dữ liệu như thế nào? Phân bố đó có bị lệch không? Mỗi trường dữ liệu có bao nhiêu giá trị bị khuyết, có nhiều phần tử ngoại lai (_outliers_) không.

6. Trong các trường dữ liệu đó, trường dữ liệu nào có mối tương quan cao với nhãn cần dự đoán.

Trả lời được những câu hỏi này sẽ giúp ích rất nhiều cho việc làm sạch dữ liệu và xây dựng đặc trưng sau này.

Việc khám phá dữ liệu không chỉ dừng lại ở lần đầu tiên trước khi xây dựng đặc trưng mà còn cần được thực hiện trong suốt quá trình phát triển hệ thống.
Sau khi xây dựng xong các đặc trưng, bạn cũng cần làm lại EDA một lần nữa để xem dữ liệu đã qua xử lý đó đã thực sự _sạch_ chưa.
Ngoài ra, sau khi xây dựng và phân tích mô hình, ta cũng thường xuyên cần quay lại EDA để tiếp tục khám phá những điều còn ẩn giấu trong dữ liệu bài toán. Càng hiểu sâu về dữ liệu, bạn sẽ càng sớm giải thích được những hành vi của mô hình và đưa ra những thay đổi phù hợp.

## Ví dụ với dữ liệu Titanic

Cùng làm quen với EDA thông qua ví dụ với bộ dữ liệu đơn giản Titanic.
Bộ dữ liệu này gồm có ba file:

!cd ../
!ls data/titanic

Cùng xem nhanh dữ liệu trong ba file này bằng cách mở một vài dòng đầu tiên của mỗi file.

import pandas as pd
df_train = pd.read_csv("../data/titanic/train.csv")
df_train.head(5)

df_test = pd.read_csv("../data/titanic/test.csv")
df_test.head(5)

df_sub = pd.read_csv("../data/titanic/gender_submission.csv")
df_sub.head(5)

Chúng ta có thể thấy nhanh rằng:

* File `train.csv` và `test.csv` có tập hợp các cột với tên gần như nhau, ngoài trừ việc cột `"Survived"` không xuất hiện ở file `test.csv`. Bài toán đặt ra là dùng các cột còn lại của file `train.csv` để huấn luyện một mô hình sao cho nó có thể dự đoán được cột `"Survived"` này dựa trên những cột của file `test.csv`.

* File `gender_submission.csv` chỉ có hai cột `"PassengerID"` và `"Survived"`; đây là file nộp bài mẫu mà người chơi cần hoàn thiện. Cột `"PassengerID"` bao gồm những mã số hành khách có trong tập `test.csv` trong khi cột `"Survived"` chứa các giá trị dự đoán mà người chơi cần thay thế. Các giá trị mẫu này tương ứng với việc dự đoán chỉ có giới tính `"female"` là sống sót. Đây có thể coi là một giải pháp nền (_baseline_) cho bài toán khi chỉ sử dụng một đặc trưng duy nhất là `"Sex"`.

* Cột `"Cabin"` trong hai file dữ liệu có những giá trị bị khuyết.

**Ý nghĩa của từng trường thông tin**

Trước khi đi tìm hướng giải quyết bài toán, chúng ta cần biết ý nghĩa của các cột còn lại (được tìm thấy tại [trang web cuộc thi](https://www.kaggle.com/c/titanic/data):

* `"Pclass"`: hạng ghế. 1 = hạng _Upper_, 2 = hạng _Middle_, 3 = hạng _Lower_. Như vậy, trường thông tin `"Pclass"` vừa có thể coi là một đặc trưng hạng mục, vừa có thể coi là một đặc trưng dạng số vì nó có thứ tự. Đặc trưng này khả năng ảnh hưởng tới khả năng sống sót của hành khách vì hạng sang hơn có thể có các biện pháp an toàn tốt hơn (hoặc cũng có thể ngược lại là chủ quan hơn).

* `"Sex"`: giới tính hành khách.

* `"Age"`: tuổi của hành khách. Nếu tuổi nhỏ hơn 1 thì ở dạng số lẻ (0.42), nếu tuổi là ước lượng thì ở dạng xx.5. Rõ ràng đây cùng sẽ là một đặc trưng tiềm năng để dự đoán kết quả cho bài toán vì trẻ em và người già ở vào nhóm có nguy cơ cao hơn.

* `"Sibsp"`: số lượng anh chị em hoặc vợ/chồng cùng ở trên tàu.

* `"Parch"`: số lượng bô mẹ/con cái cùng ở trên tàu.

* `"Ticket"`: mã số vé.

* `"Fare"`: giá vé.

* `"Cabin"`: mã số cabin.

* `"Embarked"`: Nơi lên tàu, `C` = Cherbourg, `Q` = Queenstown, `S` = Southamton. 

Trong những thông tin trên, chúng ta có thể thấy có những thông tin ở dạng số như `Age, Fare, Parch, Sibsp`, có những thông tin ở dạng hạng mục như `Pclass, Sex, Ticket, Cabin, Embarked`. Đánh giá ban đầu có thể cho ta nhận định rằng có những thông tin có thể hữu ích cho việc xây dựng mô hình như `Pclass, Age, Parch, Sibsp` và những thông in có thể ít hữu ích hơn như `Cabin, Embarked, Ticket, Fare`.


Đây là một bộ dữ liệu nhỏ với chỉ hơn 1000 mẫu trong cả hai tập huấn luyện và kiểm tra.
Khi dữ liệu lơn hơn, chúng ta cần có cái nhìn bao quát hơn về dữ liệu thông qua các bảng thống kê của từng trường thông tin.
Thư viện [`pandas`](https://pandas.pydata.org/) là một trong các thư viện phổ biến nhất để xử lý dữ liệu dạng bảng.

```{margin}
Vì pandas thường cần load toàn bộ file vào RAM nên nó không phù hợp với các bộ dữ liệu lớn.
Với dữ liệu lớn, mời bạn đọc thêm về [dask](https://dask.org/), [modin](https://modin.readthedocs.io/en/latest/) với cú pháp tương tự pandas hoặc [pyspark](https://spark.apache.org/docs/latest/api/python/) cho việc xử lý dữ liệu trên các hệ phân tán. 
```


```{toctree}
:hidden:
:titlesonly:


titanic_overview
titanic_numeric
titanic_categorical
pandas-profiling
```