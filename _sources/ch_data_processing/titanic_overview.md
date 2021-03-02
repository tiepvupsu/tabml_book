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

# Khái quát về bộ dữ liệu Titanic

Trước khi đi sâu vào các kỹ thuật làm sạch dữ liệu và xây dựng đặc trưng, chúng ta cùng làm quen với bộ dữ liệu Titanic.
Bộ dữ liệu này gồm có ba file:

```{code-cell} ipython3
!ls ../data/titanic
```

Cùng xem nhanh dữ liệu trong ba file này bằng cách mở một vài dòng đầu tiên của mỗi file.

```{code-cell} ipython3
!echo "--------------train.csv------------------"
!csvlook ../data/titanic/train.csv | head -5
!echo "--------------test.csv------------"
!csvlook ../data/titanic/test.csv | head -5
!echo "----------------gender_submission.csv---------------"
!csvlook ../data/titanic/gender_submission.csv | head -5
```

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

```{code-cell} ipython3

```
