# Thư viện tabml đi kèm cuốn sách

Như đã đề cập, phạm vi của cuốn sách dừng lại ở việc coi dữ liệu đã có sẵn được lưu trong các file `cvs`. Việc dự đoán cũng không phải ở thời gian thực áp dụng với từng điểm dữ liệu hoàn toàn mới theo nghĩa khi phát triển hệ thống, các kỹ sư không biết chính xác phân phối của chúng. Ở đây, chúng ta sử dụng các bộ dữ liệu có sẵn, thường được chia thành hai tập. Một tập huấn luyện đã biết nhãn và tập kiểm tra còn lại thì không. Vì vậy, ngoài nhãn, chúng ta biết trước mọi thông tin về dữ liệu trong tập kiểm tra. Thêm nữa, vấn đề về tốc độ dự đoán rất quan trọng trong các hệ thống machine learning thời gian thực cũng sẽ không được ưu tiên.

Cuốn sách này sẽ đi kèm với một thư viện nhỏ tên là [`tabml`](https://github.com/tiepvupsu/tabml) để mô phỏng các thành phần của một hệ thống machine learning thu gọn. Thư viện này sẽ được phát triển song song với nội dung của cuốn sách. Thư viện `tabml` được phát triển trong quá trình tác giả làm cố vấn cho một nhóm Data Science tại [Trung tâm sáng tạo VNPT](https://icenter.ai/vi). Thư viện này được dùng lần đầu tiên khi nhóm khởi động với cuộc thi [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/leaderboard) và đạt thứ hạng cao. Hiện `tabml` cũng đang được sử dụng trong một số dự án của trung tâm.

Thư viện `tabml` sẽ có các chức năng dưới đây:

1. Xây dựng một pipeline tổng quát quát có thể được sử dụng trong nhiều bài toán với dữ liệu dạng bảng khác nhau.

2. Cho phép các kỹ sư thí nghiệm với nhiều loại mô hình khác nhau từ các mô hình trong Scikit-learn tới các mô hình Deep Learning và các thư viện nổi tiếng khác ([LightGBM](https://lightgbm.readthedocs.io/en/latest/), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/), [TabNet](https://github.com/dreamquark-ai/tabnet), v.v.) mà không phải thay đổi code quá nhiều.

3. Cho phép các kỹ sư xây dựng đặc trưng một cách độc lập và an toàn. Với dữ liệu dạng bảng, việc xử lý dữ liệu và xây dựng đặc trưng thường có tầm quan trọng cao hơn so với xây dựng mô hình vì các mô hình thường đã có thư viện có sẵn. Khi xây dựng đặc trưng, chúng ta cần thử nghiệm thêm bớt rất nhiều đặc trưng khác nhau với mối quan hệ chằng chịt. Việc có một _bộ quản lý đặc trưng_ (feature manager) là cực kỳ quan trọng.

4. Ngoài ra, thư viện này sẽ giúp đánh giá chất lượng hệ thống trên nhiều hạng mục dữ liệu khác nhau để kiểm tra chất lượng của mô hình trên từng nhóm đối tượng. Việc này giúp các kỹ sư nhận ra hạng mục nào mà mô hình có chất lượng tệ để đưa ra những cải tiến.

Để có một hệ thống machine learning hoạt động tốt, bạn cần thử rất nhiều ý tưởng cải thiện mô hình. Nếu pipeline giúp bạn thử nghiệm ý tưởng càng nhanh, thì khả năng bạn có một mô hình với chất lượng tốt ngày càng cao.


## Cài đặt

Chạy dòng lệnh sau để cài đặt bản cập nhật nhất của `tabml`:

```
python -m pip install --ignore-installed --no-deps  git+https://github.com/tiepvupsu/tabml.git
```
