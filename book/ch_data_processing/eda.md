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