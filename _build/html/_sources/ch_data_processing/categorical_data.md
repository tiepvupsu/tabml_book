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


```{code-cell} ipython3

```
