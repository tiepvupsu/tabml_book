# Tại sao cần xây dựng pipeline

Tại sao chúng ta cần có pipeline khi xây dựng một hệ thống Machine Learning?

<!-- ## Tại sao notebook không phải là một lựa chọn tốt? -->

Hệ thống machine learning thường bao gồm rất nhiều các thành phần nhỏ như xử lý dữ liệu, huấn luyện mô hình, đánh giá mô hình, dự đoán với dữ liệu mới, v.v. Nếu không xây dựng một pipeline hoàn chỉnh với từng thành phần tách biệt rõ ràng, sẽ có rất nhiều vấn đề xảy ra.

Điểm đặc biệt của machine learning khi so sánh với các hệ thống phần mềm thông thường là rất khó để viết tests cho từng thành phần. Code của bạn có thể có rất nhiều lỗi nhưng hệ thống vẫn chạy bình thường. Dữ liệu có thể thay đổi phân phối, một đặc trưng nào đó có thể có lỗi khi thu thập, metric đánh giá có thể có lỗi, hệ thống của bạn vẫn có thể âm thầm chạy.
Và thật nguy hiểm nếu hệ thống vẫn chạy và cho ra kết quả ngẫu nhiên. Việc tách cả hệ thống ra từng thành phần rồi ghép lại thành một pipeline giúp bạn có thể viết các logic kiểm tra cho từng bước và chia nhỏ pipeline ra để tìm thành phần gây lỗi mỗi khi có điều bất thường.

Ngoài sự thuận tiện trong việc rà soát lỗi, có pipeline với các thành phần riêng biệt còn giúp việc làm việc nhóm trở nên dễ dàng hơn. Nếu có nhóm lớn, ta có thể chia ra thành từng nhóm nhỏ: một nhóm chuyên làm sạch dữ liệu, một nhóm chuyên tạo các đặc trưng, nhóm khác đi xây dựng và huấn luyện mô hình và một nhóm khác tập trung vào đánh giá và theo dõi hoạt động của mô hình. Những khối công việc này nếu được tách nhỏ và chuyên biệt sẽ giúp các nhóm đi sâu vào cải thiện chất lượng của từng khối mà không lo đến việc làm hỏng code của nhóm khác.
Ngoài ra, mỗi khi có một ý tưởng mới hoặc mô hình mới, bạn cũng sẽ dễ dàng thay đổi thành phần đó để kiểm nghiệm mà không phải đi xây dựng lại toàn bộ hệ thống.
Việc này hữu ích không chỉ trong nhóm lớn mà với khi làm việc độc lập.
Bạn có 5-10 ý tưởng về việc xây dựng mô hình khác nhau, bạn không cần phải xây dựng lại cả 5-10 khối code khác nhau cho từng mô hình mà chỉ cần thay thế phần tử "mô hình" trong cả pipelie.

## Tại sao notebook không phải là lựa chọn tốt

Với những dự án nhỏ chỉ có một thành viên, toàn bộ các thành phần có thể được thực hiện trong một notebook (jupyter, colab, kaggle, v.v.).
Tuy nhiên, notebook chỉ hữu ích khi bạn muốn kiểm tra ý tưởng, khám phá dữ liệu hoặc minh họa kết quả. Notebook không bao giờ nên là nơi bạn phát triển sản phẩm trong dài hạn.

Thứ nhất, phát triển phần mềm cần đi đôi với việc duy trì liên tục. Không một sản phẩm phần mềm nào viết một lần rồi có thể sử dụng mãi mãi. Một ngày nào đó bạn sẽ nhận ra có lỗi bảo mật, có bug hoặc phiên bản thư viện đã lỗi thời, bạn cần phải cập nhật. Việc có một công cụ quản lý phiên bản code, ví dụ git, là việc tối quan trọng. Các notebook rất tiếc không hỗ trợ tốt việc quản lý phiên bản.

Thứ hai, notebook cho phép bạn chạy code không theo thứ tự. Bạn có thể chạy một code cell rồi quay lại chạy một code cell khác trước nó. Việc linh động này tốt cho việc kiểm định các ý tưởng ban đầu khi bạn thay đổi ý tưởng liên tục. Tuy nhiên, nó lại rất nguy hiểm khi notebook lớn lên với nhiều ràng buộc giữa các code cell. Khi đó bạn có thể dễ dàng có những biến bị thay đổi giá trị khiến kết quả cuối cùng không như mong đợi. Và bạn muốn debug? Chúc may mắn!

Thứ ba, viết test cho notebook rất khó. Việc bạn không thể gọi các hàm đã viết trong một notebook trong một môi trường khác (notebook khác, file python, v.v.) khiến việc viết test cho các khối code gần như là không thể. Bạn có thể để test trong notebook đó, nhưng đây không phải là một ý tưởng tốt khi mà notebook của bạn sẽ lớn lên nhanh chóng. Nếu bạn chưa bao giờ viết test, bạn chưa bao giờ lập trình một cách nghiêm túc!

(Xem thêm [5 reasons why jupyter notebooks suck](https://towardsdatascience.com/5-reasons-why-jupyter-notebooks-suck-4dc201e27086))

Ngoài các lý do kể trên, các vấn đề về format code, linting, chuẩn hóa phong cách code cũng rất khó được thực hiện khi làm việc trên notebook. Tôi đã từng gặp các nhóm sinh viên và kỹ sư chỉ hoàn toàn làm việc trên notebook. Và khi cần chuyển hóa notebook thành các file `.py` để đưa vào repo code chung của nhóm, họ mất rất nhiều thời gian để đạt được kết quả tương tự như trên notebook của họ.

```{note}
Notebook là một công cụ tốt để kiểm định các ý tưởng ban đầu và minh họa kết quả sau khi xây dựng hệ thống hoàn chỉnh. Notebook không bao giờ nên là môi trường phát triển sản phẩm lâu dài.
```
