# Giới thiệu

## Mục đích của dự án

Kể từ sau thành công của AlexNet trong cuộc thi ImageNet năm 2012, Machine Learning (ML)
đã trở thành một chủ đề hấp dẫn đối với sinh viên và các kỹ sư công nghệ. Các tập đoàn
lớn đổ dồn tài nguyên vào phát triển các trung tâm nghiên cứu và các hệ thống tính toán
để giải quyết các bài toán kinh doanh cũng như thu hút nhân tài. Việc một trường đại học
có thêm một khoa riêng về Trí Tuệ Nhân Tạo (AI) cũng không phải hiếm. Các blog, khoá
học, seminar về AI cũng mọc lên như nấm.

Nhắc tới ML, phần nhiều bạn đọc đồng nhất nó với Deep Learning (DL) vì vô số tài liệu
hiện tại chỉ nói về nó mà ít đề cập tới lịch sử lâu dài của ngành ML. Nhắc tới Machine
Learning, bạn có thể chỉ thấy các bài toán mà đầu vào là ảnh/video hoặc các loại văn
bản. Bạn cũng có thể đã từng mày mò với các mã nguồn có sẵn với những bộ dữ liệu đã được
làm sạch sẽ và tập trung nhiều vào thuật toán Machine Learning. Tuy nhiên, dữ liệu và
bài toán thực tế khác rất nhiều so với những demo mà các bạn thường thấy. Để giải
quyết các bài toán kinh doanh, các kỹ sư cần rất nhiều dữ liệu khác ngoài ảnh và văn
bản.

Ví dụ, trong bài toán gợi ý sản phẩm tới khách hàng, các thông số về sản phẩm và về
khách hàng đóng vai trò vô cùng quan trọng. Ảnh sản phầm, nội dung comment chỉ là một
phần rất nhỏ trong những dữ liệu có thể mang lại độ chính xác cao cho thuật toán. Trong
một ví dụ khác về nhận diện kỳ quan, ngoài bức ảnh, các thông tin khác về người chụp, vị
trí địa lý, loại máy, v.v. chắc chắn sẽ mang lại kết quả cao hơn cho các mô hình.

Có những bài toán rất quan trọng mà ở đó các thông tin về hình ảnh và văn bản có thể
không hề tồn tại, chẳng hạn bài toán dự đoán nợ xấu, bài toán dự đoán lượng mua để cân
đối kho hàng hay bài toán dự đoán lưu lượng server trong ngày Tết để chuẩn bị lượng máy
chủ cho phù hợp. Để giải quyết các bài toán này, các hệ thống ML cần nhiều thông tin
khác nhau về mỗi đối tượng. Các thông tin này thường được lưu dưới dạng bảng và thường
được gọi là **tabular data**. Đáng tiếc thay, Deep Learning không phải lúc nào cũng hoạt
động tốt với dữ liệu dạng này (Đọc thêm [The Unreasonable Ineffectiveness of Deep Learning on Tabular Data](https://towardsdatascience.com/the-unreasonable-ineffectiveness-of-deep-learning-on-tabular-data-fd784ea29c33)).

Các kỹ năng giải quyết các bài toán kinh doanh dựa trên dữ liệu dạng bảng này chưa được
trình bày nhiều trong các tài liệu tiếng Việt. Dự án "Machine Learning cho dữ liệu dạng
bảng" này được ra đời với mong muốn mang tới cho bạn đọc những kiến thức và kinh nghiệm
mà cá nhân tôi thu được kể từ khi đi làm thực tế. Viết cũng là cách mà tôi thấy tốt nhất
để củng cố lại kiến thức của bản thân.

## So sánh với Machine Learning cơ bản

So với [Machine Learning cơ bản](https://machinelearningcoban.com/), dự án này sẽ rất
khác. Tôi sẽ không viết thành các bài dài và công bố mỗi một hoặc hai tuần như trước mà
chia thành các mục ngắn và không thường xuyên chia sẻ. Tôi sẽ chỉ công bố rộng rãi khi
hoàn thiện một chương. Tôi cũng sẽ tập trung nhiều vào các kỹ năng giải quyết dữ liệu
thực tế hơn là đi sâu vào các thuật toán cơ bản như đã từng làm.

Khác với Machine Learning cơ bản, tôi có ý định viết một cuốn sách từ đầu trong dự án
này. Format của website cũng khác rất nhiều với thông tin tối giản phục vụ cho việc
trích xuất ra nhiều định dạng khác về sau (pdf, epub, mobi). Toàn bộ mã nguồn của cuốn
sách có thể được tìm thấy tại
[https://github.com/tiepvupsu/tabml-book](https://github.com/tiepvupsu/tabml-book).

