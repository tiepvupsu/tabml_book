# Bộ quản lý đặc trưng

Với dữ liệu dạng bảng, việc xây dựng đặc trưng được xem là có độ quan trọng còn cao hơn việc xây dựng mô hình.
Chúng ta sẽ thường xuyên phải thêm bớt, cập nhật, sửa xóa các đặc trưng trong quá trình phát triển sản phẩm.
Khi muốn thay đổi một đặc trưng nào đó, một cách đơn giản nhất là thay đổi đoạn code liên quan chạy lại toàn bộ các bước xây dựng đặc trưng.
Với các bộ dữ liệu nhỏ và lượng đặc trưng ít, việc làm này không bị ảnh hưởng nhiều bởi tốc độ và logic giữa các đặc trưng.
Với các bộ dữ liệu lớn với hàng trăm đặc trưng ràng buộc chồng chéo lẫn nhau, việc này mang lại rất nhiều phiền toái.

Thứ nhất, việc chạy lại toàn bộ phần code xây dựng đặc trưng có thể mất nhiều giờ đồng hồ.
Điều này đồng nghĩa với việc bạn sẽ phải chờ chừng đó thời gian với chỉ một thay đổi nhỏ trong đặc trưng.

Thứ hai, khi có nhiều đặc trưng, sẽ có sự phụ thuộc giữa chúng.
Lấy ví dụ bạn có một đặc trưng về tuổi có dữ liệu bị khuyết.
Bước đầu tiên bạn dự đoán các giá trị bị khuyết bằng cách lấy trung bình và có đặc trưng `age`.
Bước thứ hai bạn tạo đặc trưng khoảng `bucketized_age` bằng cách chia tuổi vào các mục "trẻ em, thiếu tiên, thanh niên, trung niên, người già".
Sau đó, ở bước thứ ba, bạn lại kết hợpc đặc trưng khoảng này với đặc trưng giới tính để tạo ra đặc trưng `bucketized_age_X_gender`.
Một ngày nào đó, bạn nhận ra rằng việc dự đoán giá trị bị khuyết bằng cách lấy trung vị có vẻ hợp lý hơn và muốn thử.
Câu hỏi đặt ra là bạn sẽ thay đổi những đặc trưng nào? Chỉ `age` hay cả ba đặc trưng liên quan?
Nhiều khả năng bạn sẽ muốn thay đổi cả ba đặc trưng đó.
Trong trường hợp đó, bạn cần có một cách thông minh hơn để lưu giữ những sự phụ thuộc giữa các đặc trưng để khi sửa/xóa, mọi đặc trưng phụ thuộc đều được cập nhật theo.

Ở đây ta cần định nghĩa sự phụ thuộc giữa các đặc trưng:

1. Nếu việc tính toán đặc trưng `"f1"` cần đặc trưng `"f2"` thì ta nói `"f2"` là một _dependency_ của `"f1"` và ngược lại, `"f1"` được gọi là _dependent_ của `"f2"`.

2. Một đặc trưng có thể có nhiều _dependency_ và nhiều _dependent_.

3. Nếu `"f1"` phụ thuộc vào `"f2"`, `"f2"` phụ thuộc vào `"f3"` thì ta cũng có `"f1"` phụ thuộc vào `"f3"`.
Khi đó ta cũng nói `"f3"` là một _dependency_ của `"f1"` và `"f1"` là một _dependent_ của `"f3"`.

Rõ ràng ta cần một bộ quản lý đặc trưng trong trường hợp này. Bộ quản lý này cần giải quyết các bài toán sau đây một cách hiệu quả:

1. Quản lý sự phụ thuộc giữa các đặc trưng.
Với mỗi đặc trưng, nó cần biết những _dependecy_ cũng như _dependent_ của đặc trưng này.

2. Thêm một đặc trưng mới mà không phải tính toán lại các đặc trưng khác.

3. Cập nhật một đặc trưng sẽ cập nhật toàn bộ các _dependent_ của nó.

4. Một đặc trưng có thể được xóa chỉ khi nó không có _dependent_ nào hoặc tất cả các _dependent_ cũng được xóa cùng lúc.
