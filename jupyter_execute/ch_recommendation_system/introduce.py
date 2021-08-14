# Hệ thống gợi ý

## Giới thiệu

Hệ thống gợi ý (Recommendation system) dần trở thành một thành phần không thể thiếu của các sản phẩm điện tử có nhiều người dùng.
Các sản phẩm cá nhân hóa điện tử ngày càng phổ biến với mục đích mang sản phẩm phù hợp tới người dùng hoặc giúp người dùng có các trải nghiệm tốt hơn.
Nếu quảng cáo sản phẩm tới đúng người dùng, khả năng các món hàng được mua nhiều hơn.
Nếu gợi ý một video mà người dùng nhiều khả năng thích hoặc gợi ý kết bạn đến đúng đối tượng, họ sẽ ở lại trên nền tảng của bạn lâu hơn.
Khi họ ở trên nền tảng của bạn lâu hơn, họ sẽ nhìn thấy nhiều quảng cáo hơn và lợi nhuận từ quảng cáo sẽ lại càng nhiều hơn.

Quảng cáo điện tử ngoài việc giúp các doanh nghiệp bán được nhiều hàng còn giúp họ tiết kiệm được chi phí kho bãi.
Họ sẽ không cần các cửa hàng ở vị trí thuận lợi để thu hút khách hàng hay phải trưng ra mọi mặt hàng ở vị trí đắc địa nhất trong cửa hàng.
Mọi thứ có thể được cá nhân hóa sao cho mỗi người dùng nhìn thấy những sản phẩm khác nhau phù hợp với nhu cầu và sở thích của họ.

Quay ngược lại với quảng cáo truyền thống.
Trước đây, chỉ có các đài truyền hình và báo chí mới có khả năng quảng bá sản phẩm tới lượng lớn người dùng.
Những quảng cáo "giờ vàng" trên truyền hình thường được trả giá rất cao vì lượng khán giả trong giờ đó là cao nhất.
Chi phí cho việc này khá lớn và hiệu quả nhưng lại không mang tính cá nhân hóa.
Toàn bộ người xem truyền hình hay đọc báo đều được xem những quảng cáo đó nhưng không phải ai cũng thực sự quan tâm tới sản phẩm đó.
Không phải ai thức xem bóng đá lúc nửa đêm cũng quan tâm tới máy lọc nước; không phải ai xem thời sự buổi tối cũng đủ tiền để nghĩ tới những sản phẩm xa xỉ đắt tiền.
Bản thân từ "quảng" trong "quảng cáo" đã mang nghĩa loan rộng thông tin và người đăng tin muốn sản phẩm của mình được càng nhiều người biết đến càng tốt.
Việc này dẫn đến những doanh nghiệp nhỏ khó cạnh tranh được các vị trí vào giờ vàng để quảng bá sản phẩm; ngược lại, phần lớn người dùng không được tiếp cận tới những sản phẩm hợp với nhu cầu của họ.

Để tăng tính cá nhân hóa quảng cáo, các đài truyền hình đã tạo ra nhiều kênh khác nhau cho từng nhóm đối tượng.
Các kênh khoa học, giải trí, thiếu nhi, v.v ra đời giúp khán giả có thể xem những nội dung mình yêu thích và giúp nhà đài quảng cáo tới đúng đối tượng hơn.
Thú vị hơn, giải bóng đá Ngoại hạng Anh có thể [truyền các tín hiệu tới các khu vực khác nhau với nội dung ở các biển quảng cáo trên sân khác nhau](https://the18.com/en/soccer-entertainment/virtual-advertising-boards-different-ads-country-channel-premier-league).
Bạn có thể vài lần thấy các thương hiệu Việt Nam trên các nội dung thể thao quốc tế, nhưng điều đó không đồng nghĩa với việc các khán giả ở Mỹ cũng nhìn thấy nội dung quảng cáo đó.
Việc này vừa giảm chi phí quảng cáo cho các thương hiệu Việt vừa mang lại lợi nhuận cao hơn cho đơn vị phát sóng vì mang tới nhiều quảng cáo chất lượng tới nhiều nơi hơn.
Tuy nhiên, việc này cũng chỉ giúp phân loại người dùng ra một số lượng nhóm nhất định.

Quảng cáo trên Internet đã chiếm thị phần ngày càng cao so với quảng cáo truyền hình nhờ sự đa dạng và cá nhân hóa một cách tối đa.
Một người dùng 20-30 tuổi thường xuyên nghe nhạc rap ít có khả năng thích nhạc Bolero.
Một người dùng tìm kiếm các thông tin về xe hơi nhiều khả năng sắp mua xe và quan tâm tới những dịch vụ sửa và rửa xe.
Một người dùng thường xuyên xe các video về làm vườn nhiều khả năng sẽ quan tâm tới việc mua bán hạt giống.
Từ những thông tin thu thập được từ hành vi người dùng, hệ thống có thể gợi ý ra những lựa chọn phù hợp để đạt được hiệu quả cao nhất.

## Ma trận utility

Gợi ý sản phẩm là một bài toán machine learning có giám sát với nhãn dựa trên hành vi của người dùng trong quá khứ. Có hai nhóm đối tượng chính là người dùng và sản phẩm. Đầu vào của hệ thống là những thông tin về người dùng và sản phẩm. Thông tin người dùng có thể là giới tính, tuổi, nghề nghiệp, vị trí địa lý, thời điểm truy cập, trình duyệt, thiết bị, v.v; những thông tin này thường biến đổi theo thời gian. Thông tin về sản phẩm có thể là loại mặt hàng, nơi sản xuất, thời điểm sản xuất, v.v. và là những thông tin ít thay đổi. Nhãn của bài toán này là những hành vi của người dùng có liên quan tới sản phẩm như đã xem, đã mua, v.v.

Dữ liệu của bài toán gợi ý thường được biểu diễn dưới dạng ma trận như hình dưới:

![](imgs/utility_matrix_0.png)

Mỗi hàng thể hiện một người dùng, mỗi cột thể hiện một sản phẩm. Các ô có chấm đen thể hiện đã có thông tin về mức độ quan tâm của người dùng tới sản phẩm tương ứng. Bài toán đặt ra là dự đoán mức độ quan tâm ở những ô trống dựa trên những ô đen đã biết trước giá trị và những thông tin liên quan về người dùng và sản phẩm.

Ma trận trên đây còn được gọi là ma trận utility. Bài toán gợi ý sản phẩm có mối quan hệ chặt chẽ tới bài toán *Hoàn thiện ma trận* (Matrix Completion). 

## Khó khăn và thách thức

**Xây dựng nhãn**: 
Các nhãn có thể được thể hiện một cách tường minh như việc mua sản phẩm hay không, việc đánh giá số sao của người dùng cho sản phẩm, hay việc chấp nhận kết bạn hay không.
Những nhãn này còn được gọi là *phản hồi tường minh* (explicit feedback).
Tuy nhiên, không phải hệ thống gợi ý nào cũng phục vụ cho việc mua bán sản phẩm hay không phải người dùng nào cũng sẵn sàng bỏ thời gian ra đánh giá sản phẩm.
Rất nhiều trường hợp, nhãn được xây dựng dựa trên những *phản hồi ẩn* (implicit feedback) từ người dùng.
Ví dụ, người dùng có thể không mua hàng nhưng họ đã click vào sản phẩm hoặc dành thời gian đọc về thông tin sản phẩm.
Đôi khi, người dùng không click nhưng đã dừng lại ở phần quảng cáo sản phẩm đó trong một thời gian đủ lớn và bật âm thanh lớn để nghe về sản phẩm cũng là một tín hiệu hữu ích.

**Dữ liệu lệch**: Một khó khăn trong việc xây dựng các mô hình gợi ý là việc nhãn thường bị lệch một cách nghiêm trọng. Số lượng mẫu có nhãn dương (có đánh giá tốt, có click, có mua hàng, v.v.) thường rất nhỏ so với lượng mẫu không có phản hồi. Và việc không có phản hồi chưa chắc đã có nghĩa rằng người dùng không quan tâm tới sản phẩm. Sự chênh lệch nhãn này khiến việc xây dựng mô hình trở lên phức tạp hơn. Việc chọn phương pháp đánh giá cũng hết sức quan trọng.

**Hiện tượng đuôi dài**: Không những bị lệch về lượng mẫu có và không có phản hồi mà lượng phản hồi cho các sản phẩm cũng chênh nhau đáng kể. Sẽ có những sản phẩm phổ biến có rất nhiều dữ liệu những cũng có nhiều lần số sản phẩm ít phổ biến có rất ít phản hồi.

**Vòng phản hồi (feedback loop)**: Đôi khi, việc gợi ý cho người dùng dựa hoàn toàn vào phản hồi của họ lại không thực sự thú vị. Nếu một người xem một video về chó mèo và rồi hệ thống gợi ý đúng về các video chó mèo khác và người đó tiếp tục xem thì dần dần người đó sẽ hoàn toàn nhận được các gợi ý về chó mèo mà không có thể loại nào khác. Với các hệ thống gợi ý, nhãn thu được bị ảnh hưởng một phần từ những gì mà hệ thống đã gợi ý trong quá khứ. Nếu tiếp tục phụ thuộc hoàn toàn vào nhãn thì kết quả gợi ý sẽ dần hội tụ về một lượng nhỏ các video. Vòng phản hồi này có tác động tiêu cực tới trải nghiệm người dùng và cần được hạn chế.

**Khởi đầu lạnh (cold start)**: Khởi đầu lạnh xảy ra khi hệ thống không thể đưa ra một gợi ý đáng tin cậy khi lượng dữ liệu có là quá ít. Khi bắt đầu xây dựng hệ thống, khi có người dùng mới, hoặc khi có sản phẩm mới là những trường hợp mà xuất phát lạnh xảy ra. Với những sản phẩm mới, chưa có người dùng nào tương tác với nó, lúc này hệ thống cần có càng nhiều thông tin mô tả về sản phẩm càng tốt để gán nó vào gần với những nhóm đã có tương tác với người dùng. Với những người dùng mới, hệ thống gần như không có thông tin gì về sở thích hay thói quen của họ. Lúc này, hệ thống cần đưa ra những quyết định dựa trên lượng thông tin ít ỏi mà nó có thể suy đoán được như vị trí địa lý, ngôn ngữ, giới tính, tuổi, v.v. Những quyết định ban đầu này có thể ảnh hưởng trực tiếp tới những gợi ý tiếp theo và trải dụng của người dùng. Nếu một hệ thống hoàn toán mới chưa có cả người dùng và sản phẩm, tốt nhất chưa nên sử dụng machine learning mà dựa vào các phương pháp đơn giản khác.

**Tốc độ xử lý**: Cá nhân hóa đồng nghĩa với việc hệ thống phải đưa ra những quyết định khác nhau với mỗi người dùng tại mỗi thời điểm khác nhau.
Tại một thời điểm, có thể có hàng triệu người dùng và hàng triệu sản phẩm mà hệ thống cần gợi ý.
Tốc độ tính toán là một điểm tối quan trọng trong một hệ thống gợi ý. Nếu bạn có thể xây dựng được một hệ thống có thể dự đoán với độ chính xác cao ở một bộ dữ liệu kiểm thử nhưng không thể triển khai trong thực tế thì cũng vô nghĩa.
Nếu trải nghiệm người dùng bị ảnh hưởng bởi tốc độ hiển thị, họ sẽ dần rời khỏi nền tảng.

## Hai nhóm thuật toán chính 

Các thuật toán machine learning trong hệ thống gợi ý thường được chia thành hai nhóm lớn:

**Hệ thống dựa trên nội dung (content-based systems)**: nhóm thuật toán này gợi ý cho người dùng những sản phẩm tương tự như những sản phẩm mà người dùng đã có phản hồi tích cực. Hệ thống này cần xây dựng đặc trưng cho các sản phẩm sao cho những sản phẩm tương tự nhau có khoảng cách tới nhau nhỏ. Việc này khá tương tự như việc xây dựng các embedding cho các sản phẩm. Việc dự đoán cho mỗi người dùng hoàn toàn chỉ dựa trên lịch sử thông tin của người dùng đó.

**Lọc cộng tác (collaborative filtering)**: nhóm thuật toán này không chỉ dựa trên thông tin về sản phẩm tương tự mà còn dựa trên hành vi của những người dùng tương tự. Ví dụ: người dùng A, B, C đều thích các bài hát của Noo Phước Thịnh. Ngoài ra, hệ thống biết rằng B, C cũng thích các bài hát của Bích Phương nhưng chưa có thông tin về việc liệu user A có thích Bích Phương hay không. Dựa trên thông tin của những người dùng tương tự là B và C, hệ thống có thể dự đoán rằng A cũng thích Bích Phương và gợi ý các bài hát của ca sĩ này tới A.

---- 
Trên đây là các thách thức gặp phải khi xây dựng các hệ thống gợi ý. Trong cuốn sách này, các thuật toán từ đơn giản tới phức tạp sẽ được trình bày để giải quyết bài toán gợi ý. Bạn đọc có thể quan tâm tới loại bài về Hệ thống gợi ý trên [Machine Learning cơ bản](https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/).

