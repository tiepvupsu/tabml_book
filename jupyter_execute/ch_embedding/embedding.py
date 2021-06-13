(sec_embedding)=
# Embedding


## Giới thiệu

Embedding là một kỹ thuật đưa một vector có số chiều lớn, thường ở dạng thưa, về một vector có số chiều nhỏ, thường ở dạng dày đặc. Phương pháp này đặc biệt hữu ích với những đặc trưng hạng mục có số phần tử lớn ở đó phương pháp chủ yếu để biểu diễn mỗi giá trị thường là một vector dạng one-hot. Một cách lý tưởng, các giá trị có ý nghĩa  tương tự nhau nằm gần nhau trong không gian embedding.

Ví dụ nổi bật nhất là biểu diễn các từ trong một bộ từ điển dưới dạng số. Khi từ điển có hàng triệu từ,  biểu diễn các từ dưới dạng one-hot vector dẫn tới số chiều vô cùng lớn. Hơn nữa, các từ này sẽ có khoảng cách đều nhau tới mọi từ khác (căn bậc hai của 2), dẫn đến việc thiếu thông tin giá trị cho việc huấn luyện mô hình machine learning. Một cách biểu diễn tốt cần mô tả tốt sự liên quan giữa cặp từ (vua, hoàng_hậu) và (chồng, vợ) vì chúng có ý nghĩa gần nhau.

## Biểu diễn toán học

Giả sử một tự điển nào đó chỉ có sáu giá trị (Hà Nội, Hải Phòng, Tp HCM, Bình Dương, Lào Cai, Sóc Trăng). Hình vẽ dưới đây thể hiện cách biểu diễn của các giá trị này trong không gian one-hot và không gian embedding:

Trong không gian one-hot, các giá trị này không ý nghĩa gì ngoài việc xác định mỗi giá trị có phải là một từ nào đó trong từ điển hay không. Trong không gian embedding, số chiều để biểu diễn đã giảm từ 6 xuống còn 2; các giá trị trở thành dạng số thực thay vì các giá trị nhị phân với chỉ một phần tử bằng 1 như ví dụ trong {numref}`img_onehot_emb`.

```{figure} imgs/emb1.png
---
name: img_onehot_emb
---
Biểu diễn các giá trị hạng mục dưới dạng one-hot vector và embedding
```

Ở đây, các giá trị trong không gian embedding được lấy ví dụ bằng tay với chiều thứ nhất thể hiện dân số và chiều thứ hai thể hiện vĩ độ đã chuẩn hóa của mỗi giá trị. Vị trí của mỗi vector embedding trong không gian hai chiều được minh hoạt trong {numref}`img_exp_emb_viz`. Trong không gian này, Hà Nội, Hải Phòng và Hà Giang gần nhau về vị trí địa lý. Nếu chúng ta có một bài toán nào đó mà dân số có thể là một đặc trưng tốt, ta chỉ cần co trục tung và giãn trục hoành là có thể mang những tỉnh thành có dân số giống nhau gần với nhau hơn.

```{figure} imgs/emb2.png
---
name: img_exp_emb_viz
---
Biểu diễn các vector embedding trong không gian
```

Với một từ điển bất kỳ với $N$ từ $(w_0, w_1, \dots, w_{N-1})$. Giả sử số chiều của không gian embedding là $d$, ta có thể biểu diễn toàn bộ các embedding cho $N$ từ này dưới dạng một ma trận $\mathbf{E} \in \mathbb{R}^{N\times k}$ với hàng thứ $i$ là biểu diễn embedding cho từ $w_{i-1}$.

Nếu vector $\mathbf{o}_i \in \mathbb{R}^{N \times 1}$ là biểu diễn one-hot của từ $w_i$, ta có ngay $\mathbf{e} = \mathbf{o}_i^T\mathbf{E} \in \mathbb{R}^{1 \times k}$  là biểu diễn embedding của từ đó.

## Tạo embedding như thế nào?

Cách biểu diễn các tỉnh thành trong ví dụ trên đây chỉ là một ví dụ minh họa khi chúng ta có thể định nghĩa các trục một cách cụ thể dựa vào kiến thức nhất định đã có về dữ liệu. Cách làm này không khả thi với những dữ liệu vô cùng nhiều chiều và không có những ý nghĩa từng trục rõ ràng như trên. Việc tìm ra ma trận $\mathbf{E}$ cần thông qua một quá trình "học" dựa trên mối quan hệ vốn có của dữ liệu.

Ta có thể thấy rằng ma trận $\mathbf{E}$ có thể được coi là một ma trận trọng số của một tầng tuyến tính trong một mạng học sâu như trong {numref}`img_exp_emb_neural`.

```{figure} imgs/emb3.png
---
name: img_exp_emb_neural
---
Ma trận embedding có thể coi là một ma trận trọng số trong một mạng neural
```

Như vậy, ma trận này cũng có thể được xây dựng bằng cách đặt nó vào một mạng học sâu với một hàm mất mát nào đó. Trong Tensorflow (phiên bản 2.5), tầng embedding có thể được khai báo bởi [`tf.keras.layers.Embedding`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding). Trong Pytorch 1.8.1, ta có thể dùng [`torch.nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html). Trong cuốn sách này, chúng ta sẽ có một ví dụ về việc xây dựng embedding cho các sản phẩm dựa trên nền tảng Pytorch.

Embedding có thể được học trong cả một bài toán tổng thể hoặc học riêng rẽ ở một dữ liệu khác trước khi đưa vào một bài toán cụ thể. Embedding thu được có thể được dùng trong các bài toán khác như một đặc trưng nhiều chiều, chúng cũng có thể được đưa vào các mô hình không phải học sâu.

Word2vec là một trong những phương pháp tiên phong về việc xây dựng embedding dựa trên một mạng học sâu. Các embedding vectors này được học chỉ dựa trên các câu trong một bộ dữ liệu lớn mà không cần biết ý nghĩa cụ thể của từng câu hay mối quan hệ đặc biệt nào giữa chúng. Các embedding vector này có thể được dùng để tạo các biểu diễn cho một câu hay một văn bản để giải quyết các bài toán khác.

Trong mục tiếp theo, chúng ta sẽ tìm hiểu thuật toán word2vec. Sau đó

