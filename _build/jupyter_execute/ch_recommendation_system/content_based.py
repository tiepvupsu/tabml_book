#!/usr/bin/env python
# coding: utf-8

# # Hệ thống dựa trên nội dung
# 
# ## Giới thiệu
# 
# Hệ thống gợi ý dựa trên nội dung là hệ thống đơn giản nhất. Trong hệ thống này, mô hình dự đoán liệu một người dùng có thích một sản phẩm không dựa trên lịch sử dữ liệu của người dùng đó đối với các sản phẩm tương tự. Độ quan tâm của những người dùng khác không được sử dụng.
# 
# Nhìn dưới góc độ xây dựng mô hình dự đoán, hệ thống xây dựng một mô hình machine learning cho mỗi người dùng. Mỗi sản phẩm sẽ được mô tả bởi một vector đặc trưng. Để dự đoán mức độ yêu thích của mỗi người dùng đối với một sản phẩm, ta chỉ cần đưa vector đặc trưng của sản phẩm vào mô hình đã được xây dựng cho người dùng đó. Bài toán *Hoàn thiện ma trận* bây giờ đơn giản là bài toán *Hoàn thiện vector*.
# 
# ```{figure} imgs/one_user.png
# ---
# name: img_utility_vector
# ---
# Vector Utility
# ```
# 
# 
# **Ưu điểm**:
# 
# * Việc xây dựng mô hình cho mỗi người dùng độc lập với nhau, vì vậy khi có dữ liệu mới từ những người dùng khác, ta không cần cập nhật mô hình cho người dùng này. Việc này giúp hệ thống có thể phục vụ được lượng người dùng lớn một cách dễ dàng.
# 
# * Nếu một sản phẩm rất ít người quan tâm nhưng giống với những sản phẩm khác mà một người dùng từng thích, sản phẩm đó sẽ có cơ hội cao được giới thiệu tới người dùng.
# 
# **Nhược điểm**:
# 
# * Thông tin về những người dùng tương tự có thể rất hữu ích nhưng không được khai thác, làm giảm độ chính xác của mô hình.
# 
# * Mô hình chỉ dựa trên những dữ liệu đã có mà không mở rộng sự yêu thích của người dùng tới những sản phẩm khác.
# 
# * Việc xây dựng đặc trưng cho sản phẩm đôi khi phải được thực hiện thủ công, ví dụ gán nhãn thể loại. Điều này sẽ hạn chế năng lực của hệ thống khi có rất nhiều sản phẩm.
# 
# Mục dưới đây sẽ mang đến cho bạn đọc một ví dụ về việc sử dụng thể loại để xây dựng vector đặc trưng cho các bộ phim và xây dựng mô hình cho mỗi người dùng.

# ## Ví dụ với bộ dữ liệu MovieLens-1M

# ### Ý tưởng
# Bộ dữ liệu MovieLens-1M đã có sẵn các thể loại phim, ta có thể trực tiếp sử dụng các thể loại này để xây dựng vector đặc trưng cho mỗi bộ phim.
# 
# Ta có thể xây dựng vector đặc trưng cho mỗi bộ phim như sau. Vì có 19 thể loại phim, ta xây dựng một vector nhị phân $\mathbf{x}$ trong không gian 19 chiều, mỗi chiều tương ứng với một thể loại. Nếu một bộ phim thuộc vào một thể loại có chỉ số $k$, ta gán phần tử tương ứng $x_k$ giá chị bằng 1. Ngược lại, giá trị của phần tử tương ứng bằng 0.
# 
# Với mỗi người dùng, một mô hình đơn giản là đi tìm mức độ yêu thích của người dùng tới từng thể loại. Ta có thể dùng một vector 19 chiều $\mathbf{w}_i$ để mô tả các mức độ đó. Mỗi đánh giá của người dùng $i$ cho một bộ phim $j$ có thể được mô tả bởi:
# 
# ```{math}
# :label: content_based_lr
# r_{ij} \approx \mathbf{w}_i^T\mathbf{x}_j + b_i = w_i^0 x_j^0 + w_i^1x_j^1 + \dots w_i^{18} x_j^{18} + b_i  
# ```
# 
# Việc xâp xỉ này khá dễ hình dung:
# 
# * Nếu người dùng thích thể loại phim $k$ thì hệ số $w_i^k$ lớn để số hạng $w_i^kx_j^k$ cũng lớn.
# * Nếu người dùng thường xuyên cho đánh giá thấp phim thuộc thể loại $l$, hệ số $w_i^l$ cần nhỏ để số hạng $w_i^lx_j^l$ nhỏ.
# * Hệ số tự do $b_i$ có thể được coi là "độ khó tính" của người dùng. Nếu giá trị này nhỏ, $r_ij$ cùng nhỏ theo và ngược lại.
# 
# Nhận thấy rằng biểu thức {eq}`content_based_lr` chính là dạng của [hồi quy tuyến tính](https://machinelearningcoban.com/2016/12/28/linearregression/). Ta có thể sử dụng [Hồi quy Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) (hồi quy tuyến tính + regularization trên trọng số $\mathbf{w}$) cho hệ thống này.
# 
# ### Triển khai ý tưởng 
# Đầu tiên ta download bộ dữ liệu Movielens-1m và lấy 10% số lượng đánh giá ra làm dữ liệu kiểm thử.
# 
# :::{note}
# Trong bài toán thực tế, ta cần phân chia dữ liệu theo thời gian. Những đánh giá xuất hiện sau nên được tách ra làm bộ kiểm thử. Việc làm này có thể giúp mô hình hóa hành vi của người dùng theo thời gian. Khi đó ta cần dùng thêm cột thời gian trong dữ liệu đánh giá làm đặc trưng. Bạn đọc có thể thực hành bằng cách thay đổi cách phân chia dữ liệu theo thời gian. Để giữ nội dung phần này đơn giản, chúng ta sẽ phân chia dữ liệu ngẫu nhiên và không sử dụng biến thời gian.
# :::
# 
# #### Tải và phân chia dữ liệu

# In[1]:


import pandas as pd
import tabml.datasets
from sklearn.model_selection import train_test_split

df_dict = tabml.datasets.download_movielen_1m()
users, movies, ratings = df_dict["users"], df_dict["movies"], df_dict["ratings"]

train_ratings, validation_ratings = train_test_split(
    ratings, test_size=0.1, random_state=42
)


# In[2]:


users_in_validation = validation_ratings["UserID"].unique()
all_users = users["UserID"].unique()

print(f"There are {len(users_in_validation)} users in validation set.")
print(f"Total number of users: {len(all_users)}")


# Sau khi phân chia dữ liệu, tập kiểm thử (validation set) chỉ chứa dữ liệu của 5970/6040 người dùng. 
# 
# #### Xây dựng đặc trưng cho các bộ phim
# Việc đầu tiên là đánh số thứ tự cho các bộ phim và xây dựng một ánh xạ giữa các chỉ số này tới `MovieID`.

# In[3]:


movie_index_by_id = {id: i for i, id in enumerate(movies["MovieID"])}


# Tiếp theo, ta sẽ xây dựng vector đặc trưng nhị phân cho mỗi bộ phim như mô tả ở trên. Các vector này được gom lại trong một mảng numpy.

# In[4]:


genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
genre_index_by_name = {name:i for i, name in enumerate(genres)}

import numpy as np
# build binary array for movie genres
movie_features = np.zeros((len(movies), len(genres)))
for i, movie_genres in enumerate(movies["Genres"]):
    for genre in movie_genres.split("|"):        
        genre_index = genre_index_by_name[genre]
        movie_features[i, genre_index] = 1


# #### Huấn luyện mô hình
# 
# Với mỗi người dùng, ta sẽ xây dựng một bộ hồi quy Ridge dựa trên các đánh giá cho các bộ phim trong tập huấn luyện:

# In[5]:


from sklearn.linear_model import Ridge


def train_user_model(user_id):
    user_ratings = train_ratings[train_ratings["UserID"] == user_id]
    movie_indexes = [
        movie_index_by_id[movie_id] for movie_id in user_ratings["MovieID"]
    ]
    train_data = movie_features[movie_indexes]
    train_label = user_ratings["Rating"]
    model = Ridge(alpha=0.1)
    model.fit(train_data, train_label)
    return model


# build model for each user
user_model_dict = {}
for user_id in users["UserID"].unique():
    user_model_dict[user_id] = train_user_model(user_id)


# Để dự đoán độ yêu thích của một người dùng tới một bộ phim, ta chỉ cần đưa vector đặc trưng vào mô hình ứng với người dùng đó. Vì các đánh giá nằm trong đoạn từ 1 tới 5, ta cần _cắt_ những dự đoán nằm ngoài khoảng này:

# In[6]:


def predict(user_id, movie_id):
    movie_feature = movie_features[movie_index_by_id[movie_id]].reshape((1, -1))
    pred = user_model_dict[user_id].predict(movie_feature)
    return min(max(pred, 1), 5)


# Cuối cùng, ta có thể đánh giá chất lượng hệ thống trên hai bộ dữ liệu huấn luyện và kiểm thử bằng [Root Mean Square Error - RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation):

# In[7]:


from sklearn.metrics import mean_squared_error

def eval_rmse(ratings: pd.DataFrame) -> float:
    predictions = np.zeros(len(ratings))
    for index, row in enumerate(ratings.itertuples(index=False)):
        predictions[index] = predict(row[0], row[1])
    rmse = mean_squared_error(ratings["Rating"], predictions, squared=False)
    return rmse
    
print(f"RMSE train: {eval_rmse(train_ratings)}")
print(f"RMSE validation: {eval_rmse(validation_ratings)}")
    


# Trên tập huấn luyện, RMSE = 0.93, trên tập kiểm thử, RMSE = 1.04. Như vậy, trên tập kiểm thử, mỗi dự đoán bị lệch khoảng 1.04 điểm. Không quá tệ cho một hệ thống đơn giản.
# 
# #### Kiểm tra kết quả
# Ta cùng xem các hệ số của người dùng có mã số 160:

# In[8]:


user_id = 160
for genre, coef in zip(genres, user_model_dict[user_id].coef_):
    print("{:15s}: {:.3f}".format(genre, coef))


# Từ những hệ số thu được, ta có thể dự đoán được:
# 1. Người dùng này ưa thích các bộ phim về `Adventure` và `Romance`, là những thể loại có trọng số lớn.
# 
# 2. Người dùng này không thích các bộ phim về `Comedy` và `War`, hệ thống không nên gợi ý những bộ phim thuộc hai thể loại này cho người dùng.
# 
# 3. Có một số thể loại như `Documentary, Mystery` và `Western` có hệ số bằng 0. Điều này chứng tỏ có thể người dùng chưa từng đánh giá các phim thuộc thể loại này. Các hệ số bằng 0 này xảy ra nhờ có thành phần regularization trong hồi quy Ridge. Nếu không có ràng buộc từ regularization, hệ số này có thể nhận giá trị bất kỳ mà không ảnh hưởng tới kết quả tìm được do giá trị tương ứng trong vector đặc trưng của các bộ phim mà người dùng này từng đánh giá luôn luôn bằng 0.
# 
# Với một bộ phim mới mà các thể loại chỉ nằm trong số những thể loại người dùng chưa từng đánh giá, hệ thống luôn luôn dự đoán điểm đánh giá bằng hệ số tự do $b_i$. Với người dùng ở vị trị 160, hệ số này bằng:

# In[9]:


user_model_dict[user_id].intercept_


# Đây là một số điểm cao, chỉ cho ta thấy rằng nên gợi ý những phim ở các thể loại này. Với những người dùng khác mà hệ số này thấp một cách ngẫu nhiên, không lẽ hệ thống nên luôn luôn tránh gợi ý? Điều này rõ ràng vô lý vì đây chỉ là một giá trị ngẫu nhiên, hệ thống chưa bao giờ biết độ quan tâm của người dùng tới những bộ phim này.

# ## Thảo luận 
# 
# * Các hệ thống dựa trên nội dung tương đối đơn giản và có thể được huấn luyện một cách nhanh chóng. Việc dự đoán cho mỗi người dùng chỉ dựa trên dữ liệu của người dùng đó mà không quan tâm đến những người dùng khác.
# 
# * Hệ thống dựa trên đặc trưng gặp khó khăn trong các trường hợp người dùng mới khi không có thông tin gì của người dùng đó với các sản phẩm.
# 
# * Bạn đọc có thể thử với các cách xây dựng vector đặc trưng khác cho mỗi bộ phim. Chẳng hạn, thêm đặc trưng phim cũ/mới dựa trên năm ra mắt hoặc chuẩn hóa vector đặc trưng cho mỗi bộ phim sao cho tổng các trọng số cho các thể loại bằng 1.
