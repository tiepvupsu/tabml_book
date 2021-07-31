#!/usr/bin/env python
# coding: utf-8

# # Bộ dữ liệu Movielens 1M
# 
# ## Giới thiệu
# 
# [Movielens](https://grouplens.org/datasets/movielens/) là một nhóm cung cấp các bộ dữ liệu cho các bài toán xây dựng Hệ thống gợi ý. Các bộ dữ liệu trong tập này bao gồm thông tin đánh giá xếp hạng của người dùng tới các bộ phim. Những thông tin về người dùng hay các bộ phim cũng được cung cấp.
# 
# Trong cuốn sách này, chúng ta sẽ sử dụng bộ dữ liệu [Movielens 1M](https://grouplens.org/datasets/movielens/1m/). Bộ dữ liệu này bao gồm xâp xỉ 1 triệu bộ `(user, movie, rating)` từ khoảng 3900 bộ phim và  6040 người dùng.
# 
# Trong các phần tiếp theo, khoảng 90% của số đánh giá sẽ được tách ra làm dữ liệu huấn luyện, 10% còn lại được dùng làm dữ liệu đánh giá.
# 
# ## Download bộ dữ liệu
# Bộ dữ liệu này có thể được download trực tiếp từ trang chủ [Movielens 1M](https://grouplens.org/datasets/movielens/1m/) hoặc sử dụng `tabml.datasets`:

# In[1]:


import tabml.datasets
df_dict = tabml.datasets.download_as_dataframes("movielen-1m")
df_dict.keys()


# Có ba dataframe trong bộ dữ liệu này là `users, movies` và `ratings` lần lượt chứa thông tin của người dùng, bộ phim và các đánh giá.
# 
# 
# ## Khám phá dữ liệu
# 
# ### Rating
# 
# Dưới đây là 10 dòng đầu tiên của dataframe rating. Dữ liệu rating bao gồm thông tin về mã người dùng `UserID`, mã phim `MovieID`, đánh giá trong thang điểm 5 và thời điểm đánh giá `Timestamp`.

# In[2]:


ratings = df_dict["ratings"]
ratings.head(10)


# Phân phối của các điểm đánh giá cho trong biểu đồ dưới đây cho chúng ta thấy rằng điểm 4 được đánh giá nhiều nhất trong khi các điểm 1 và 2 có ít lượng đánh giá nhất. Điều này có thể được giải thích bằng sự thật là người dùng thường đánh giá khi họ rất thích một bộ phim; khi họ không thực sự thích, họ sẽ ít ra đánh giá hơn.

# In[3]:


ratings["Rating"].plot.hist()


# Tiếp theo, chúng ta sẽ xem số lượng bộ phim mà mỗi người dùng đánh giá cũng như số lượng đánh giá mà mỗi bộ phim nhận được:

# In[4]:


ratings["UserID"].value_counts()


# Ta thấy rằng người dùng có mã số 4169 đánh giá tới 2314 bộ phim và 20 là số lượng đánh giá ít nhất mà mỗi người dùng đưa ra. Có thể thấy rằng nhóm tác giả của bộ dữ liệu này đã lọc đi các người dùng có ít đánh giá. Sự lý tưởng này khó đạt được trong thực tế vì phần lớn người dùng không đưa ra đánh giá nào. Việc mỗi người dùng đánh giá nhiều bộ phim khiến cho độ chính xác khi gợi ý được cao hơn.

# In[5]:


ratings["MovieID"].value_counts()


# Ở khía cạnh bộ phim, bộ phim có mã số 2858 được đánh giá nhiều nhất với 3428 lần trong khi rất nhiều bộ phim chỉ nhận được một đánh giá.

# ### Dữ liệu người dùng

# In[6]:


users = df_dict["users"]
users.info()


# Như vậy có 6040 người dùng cùng với đầy đủ các thông tin về giới tính, tuổi, nghề nghiệp và Zip-code. Chúng ta sẽ không sử dụng thông tin về Zip-code vì số lượng các giá trị phân biệt là quá lớn. Chúng ta cùng xem nhanh phân bố của các thông tin về giới tính, tuổi và nghề nghiệp.

# In[7]:


users["Gender"].value_counts()


# Có 4331 người dùng là nam và 1709 người dùng là nữ.

# In[8]:


users["Age"].hist()


# Phần lớn người dùng có độ tuổi từ 18 đến 34, nhóm dưới 18 tuổi có số người dùng nhỏ nhất.
# Dữ liệu về nghệ nghiệp đã được mã hóa thành các số từ 0 đến 20:

# In[9]:


import matplotlib
from matplotlib import pyplot as plt


occupation_mapping = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

occupation_id_count = users["Occupation"].value_counts().to_dict()
occupation_count = {
    occupation_mapping[id]: count for id, count in occupation_id_count.items()
}


matplotlib.rcParams.update({"font.size": 14})
plt.figure(figsize=(20, 10))
plt.bar(x=occupation_count.keys(), height=occupation_count.values())
plt.xticks(rotation=90)
plt.show()


# Không có gì bất ngờ, các bạn sinh viên xuất hiện nhiều trong bộ dữ liệu nhất còn các bác nông dân xuất hiện ít nhất.

# ### Dữ liệu bộ phim

# In[10]:


movies = df_dict["movies"]
movies.info()


# Có 3883 bộ phim với đầy đủ thông tin về tiêu đề (`Title`) và các thể loại (`Genres`). Cùng xem một vài dòng đầu của dataframe này:

# In[11]:


movies.head(10)


# Như vậy, năm sản xuất của bộ phim cũng xuất hiện trong tiêu đề. Thông tin về năm sản xuất cũng hoàn toàn có thể là một đặc trưng tốt cho việc xây dựng mô hình. Ngoài ra, một bộ phim có thể thuộc nhiều thể loại.
# 
# Tiếp theo, chúng ta cùng trả lời hai câu hỏi:
# 
# 1. Số lượng thể loại mà mỗi bộ phim thuộc về.
# 2. Số lượng bộ phim thuộc mỗi thể loại

# In[12]:


movies["num_genres"] = movies["Genres"].apply(lambda x: len(x.split('|')))
movies["num_genres"].value_counts()


# Trả lời cho câu hỏi thứ nhất, hầu hết các bộ phim thuộc vào một thể loại. Số lượng thể loại nhiều nhất mà một bộ phim thuộc về là 6 và chỉ có một bộ phim như vậy.

# In[13]:


from collections import defaultdict

genres_counter = defaultdict(int)
for genre_str in movies["Genres"]:
    genres = genre_str.split('|')
    for genre in genres:
        genres_counter[genre] += 1
        
plt.figure(figsize=(20, 10))
plt.bar(x=genres_counter.keys(), height=genres_counter.values())
plt.xticks(rotation=90)
plt.show();


# Trả lời cho câu hỏi thứ hai, ta thấy rằng thể loại `Drama` và `Comedy` có nhiều bộ phim nhất. Các thể loại `Animation, Fantasy, "Documentary, War, Mystery, Film-Noir` và `Western` có ít bộ phim nhất với khoảng từ 50 đến 100 bộ phim.
