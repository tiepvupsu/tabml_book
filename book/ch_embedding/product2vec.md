---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec_prod2vec)=
# Instacart Product2vec

## Giới thiệu

[Instacart.com](https://www.instacart.com/) là trang web cho phép người dùng mua bán đồ ăn tươi và các vật phẩm gia dụng online. Một nhân viên giao hàng sẽ đi mua hàng giúp người dùng và giao trong thời gian rất ngắn. Năm 2017, instacart công bố một bộ dữ liệu và tổ chức [một cuộc thi trên Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis). Cuộc thi này yêu cầu người chơi dự đoán những sản phẩm mà người dùng sẽ mua ở một thời điểm nhất định dựa trên những đơn hàng trước đó của họ và nhiều người dùng khác. Một vài thông số về bộ dữ liệu này có thể được tìm thấy tại bài báo [3 Million Instacart Orders, Open Sourced](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2).

Việc có được embedding của các sản phẩm sẽ rất hữu ích trong việc gợi ý những sản phẩm _tương tự_ cho người dùng dựa trên thói quen của họ. Trong mục này, chúng ta sẽ áp dụng ý tưởng của word2vec để xây dựng một mô hình _product2vec_ học các embedding từ dữ liệu huấn luyện.

Có một điểm thú vị là dữ liệu cung cấp các đơn hàng (order) và thứ tự các loại sản phẩm (product) trong đơn hàng đó. Nếu người dùng trực tiếp mua ở cửa hàng, thứ tự mua hàng có thể bị xáo trộn trong quá trình thanh toán vì tất cả đã được cho vào một giỏ. Trong trường hợp này, do người dùng mua hàng online nên hệ thống có thể lưu lại được thứ tự các mặt hàng mà họ đặt vào "giỏ".

Nếu ta coi mỗi sản phẩm là một từ thì mỗi đơn hàng có thể được coi là một câu văn. Những sản phẩm gần nhau trong thứ tứ đặt hàng thường có mối quan hệ ngữ cảnh nào đó. Chẳng hạn, nếu hai sản phẩm đầu tiên được cho vào giỏ là "bún" và "đậu phụ" thì các sản phẩm tiếp theo nhiều khả năng cao là "mắm tôm", "rau sống", "chanh". Nếu biết các sản phẩm ngữ cảnh là "bún", "đậu phụ", "rau sống" và "chanh", hệ thống sẽ dựa trên dữ liệu trong quá khứ để tính xác suất sản phẩm đích là "mắm tôm".

Bộ dữ liệu này cũng có thể được tìm thấy tại [dataset repo](https://github.com/tiepvupsu/tabml_data/tree/master/instacart) của cuốn sách này.


Chúng ta cùng triển khai code python cho việc xây dựng embedding cho các sản phẩm. Việc huấn luyện mô hình được thực hiện dựa trên thư viện [pytorch-lightning](https://www.pytorchlightning.ai/).

## Tiền xử lý dữ liệu

Trước hết chúng ta khai báo các thư viện cần thiết và đặt `seed` cho các thành phần ngẫu nhiên.

```{code-cell} ipython3
from collections import Counter
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
import tqdm


GLOBAL_SEED = 42  # number of life
torch.manual_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
```

Tiếp theo ta tải dữ liệu của những đơn hàng trước đây trong `order_products__train.csv`. File `order_products__prior.csv` chứa nhiều hơn những đơn hàng trong quá khứ và có thể giúp các embedding có độ chính xác tốt hơn; tuy nhiên, trong ví dụ này chúng ta sẽ sử dụng tập dữ liệu nhỏ hơn để kiểm tra tính khả thi của thuật toán. Bạn đọ có thể sử dụng thêm cả `order_products__prior.csv` để có kết quả tốt hơn. Việc thay đổi code không quá phức tạp.

```{code-cell} ipython3
# TODO: use the github link
instacart_path = (
    "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/instacart/"
)

order_df = pd.read_csv(instacart_path + "order_products__train.csv")
# order_df = pd.read_csv("order_products__train.csv")
print(order_df.info())
order_df.head(5)
```

Có tổng cộng 1384617 sản phẩm trong các đơn hàng kể cả lặp. Cột `order_id`, `product_id`, `add_to_cart_order` lần lượ là mã đơn, mã sản phẩm, và thứ tự của sản phẩm trong đơn. Ta cần chuyển dữ liệu này về dạng một list các đơn hàng, mỗi đơn là một list các mã sản phẩm theo thứ tự chúng được đặt vào giỏ hàng.

```{code-cell} ipython3
def get_list_orders(order_df: pd.DataFrame) -> List[List[int]]:
    order_df = order_df.sort_values(by=["order_id", "add_to_cart_order"])
    return order_df.groupby("order_id")["product_id"].apply(list).tolist()

all_orders = get_list_orders(order_df)
print(f"Number of orders: {len(all_orders)}")
print(f"First 3 orders: {all_orders[:3]}")
```

Như vậy có tổng cộng 131209 đơn hàng. Trong mỗi đơn hàng là các mã sản phẩm. Ta sẽ lọc bỏ đi các đơn hàng chỉ có một sản phẩm vì chúng không tạo ra các "ngữ cảnh" cho việc huấn luyện.

```{code-cell} ipython3
min_product_per_order = 2
orders = [order for order in all_orders if len(order) >= min_product_per_order]
print(f"Number of orders with at least {min_product_per_order} products: {len(orders)}")
```

Ta cần biết tên của từng sản phẩm để kiểm tra chất lượng của các embedding thu được ở cuối.

Thông tin sản phẩm được lưu ở file `products.csv`.

```{code-cell} ipython3
product_df = pd.read_csv(
    instacart_path + "products.csv", usecols=["product_id", "product_name"]
)
product_df.head(5)
```

Ta sẽ xây dựng dictionary ánh xạ giữa mã sản phẩm và tên sản phẩm để tiện tra cứu về sau:

```{code-cell} ipython3
def get_product_name_by_id(product_df: pd.DataFrame) -> Dict[int, str]:
    return product_df.set_index('product_id').to_dict()['product_name']

product_name_by_id = get_product_name_by_id(product_df)
print(f"Number of product: {len(product_name_by_id)}")
print(list(product_name_by_id.items())[:5])
```

Có tổng cộng gần 50 nghìn sản phẩm trong bộ dữ liệu này. Số lượng sản phẩm này bao gồm cả những sản phẩm trong các đơn hàng ở `order_products__prior.csv`. Ta cần lọc bỏ đi những sản phẩm không xuất hiện trong `orders`, đồng thời coi những sản phẩm "hiếm" ít được đặt là "<UNKNOWN>".

```{code-cell} ipython3
ordered_products = [product for order in orders for product in order]
product_freq = Counter(ordered_products)
unique_ordered_products = set(ordered_products)
print("Number of products in orders:", len(unique_ordered_products))

min_frequency = 10 ## products appear < min_fequency times are considered as <UNKNOWN>

rare_products = [
    product for product in unique_ordered_products if product_freq.get(product) < min_frequency
]
print("Number of rare products:", len(rare_products))
```

```{code-cell} ipython3
class Product:
    def __init__(self, product_name_by_id, orders, min_frequency=2):
        self.product_name_by_id = product_name_by_id.copy()
        self.orders = orders
        self.min_frequency = min_frequency
        self._ordered_products = [product for order in orders for product in order]
        self._rare_products = self.find_rare_products()
        self.rare_name = "<UNKNOWN>"
        self.name_by_id = self.get_name_by_id()
        self.index_by_id, self.name_by_index = self.get_mappings()

    def find_rare_products(self):
        product_freq = Counter(self._ordered_products)
        unique_ordered_products = set(self._ordered_products)
        return [
            product for product in unique_ordered_products if product_freq[product] < self.min_frequency
        ]

    def get_name_by_id(self):
        # count = 0 -> ignore
        # count < min_freqency -> <UNKNOWN>
        product_freq = Counter(self._ordered_products)
        res = dict()
        for product_id in set(self._ordered_products):
            if product_freq[product_id] < self.min_frequency:
                res[product_id] = self.rare_name
            else:
                res[product_id] = self.product_name_by_id[product_id]
        return res

    def get_mappings(self):
        total_product_ids = len(self.name_by_id)
        num_rare_products = len(self._rare_products)
        total_products = total_product_ids - num_rare_products + 1
        index_by_id = dict()
        name_by_index = dict()
        index = 0
        for product_id, product_name in self.name_by_id.items():
            # rare product holds the last index
            if product_id in self._rare_products:
                index_by_id[product_id] = total_products - 1
                name_by_index[total_products - 1] = self.rare_name
            else:
                index_by_id[product_id] = index
                name_by_index[index] = product_name
                index += 1
        return index_by_id, name_by_index

product_mapping = Product(product_name_by_id, orders, min_frequency=0)
len(product_mapping.index_by_id)
```

```{code-cell} ipython3
print(len(product_mapping.name_by_id))
print(len(product_mapping.name_by_index))
len(product_mapping._rare_products)
```

Mapping

```{code-cell} ipython3
orders = [[product_mapping.index_by_id[product_id] for product_id in order] for order in orders]
```

## Xây dựng dữ liệu huấn luyện

Với mỗi sản phẩm đích `targer_product`, ta sẽ xây dựng một bộ ba `(targer_product, context_products, labels)`. Trong đó:

* `context_products` là mảng gồm các sản phẩm ngữ cảnh dương (`positive_context_products`) VÀ các sản phẩm không trong ngữ cảnh đó tìm được qua phép lấy mẫu âm, tạm gọi là những sản phẩm ngữ cảnh âm (`negative_context_products`).

* `labels` là một mảng nhị phân có độ dài bằng với `context_products` để phân biệt `positive_context_products` và `negative_context_products`. Mảng nhị phân cũng được dùng để tính giá trị hàm mất mát.

Để có thể huấn luyện dưới dạng batch, độ dài của `context_products` cần phải giống nhau giữa các mẫu huấn luyện. Do số sản phẩm ngữ cảnh có độ dài biến đổi tùy thuộc vào vị trí của sản phẩm đích trong đơn hàng, ta sẽ chọn số lượng sản phẩm _âm_ sao cho tổng số phần tử trong `context_products` bằng hằng số.

Trước tiên, ta đi xây dựng những thành phần _cố định_ của mỗi mẫu huấn luyện. Các thành phần không cố định từ phép lấy mẫu âm sẽ được thêm vào trong quá trình huấn luyện.

### Xây dựng dữ liệu ngữ cảnh dương

```{code-cell} ipython3
context_window = 2
# total number of context products, including positive and negative products
all_targets = []
all_positive_contexts = []
for order in tqdm.tqdm(orders):
    for i, product in enumerate(order):
        all_targets.append(product)
        positive_context = [
            order[j]
            for j in range(
                max(0, i - context_window), min(len(order), i + context_window + 1)
            )
            if j != i
        ]
        all_positive_contexts.append(positive_context)

print("Samples:")
for i in range(3):
    print(f"Target product: {all_targets[i]}", end = ", ")
    print(f"Positive context products: {all_positive_contexts[i]}")
```

### Xây dựng bộ lấy mẫu âm

Theo

```{code-cell} ipython3
# define sampling weights for products
def get_sampling_weights(orders):
    product_freq = Counter([product for order in orders for product in order])
    sampling_weights = [0 for _ in product_freq]
    for product_index, count in product_freq.items():
        sampling_weights[product_index] = count**0.75
    return sampling_weights

sampling_weights = get_sampling_weights(orders)
```

```{code-cell} ipython3
from numpy.random import choice
import random


class ProductSampler:
    def __init__(self, products, weights, pre_drawn=10_000_000):
        self.products = products
        self.weights = weights
        self.i = pre_drawn - 1
        self.pre_drawn = pre_drawn
        self.pre_drawn_products = []

    def draw(self):
        if self.i == self.pre_drawn - 1:
            # re draw
            self.pre_drawn_products = random.choices(
                population=self.products, weights=self.weights, k=self.pre_drawn
            )
            self.i = -1
        self.i += 1
        return self.pre_drawn_products[self.i]


sampler = ProductSampler([1, 2, 3], [2, 3, 5])
print([sampler.draw() for _ in range(10)])
```

```{code-cell} ipython3
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset

product_sampler = ProductSampler(
    products=range(len(sampling_weights)),
    weights=sampling_weights,
    pre_drawn=10_000_000,
)


class TargetContextDataset(Dataset):
    """Dataset class that returns a pair of (context, target) product ids.

    The pair is a random combination of 2 products in the same order.

    """

    def __init__(
        self,
        all_targets,
        all_positive_contexts,
        product_sampler,
        num_context_products=10,
    ):
        self.all_targets = all_targets
        self.all_positive_contexts = all_positive_contexts
        self.num_context_products = num_context_products
        self.product_sampler = product_sampler

    def __len__(self):
        return len(self.all_targets)

    def __getitem__(self, index):
        target = self.all_targets[index]
        positive_contexts = self.all_positive_contexts[index].copy()
        num_pos = len(positive_contexts)
        num_neg = self.num_context_products - len(positive_contexts)
        mask = [1] * num_pos + [0] * num_neg
        #         negative_contexts = [0 for _ in range(num_neg)]
        while len(positive_contexts) < self.num_context_products:
            product = self.product_sampler.draw()
            if product not in positive_contexts:  #
                positive_contexts.append(product)

        contexts = torch.IntTensor(positive_contexts)
        mask = torch.FloatTensor(mask)
        return torch.IntTensor([target]), contexts, mask


training_data = TargetContextDataset(
    all_targets, all_positive_contexts, product_sampler, num_context_products=20
)
```

```{code-cell} ipython3
import time

for i in range(1):
    batch_size = 2 ** (i + 12)
    train_dataloader2 = DataLoader(
        training_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    t0 = time.time()
    count = 0
    for batch in tqdm.tqdm(train_dataloader2):
        #         print(batch)
        count += 1
        if count == 2:
            break

    print(f"batch_size = {batch_size}: execution time {time.time() - t0}")
```

```{code-cell} ipython3
# 7. Define loss function


class SigmoidBCELoss(nn.Module):
    "BCEWithLogitLoss with masking on call."

    def __init__(self):
        super().__init__()

    def forward(self, inputs, label):
        inputs = torch.reshape(inputs, (inputs.shape[0], -1))
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, label, reduction="none")

        return torch.mean(out)

loss_fn = SigmoidBCELoss()
```

```{code-cell} ipython3
# 8. Define pytorch lightning class
train_dataloader = DataLoader(
        training_data, batch_size=8192, shuffle=True, num_workers=12
    )

class Prod2VecModel(pl.LightningModule):
    def __init__(self, num_products, embed_size: int = 50):
        super().__init__()
        self.embed_size = embed_size
        self.embed_t = nn.Embedding(num_products, self.embed_size)
        self.embed_c = nn.Embedding(num_products, self.embed_size)


    def forward(self, targets, contexts):
        v = self.embed_t(targets)      
        u = self.embed_c(contexts)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    def training_step(self, batch, batch_idx):
        targets, contexts, labels = batch
        output = self.forward(targets, contexts)
        loss = loss_fn(output, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=5e-3, # weight_decay=1e-6
        )  # learning rate
        return optimizer

# 9. Train and save model

num_products = len(sampling_weights)
embed_size = 100
model = Prod2VecModel(num_products, embed_size)
trainer = pl.Trainer(gpus=1, max_epochs=50) # max_steps=1000)
# trainer = pl.Trainer(gpus=1, max_steps=30) # max_steps=1000)
trainer.fit(model, train_dataloader, train_dataloader)
```

```{code-cell} ipython3
torch.save(model.state_dict(), 'model_v3.pt')
model2 = torch.load('model_v3.pt')
embs = model2['embed_t.weight']
embs_arr = embs.detach().numpy()
```

```{code-cell} ipython3
from tabml.utils import embedding

def find_similar(embs_arr, ind, names):
#     ids = embedding.find_nearest_neighbors(embs_arr[ind], embs_arr, measure="cosine", k=3)
    ids = embedding.NearestNeighbor(embs_arr, measure="cosine").find_nearest_neighbors(embs_arr[ind], k=3)
    return [names[ind] for ind in ids]

def find_similar_by_name(embs_arr, sub_name, names):
    ids = [ind for ind in range(len(names)) if sub_name in names[ind]]
    for ind in ids[:3]:
        print('==========')
        print(f'Similar items of "{names[ind]}":')
        print(find_similar(embs_arr, ind, names))
        
# product_name_by_index = {index: name for name, index in product_mapping.index_by_name.items()}
names = list(product_mapping.name_by_index.values())
find_similar_by_name(embs_arr, 'Organic Yogurt', names)
```

```{code-cell} ipython3
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE

X2 = TSNE(n_components=2, perplexity=10).fit_transform(embs_arr)
# X2 = PCA(n_components=2).fit_transform(embs_arr)
```

```{code-cell} ipython3
from matplotlib import pyplot as plt
plt.figure(figsize=(20, 20))
colors = ['b'] * len(product_mapping.name_by_index)
s = [1] * len(product_mapping.name_by_index)
for i, product in product_mapping.name_by_index.items():
    if "Organic Yogurt" in product:
        colors[i] = 'r'
        s[i] = 30
#     if "Cream" in product:
#         colors[i] = 'y'
plt.scatter(X2[:,0], X2[:,1], c=colors, s=s)
```

```{code-cell} ipython3
# norm vs frequency

norm = np.sqrt((embs_arr**2).sum(axis=1))
```

```{code-cell} ipython3
product_freq = Counter([product for order in orders for product in order])
```

```{code-cell} ipython3
freqs = [0]*len(product_freq)
for product_index, freq in product_freq.items():
    freqs[product_index] = freq
```

```{code-cell} ipython3
# plt.scatter(freqs,  norm, logx=True)

fig = plt.figure()
ax = plt.gca()
ax.scatter(freqs , norm)
# ax.set_yscale('log')
ax.set_xscale('log')
```

```{code-cell} ipython3
product_mapping.name_by_id[1]
```

```{code-cell} ipython3
for id in range(7, 100):
    assert product_mapping.name_by_index[product_mapping.index_by_id[id]] == product_mapping.name_by_id[id]
```

```{code-cell} ipython3

```
