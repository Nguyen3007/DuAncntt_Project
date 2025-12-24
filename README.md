# Dự Án CNTT - Hệ Thống Gợi Ý Sản Phẩm

Dự án nghiên cứu và triển khai các mô hình Graph-based Collaborative Filtering cho hệ thống gợi ý sản phẩm, sử dụng PyTorch.

## Tổng Quan

Dự án này triển khai bốn mô hình học sâu phổ biến cho bài toán Collaborative Filtering với implicit feedback:

- **LightGCN** - Mô hình Graph Convolutional Network tối giản cho gợi ý
- **NGCF** - Neural Graph Collaborative Filtering với bi-interaction
- **MFBPR** - Matrix Factorization with Bayesian Personalized Ranking
- **ALS** - Alternating Least Squares (baseline sử dụng thư viện implicit)

Các mô hình được thiết kế để hoạt động trên đồ thị lưỡng phân user-item, hỗ trợ cả trọng số nhị phân và trọng số time-decay.

Liên kết Kaggle Notebook xử lý dữ liệu: https://www.kaggle.com/code/nguyenlez/xuly-data-duan
## Cấu Trúc Dự Án

```
DuAncntt_Project/
├── src/                          # Mã nguồn chính
│   ├── models/                   # Các mô hình neural network
│   │   ├── LightGCN.py          # Mô hình LightGCN
│   │   ├── NGCF.py              # Mô hình NGCF
│   │   └── MFBPR.py             # Mô hình Matrix Factorization BPR
│   ├── data_utils/              # Công cụ xử lý dữ liệu
│   │   ├── dataloader.py        # Data loader chung
│   │   ├── graph_builder.py     # Xây dựng đồ thị nhị phân
│   │   └── graph_builder_time_decay.py  # Xây dựng đồ thị với time-decay
│   ├── trainer.py               # Unified trainer entrypoint
│   └── evaluator.py             # Unified evaluator entrypoint
├── trainer/                      # Scripts huấn luyện
│   ├── train_lightGCN_v2.py     # Huấn luyện LightGCN
│   ├── train_ngcf.py            # Huấn luyện NGCF
│   ├── train_mf_bpr.py          # Huấn luyện MFBPR
│   ├── train_als.py             # Huấn luyện ALS
│   └── resume_lightgcn.py       # Tiếp tục huấn luyện LightGCN
├── evaluate/                     # Scripts đánh giá
│   ├── evaluate_lightgcn.py     # Đánh giá LightGCN
│   ├── evaluate_ngcf.py         # Đánh giá NGCF
│   └── evaluate_mfbpr.py        # Đánh giá MFBPR
├── data/                         # Dữ liệu
│   ├── h_m/                     # Dataset H&M
│   │   ├── train.txt            # Dữ liệu training
│   │   ├── val.txt              # Dữ liệu validation
│   │   ├── test.txt             # Dữ liệu testing
│   │   ├── user_id_map_9m.csv   # Mapping user IDs
│   │   └── item_id_map_9m.csv   # Mapping item IDs
│   └── vibrent/                 # Dataset Vibrent
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── train_time_weights_vibrent.csv
│       ├── user_id_map_vibrent.csv
│       └── item_id_map_vibrent.csv
├── train.py                      # Unified training entrypoint
├── evaluate.py                   # Unified evaluation entrypoint
├── inspect_checkpoint.py         # Tool kiểm tra checkpoint
└── requirements.txt              # Thư viện cần thiết
```

## Các Mô Hình

### 1. LightGCN (Light Graph Convolutional Network)

LightGCN là phiên bản đơn giản hóa của GCN, loại bỏ các thành phần không cần thiết như feature transformation và nonlinear activation.

**Đặc điểm:**
- Chỉ sử dụng neighborhood aggregation để lan truyền embedding
- Kết hợp embeddings từ tất cả các layers
- BPR loss với L2 regularization
- Không sử dụng self-loop

**Công thức:**
```
x^(k+1) = A_hat @ x^(k)
final_embedding = mean(x^(0), x^(1), ..., x^(K))
```

### 2. NGCF (Neural Graph Collaborative Filtering)

NGCF mở rộng GCN bằng cách thêm bi-interaction giữa user và neighbor embeddings.

**Đặc điểm:**
- Message passing với hai thành phần: sum và bi-interaction
- Sử dụng LeakyReLU activation
- Message dropout để tránh overfitting
- Sử dụng self-loop (A + I)

**Công thức:**
```
message = LeakyReLU(W_gc @ neighbor_emb) + LeakyReLU(W_bi @ (emb ⊙ neighbor_emb))
```

### 3. MFBPR (Matrix Factorization with BPR)

MFBPR là mô hình matrix factorization cơ bản được huấn luyện với BPR loss, phù hợp làm baseline đơn giản.

**Đặc điểm:**
- Mô hình đơn giản với user và item embeddings
- Score được tính bằng inner product: `score(u,i) = <p_u, q_i>`
- Huấn luyện với BPR loss
- Không sử dụng graph structure
- Nhanh và dễ triển khai

**Công thức:**
```
score(u, i) = user_emb[u] · item_emb[i]
```

### 4. ALS (Alternating Least Squares)

ALS là thuật toán matrix factorization cổ điển, được sử dụng làm baseline.

**Đặc điểm:**
- Tối ưu hóa xen kẽ user và item factors
- Không cần backpropagation
- Nhanh và hiệu quả cho datasets lớn

## Định Dạng Dữ Liệu

Dự án sử dụng định dạng text đơn giản cho dữ liệu:

### train.txt, val.txt, test.txt
```
user_id item1 item2 item3 ...
```

Mỗi dòng bắt đầu bằng user_id, theo sau là danh sách các item IDs mà user đã tương tác.

### train_time_weights.csv (Tùy chọn)
```
u,v,weight
0,123,0.95
0,456,0.87
```

File CSV chứa trọng số time-decay cho mỗi cặp user-item, trong đó:
- `u`: user ID
- `v`: item ID
- `weight`: trọng số (0-1), items gần đây có trọng số cao hơn

### user_id_map.csv và item_id_map.csv (Tùy chọn)
```csv
original_id,mapped_id
customer_123abc,0
customer_456def,1
```

Các file mapping để chuyển đổi giữa ID gốc và ID đã được mã hóa:
- `original_id`: ID ban đầu từ dataset gốc (string hoặc số)
- `mapped_id`: ID đã được mã hóa (integer liên tục từ 0)
- Giúp theo dõi và debug khi làm việc với dữ liệu thực tế

## Quick Start

### Cài Đặt Nhanh

```bash
# Clone repository (thay <username> bằng tên người dùng GitHub của bạn)
git clone https://github.com/<username>/DuAncntt_Project.git
cd DuAncntt_Project

# Cài đặt dependencies
pip install -r requirements.txt

# Huấn luyện mô hình LightGCN trên H&M dataset
python train.py --model lightgcn --data_dir data/h_m --emb_dim 64 --epochs 30

# Đánh giá mô hình
python evaluate.py --model lightgcn \
  --checkpoint checkpoints/lightgcn_hm_best.pt \
  --data_dir data/h_m
```

## Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- CUDA (tùy chọn, để sử dụng GPU)

### Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

Thư viện chính:
- `torch` - Framework deep learning
- `numpy` - Xử lý mảng số
- `pandas` - Xử lý dữ liệu dạng bảng

Thư viện bổ sung cho ALS (cài đặt riêng nếu cần):
```bash
pip install scipy implicit
```

*Lưu ý: scipy và implicit không có trong requirements.txt, cài đặt riêng khi muốn sử dụng ALS.*

## Sử Dụng

### Cách 1: Sử Dụng Unified Entrypoints (Khuyến Nghị)

Dự án cung cấp hai entrypoints thống nhất để huấn luyện và đánh giá tất cả các mô hình:

**Huấn luyện bất kỳ mô hình nào:**
```bash
python train.py --model <model_name> [model_specific_args]
```

**Đánh giá bất kỳ mô hình nào:**
```bash
python evaluate.py --model <model_name> [model_specific_args]
```

Trong đó `<model_name>` có thể là: `lightgcn`, `ngcf`, `mfbpr`, hoặc `als` (chỉ cho training).

**Ưu điểm của unified entrypoints:**
- Giao diện thống nhất cho tất cả các mô hình
- Dễ dàng chuyển đổi giữa các mô hình khác nhau
- Tự động quản lý PYTHONPATH
- Phù hợp cho automation và scripts

**Ví dụ:**
```bash
# Huấn luyện LightGCN
python train.py --model lightgcn --data_dir data/h_m --emb_dim 64 --epochs 30

# Huấn luyện MFBPR
python train.py --model mfbpr --data_dir data/h_m --emb_dim 64 --epochs 50

# Đánh giá LightGCN
python evaluate.py --model lightgcn --checkpoint checkpoints/lightgcn_hm_best.pt --data_dir data/h_m
```

### Cách 2: Sử Dụng Scripts Riêng Lẻ

Bạn cũng có thể gọi trực tiếp các scripts trong thư mục `trainer/` và `evaluate/` để có toàn quyền kiểm soát.

### 1. Huấn Luyện LightGCN

**Huấn luyện với đồ thị nhị phân:**
```bash
python trainer/train_lightGCN_v2.py \
  --data_dir data/h_m \
  --emb_dim 64 \
  --n_layers 2 \
  --lr 5e-4 \
  --batch_size 16384 \
  --epochs 30 \
  --device cuda
```

**Huấn luyện với time-decay:**
```bash
python trainer/train_lightGCN_v2.py \
  --data_dir data/h_m \
  --use_time_decay \
  --time_weight_csv data/h_m/train_time_weights.csv \
  --emb_dim 64 \
  --n_layers 2 \
  --lr 5e-4 \
  --batch_size 16384 \
  --epochs 30 \
  --device cuda
```

**Tham số quan trọng:**
- `--emb_dim`: Kích thước embedding (mặc định: 64)
- `--n_layers`: Số lớp GCN (mặc định: 2)
- `--reg_weight`: Trọng số L2 regularization (mặc định: 1e-3)
- `--lr`: Learning rate (mặc định: 5e-4)
- `--batch_size`: Kích thước batch (mặc định: 16384)
- `--early_stop_patience`: Số epoch chờ trước khi dừng sớm (mặc định: 3)

### 2. Huấn Luyện NGCF

```bash
python trainer/train_ngcf.py \
  --data_dir data/h_m \
  --emb_dim 64 \
  --layer_sizes 64 64 \
  --lr 1e-3 \
  --batch_size 4096 \
  --epochs 20 \
  --mess_dropout 0.1 \
  --device cuda
```

**Tham số NGCF:**
- `--layer_sizes`: Kích thước các hidden layers (mặc định: [64, 64])
- `--mess_dropout`: Message dropout rate (mặc định: 0.1)
- `--leaky_relu_slope`: Slope của LeakyReLU (mặc định: 0.2)

### 3. Huấn Luyện MFBPR

```bash
python trainer/train_mf_bpr.py \
  --data_dir data/h_m \
  --emb_dim 64 \
  --lr 1e-3 \
  --batch_size 8192 \
  --epochs 50 \
  --device cuda
```

**Tham số MFBPR:**
- `--emb_dim`: Kích thước embedding (mặc định: 64)
- `--reg_weight`: Trọng số L2 regularization (mặc định: 1e-4)
- `--lr`: Learning rate (mặc định: 1e-3)
- `--batch_size`: Kích thước batch (mặc định: 8192)

### 4. Huấn Luyện ALS

```bash
python trainer/train_als.py \
  --data_dir data/h_m \
  --factors 64 \
  --iterations 15 \
  --regularization 0.01 \
  --alpha 1.0
```

**Tham số ALS:**
- `--factors`: Số chiều latent factors (mặc định: 64)
- `--iterations`: Số vòng lặp ALS (mặc định: 15)
- `--regularization`: Regularization strength (mặc định: 0.01)
- `--alpha`: Confidence scaling (mặc định: 1.0)

### 5. Đánh Giá Mô Hình

**Đánh giá LightGCN:**
```bash
python evaluate/evaluate_lightgcn.py \
  --data_dir data/h_m \
  --checkpoint checkpoints/lightgcn_hm_best.pt \
  --split test \
  --K 20 \
  --device cuda
```

**Đánh giá NGCF:**
```bash
python evaluate/evaluate_ngcf.py \
  --data_dir data/h_m \
  --checkpoint checkpoints/ngcf_hm_best.pt \
  --split test \
  --K 20 \
  --device cuda
```

**Đánh giá MFBPR:**
```bash
python evaluate/evaluate_mfbpr.py \
  --data_dir data/h_m \
  --checkpoint checkpoints/mfbpr_hm_best.pt \
  --split test \
  --K 20 \
  --device cuda
```

**Metrics:**
- **Precision@K**: Tỷ lệ items đúng trong top-K
- **Recall@K**: Tỷ lệ items đúng được tìm thấy
- **HitRate@K**: Tỷ lệ users có ít nhất 1 item đúng trong top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision

### 6. Tiếp Tục Huấn Luyện

Nếu muốn tiếp tục huấn luyện từ một checkpoint:

```bash
python trainer/resume_lightgcn.py \
  --checkpoint checkpoints/lightgcn_hm_last.pt \
  --additional_epochs 10
```

### 7. Kiểm Tra Checkpoint

Dự án cung cấp tool để kiểm tra thông tin chi tiết của checkpoint:

```bash
python inspect_checkpoint.py --checkpoint checkpoints/lightgcn_hm_best.pt
```

**Với tham số bổ sung:**
```bash
python inspect_checkpoint.py \
  --checkpoint checkpoints/lightgcn_hm_best.pt \
  --show_args
```

Tool này sẽ hiển thị:
- Loại mô hình (LightGCN, NGCF, MFBPR)
- Kích thước file
- Thông tin epoch và metrics tốt nhất
- Shape của embeddings
- Số lượng users và items
- Hyperparameters chính
- Tổng số parameters
- Full args dict (nếu dùng `--show_args`)

## Tính Năng Time-Decay

Time-decay giúp mô hình chú trọng hơn vào các tương tác gần đây của user.

**Cách hoạt động:**
1. Mỗi tương tác được gán trọng số dựa trên thời gian (càng gần đây càng cao)
2. Đồ thị được xây dựng với các cạnh có trọng số thay vì nhị phân
3. Normalization được điều chỉnh để tính đến trọng số

**Khi nào sử dụng:**
- Dataset có timestamp cho mỗi tương tác
- Sở thích người dùng thay đổi theo thời gian
- Items mới quan trọng hơn items cũ

## Early Stopping

Tất cả các script huấn luyện đều hỗ trợ early stopping dựa trên Recall@K trên tập validation:

- Mô hình được đánh giá sau mỗi epoch
- Checkpoint tốt nhất được lưu khi Recall@K cải thiện
- Huấn luyện dừng nếu không cải thiện sau `early_stop_patience` epochs

## Checkpoints

Mỗi lần huấn luyện tạo ra hai checkpoints:

1. **best model** (`*_best.pt`): Mô hình tốt nhất trên validation set
   - Dùng để đánh giá final trên test set
   - Chứa: model weights, hyperparameters, best metrics

2. **last model** (`*_last.pt`): Checkpoint ở epoch cuối cùng
   - Dùng để tiếp tục huấn luyện
   - Chứa: model weights, optimizer state, training progress

## Datasets

### H&M Dataset
Dataset từ H&M Fashion Recommendations, đã được xử lý sẵn và chia thành train/val/test:
- Users: Khách hàng H&M
- Items: Sản phẩm thời trang
- Interactions: Lịch sử mua hàng
- Files bao gồm:
  - `train.txt`, `val.txt`, `test.txt`: Dữ liệu đã split
  - `user_id_map_9m.csv`: Mapping user IDs (9 tháng)
  - `item_id_map_9m.csv`: Mapping item IDs (9 tháng)
  - `split_manifest.json`: Thông tin về cách chia dữ liệu

### Vibrent Dataset
Dataset tùy chỉnh với time-decay weights, tối ưu cho bài toán gợi ý dựa trên thời gian:
- Files bao gồm:
  - `train.txt`, `val.txt`, `test.txt`: Dữ liệu đã split
  - `train_time_weights_vibrent.csv`: Time-decay weights cho training
  - `user_id_map_vibrent.csv`: Mapping user IDs
  - `item_id_map_vibrent.csv`: Mapping item IDs
- Phù hợp để test tính năng time-decay của LightGCN

## Kết Quả Mẫu

Kết quả điển hình trên H&M dataset (Recall@20):

| Model | Recall@20 | NDCG@20 | Training Time | Complexity |
|-------|-----------|---------|---------------|------------|
| ALS | ~0.05-0.08 | ~0.03-0.05 | Nhanh | Thấp |
| MFBPR | ~0.06-0.09 | ~0.04-0.06 | Nhanh | Thấp |
| LightGCN | ~0.08-0.12 | ~0.05-0.08 | Trung bình | Trung bình |
| NGCF | ~0.09-0.13 | ~0.06-0.09 | Chậm | Cao |
| LightGCN + Time-Decay | ~0.10-0.14 | ~0.06-0.09 | Trung bình | Trung bình |

*Lưu ý: 
- Kết quả phụ thuộc vào hyperparameters và cách chia dữ liệu
- MFBPR có thể đạt kết quả tốt hơn ALS nhờ được tối ưu với BPR loss và gradient descent
- Graph-based models (LightGCN, NGCF) thường cho kết quả tốt hơn do tận dụng cấu trúc đồ thị*

## Kiến Trúc Kỹ Thuật

### BPR Loss (Bayesian Personalized Ranking)

Tất cả các mô hình neural (LightGCN, NGCF, MFBPR) sử dụng BPR loss:

```python
loss = -log(σ(pos_score - neg_score)) + λ * ||Θ||²
```

Trong đó:
- `pos_score`: Điểm của item positive (user đã tương tác)
- `neg_score`: Điểm của item negative (sampling ngẫu nhiên)
- `λ`: Regularization weight
- `Θ`: Ego embeddings (embeddings gốc)

### Graph Normalization

Symmetric normalization (LightGCN/NGCF):

```
A_hat = D^(-1/2) * A * D^(-1/2)
```

Trong đó:
- `A`: Adjacency matrix (user-item bipartite graph)
- `D`: Degree matrix

## Tối Ưu Hóa

**Gradient Clipping:**
- Tất cả scripts sử dụng gradient clipping (max_norm=5.0)
- Giúp ổn định training với large batch size

**Batch Size:**
- MFBPR: 8192 (trung bình, cân bằng giữa tốc độ và ổn định)
- LightGCN: 16384 (lớn hơn để tận dụng negative sampling)
- NGCF: 4096 (nhỏ hơn do model phức tạp hơn)
- ALS: Xử lý toàn bộ ma trận cùng lúc

**Learning Rate:**
- MFBPR: 1e-3 (cao hơn do model đơn giản)
- LightGCN: 5e-4 (nhỏ hơn do cần ổn định với graph propagation)
- NGCF: 1e-3 (lớn hơn do nhiều parameters)

## Lưu Ý Kỹ Thuật

1. **Memory Management:**
   - Sparse tensors được sử dụng cho adjacency matrix
   - Propagate một lần sau đó cache kết quả khi eval
   - Batch users khi tính full ranking

2. **Reproducibility:**
   - Seed được set cho Python, NumPy, PyTorch
   - CUDNN deterministic mode được bật
   - Sử dụng cùng seed để reproducible results

3. **Device Handling:**
   - Auto fallback to CPU nếu CUDA không available
   - Checkpoints được lưu ở CPU để portable

## Mở Rộng

### Thêm Dataset Mới

1. Tạo thư mục trong `data/`:
```bash
mkdir data/my_dataset
```

2. Chuẩn bị files:
```
data/my_dataset/
├── train.txt                    # Dữ liệu training (bắt buộc)
├── val.txt                      # Dữ liệu validation (bắt buộc)
├── test.txt                     # Dữ liệu testing (bắt buộc)
├── user_id_map_*.csv            # Mapping user IDs (tùy chọn)
└── item_id_map_*.csv            # Mapping item IDs (tùy chọn)
```

3. (Tùy chọn) Tạo time-decay weights:

   Bạn cần tự tạo file `train_time_weights.csv` với format:
   ```csv
   u,v,weight
   user_id,item_id,weight_value
   ```
   
   Trong đó `weight` có thể tính bằng hàm time-decay như: `weight = exp(-α * days_since_interaction)`

4. Chạy training với unified entrypoint:
```bash
# Sử dụng unified entrypoint
python train.py --model lightgcn --data_dir data/my_dataset

# Hoặc gọi trực tiếp script
python trainer/train_lightGCN_v2.py --data_dir data/my_dataset
```

### Thêm Mô Hình Mới

1. Tạo file trong `src/models/`:
```python
# src/models/MyModel.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        # Implementation của mô hình
        # Xem MFBPR.py để tham khảo mô hình đơn giản
        # Xem LightGCN.py hoặc NGCF.py cho mô hình dựa trên graph
        
    def forward(self, users, pos_items, neg_items):
        # Logic forward pass
        # Trả về pos_scores, neg_scores để tính BPR loss
        
    def full_sort_scores(self, users, all_items):
        # Scoring cho evaluation
```

2. Tạo training script trong `trainer/`:
```python
# trainer/train_mymodel.py
from src.models.MyModel import MyModel
# Follow pattern từ train_mf_bpr.py (cho non-graph models)
# hoặc train_lightGCN_v2.py (cho graph-based models)
```

3. Tạo evaluation script trong `evaluate/`:
```python
# evaluate/evaluate_mymodel.py
from src.models.MyModel import MyModel
# Follow pattern từ evaluate_mfbpr.py hoặc evaluate_lightgcn.py
```

4. Cập nhật `src/trainer.py` và `src/evaluator.py` để thêm mô hình mới vào RUNNERS dict

## Tài Liệu Tham Khảo

**Papers:**
- LightGCN: [He et al., SIGIR 2020](https://arxiv.org/abs/2002.02126)
- NGCF: [Wang et al., SIGIR 2019](https://arxiv.org/abs/1905.08108)
- BPR: [Rendle et al., UAI 2009](https://arxiv.org/abs/1205.2618)

**Libraries:**
- PyTorch: https://pytorch.org/
- implicit (ALS): https://github.com/benfred/implicit

## Liên Hệ & Đóng Góp

Dự án này được phát triển cho mục đích học tập và nghiên cứu. Mọi đóng góp và góp ý đều được chào đón.

**Contributions:**
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## License

Dự án này được phát triển cho mục đích giáo dục.
