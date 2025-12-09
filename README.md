# Hệ Thống Gợi Ý Sản Phẩm - LightGCN & NGCF

## 📋 Tổng Quan

Repository này triển khai hai mô hình Graph Neural Network (GNN) tiên tiến cho bài toán **Collaborative Filtering** (Lọc cộng tác) trong hệ thống gợi ý:

1. **LightGCN** - Light Graph Convolutional Network
2. **NGCF** - Neural Graph Collaborative Filtering

Cả hai mô hình đều được áp dụng trên dataset **H&M Fashion** với khoảng **556,884 người dùng** và hàng chục nghìn sản phẩm thời trang.

## 🎯 Mục Đích Dự Án

Dự án này giải quyết bài toán **gợi ý sản phẩm thời trang** cho người dùng dựa trên lịch sử mua hàng của họ. Hệ thống sử dụng:
- **Implicit Feedback**: Chỉ cần biết người dùng đã mua sản phẩm nào (không cần rating)
- **Graph-based Learning**: Biểu diễn quan hệ user-item dưới dạng đồ thị để học embeddings tốt hơn
- **BPR Loss**: Bayesian Personalized Ranking để tối ưu hóa thứ tự sản phẩm

## 📁 Cấu Trúc Thư Mục

```
DuAncntt_Project/
├── src/                          # Mã nguồn chính
│   ├── models/                   # Các mô hình GNN
│   │   ├── LightGCN.py          # Mô hình LightGCN
│   │   └── NGCF.py              # Mô hình NGCF
│   └── data_utils/               # Công cụ xử lý dữ liệu
│       ├── dataloader.py         # Load dữ liệu từ file txt
│       ├── graph_builder.py      # Xây dựng đồ thị binary
│       └── graph_builder_time_decay.py  # Đồ thị với time-decay weights
├── data/                         # Thư mục dữ liệu
│   └── h_m/                      # Dataset H&M
│       ├── train.txt             # Dữ liệu training
│       ├── val.txt               # Dữ liệu validation
│       ├── test.txt              # Dữ liệu test
│       ├── split_manifest.json   # Thông tin về split
│       ├── user_id_map_9m.csv    # Mapping user IDs
│       └── item_id_map_9m.csv    # Mapping item IDs
├── train_lightGCN_v2.py         # Script training LightGCN
├── train_ngcf.py                 # Script training NGCF
├── evaluate_lightgcn.py          # Script đánh giá LightGCN
├── evaluate_ngcf.py              # Script đánh giá NGCF
└── requirements.txt              # Dependencies

```

## 🔍 Chi Tiết Về Dữ Liệu

### Format Dữ Liệu

Dữ liệu được tổ chức theo format **LightGCN-style**:

**train.txt**: Mỗi dòng chứa lịch sử mua hàng của 1 user
```
user_id item1 item2 item3 ... itemN
```
Ví dụ:
```
0 20556 5085 132 17949 5009 12202 1001 17453 ...
1 1 2 3 84 22500 16255 4020 26867 ...
```

**val.txt** và **test.txt**: Mỗi user có đúng 1 item (leave-last-1 strategy)
```
user_id item_id
```
Ví dụ:
```
0 41197
1 42332
2 33118
```

### Thống Kê Dataset H&M

- **Số lượng users**: 556,884
- **Số lượng items**: ~40,000+ sản phẩm thời trang
- **Split strategy**: Temporal split (theo thời gian)
  - Training: Tất cả items (unique) của mỗi user (trừ 2 items cuối)
  - Validation: Item cuối cùng thứ 2
  - Test: Item cuối cùng

## 🧠 Giải Thích Các Mô Hình

### 1. LightGCN (Light Graph Convolutional Network)

**Ý tưởng chính**: Đơn giản hóa GCN cho Collaborative Filtering bằng cách loại bỏ các thành phần không cần thiết.

**Kiến trúc**:
```
1. Embedding Layer: Khởi tạo vector cho mỗi user và item
2. Graph Propagation: 
   - Layer 0: E⁽⁰⁾ = Embedding gốc
   - Layer k: E⁽ᵏ⁾ = Â × E⁽ᵏ⁻¹⁾  (chỉ là matrix multiplication)
   - Â: Normalized adjacency matrix
3. Layer Combination:
   Final_Embedding = mean(E⁽⁰⁾, E⁽¹⁾, ..., E⁽ᴷ⁾)
```

**Đặc điểm**:
- ✅ Rất đơn giản, không có activation function hay transformation matrix
- ✅ Hiệu quả về mặt tính toán
- ✅ State-of-the-art performance trên nhiều datasets
- ⚙️ Không sử dụng self-loop (A, không phải A+I)

**Code chính** (`src/models/LightGCN.py`):
```python
def propagate(self, adj: torch.Tensor) -> torch.Tensor:
    x = self.embedding.weight  # E⁽⁰⁾
    embs = [x]
    
    for _ in range(self.n_layers):
        x = torch.sparse.mm(adj, x)  # E⁽ᵏ⁾ = Â × E⁽ᵏ⁻¹⁾
        embs.append(x)
    
    all_embeddings = torch.stack(embs, dim=0).mean(dim=0)
    return all_embeddings
```

### 2. NGCF (Neural Graph Collaborative Filtering)

**Ý tưởng chính**: Mô hình hóa high-order connectivity bằng cách học message passing phức tạp hơn.

**Kiến trúc**:
```
1. Embedding Layer: Vector ban đầu cho user/item
2. Message Passing (mỗi layer k):
   a) Graph Convolution:
      msg₁ = LeakyReLU((Â × E⁽ᵏ⁻¹⁾) × W₁ + b₁)
   
   b) Bi-Interaction:
      msg₂ = LeakyReLU((E⁽ᵏ⁻¹⁾ ⊙ Â×E⁽ᵏ⁻¹⁾) × W₂ + b₂)
   
   c) Combination:
      E⁽ᵏ⁾ = msg₁ + msg₂
      
3. Final Embedding: Concat tất cả layers
   Final = [E⁽⁰⁾ || E⁽¹⁾ || ... || E⁽ᴷ⁾]
```

**Đặc điểm**:
- ✅ Học được tương tác phức tạp hơn giữa users và items
- ✅ Bi-interaction term giúp capture feature interactions
- ⚙️ Sử dụng self-loop (A+I) trong adjacency matrix
- ⚙️ Message dropout để regularization

**Code chính** (`src/models/NGCF.py`):
```python
def _propagate_impl(self, adj: torch.Tensor) -> torch.Tensor:
    ego_embeddings = self.embedding.weight
    all_embeddings = [ego_embeddings]
    
    x = ego_embeddings
    for k in range(self.n_layers):
        # Graph convolution
        side_embeddings = torch.sparse.mm(adj, x)
        sum_embeddings = torch.matmul(side_embeddings, self.W_gc[k]) + self.b_gc[k]
        sum_embeddings = F.leaky_relu(sum_embeddings)
        
        # Bi-interaction
        bi = x * side_embeddings
        bi_embeddings = torch.matmul(bi, self.W_bi[k]) + self.b_bi[k]
        bi_embeddings = F.leaky_relu(bi_embeddings)
        
        # Combine
        x = sum_embeddings + bi_embeddings
        x = F.dropout(x, p=self.mess_dropout, training=self.training)
        
        all_embeddings.append(x)
    
    return torch.cat(all_embeddings, dim=1)
```

## 🎓 BPR Loss (Bayesian Personalized Ranking)

Cả hai mô hình đều sử dụng **BPR Loss** để training:

```python
# Cho mỗi user u:
#   - pos_item: item mà user đã tương tác
#   - neg_item: item mà user chưa tương tác (random sampling)

score_pos = user_emb · pos_item_emb
score_neg = user_emb · neg_item_emb

BPR_Loss = -mean(log(sigmoid(score_pos - score_neg)))
L2_Reg = ||embeddings||²

Total_Loss = BPR_Loss + λ × L2_Reg
```

**Intuition**: Model học sao cho điểm số của positive items luôn cao hơn negative items.

## ⚙️ Xây Dựng Đồ Thị (Graph Construction)

### 1. Binary Graph (GraphBuilder)

Đồ thị không trọng số: edge weight = 1 nếu user tương tác với item.

```python
gb = GraphBuilder(
    num_users=num_users,
    num_items=num_items,
    train_user_items=train_dict,
    add_self_loop=False  # LightGCN: False, NGCF: True
)
adj = gb.build_normalized_adj(device='cuda')
```

**Normalized Adjacency**: Â = D^(-1/2) × A × D^(-1/2)

### 2. Time-Decay Graph (TimeDecayGraphBuilder) 🔥

Đồ thị có trọng số dựa trên thời gian - **Tính năng nâng cao**!

**Ý tưởng**: Items mua gần đây có trọng số cao hơn (quan trọng hơn).

```python
gb = TimeDecayGraphBuilder(
    num_users=num_users,
    num_items=num_items,
    weight_csv='data/h_m/train_time_weights.csv',
    add_self_loop=False
)
adj = gb.build_normalized_adj(device='cuda')
```

**Công thức time-decay**: 
```
weight = exp(-α × Δt)
```
Trong đó:
- Δt: khoảng thời gian từ thời điểm tương tác đến hiện tại
- α: decay rate (thường là 0.001 - 0.1)

## 🚀 Hướng Dẫn Sử Dụng

### Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

Requirements:
- torch
- numpy
- pandas

### 1. Training LightGCN

**Binary Graph (đơn giản)**:
```bash
python train_lightGCN_v2.py \
    --data_dir data/h_m \
    --emb_dim 64 \
    --n_layers 2 \
    --lr 5e-4 \
    --batch_size 16384 \
    --epochs 30 \
    --steps_per_epoch 800 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --early_stop_patience 3
```

**Time-Decay Graph (nâng cao)**:
```bash
python train_lightGCN_v2.py \
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

### 2. Training NGCF

**Binary Graph**:
```bash
python train_ngcf.py \
    --data_dir data/h_m \
    --emb_dim 64 \
    --layer_sizes 64 64 \
    --lr 1e-3 \
    --batch_size 4096 \
    --epochs 20 \
    --steps_per_epoch 800 \
    --device cuda \
    --mess_dropout 0.1 \
    --early_stop_patience 5
```

**Time-Decay Graph**:
```bash
python train_ngcf.py \
    --data_dir data/h_m \
    --use_time_decay \
    --emb_dim 64 \
    --layer_sizes 64 64 \
    --lr 1e-3 \
    --batch_size 4096 \
    --epochs 20 \
    --device cuda
```

### 3. Evaluation

**Evaluate LightGCN**:
```bash
# Trên validation set
python evaluate_lightgcn.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/lightgcn_hm_best.pt \
    --split val \
    --K 20 \
    --device cuda

# Trên test set
python evaluate_lightgcn.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/lightgcn_hm_best.pt \
    --split test \
    --K 20 \
    --device cuda
```

**Evaluate NGCF**:
```bash
python evaluate_ngcf.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/ngcf_hm_best.pt \
    --split test \
    --K 20 \
    --device cuda
```

## 📊 Metrics Đánh Giá

Hệ thống sử dụng các metrics chuẩn cho Recommendation Systems:

1. **Precision@K**: Tỷ lệ items liên quan trong top-K
   ```
   Precision@K = (số items đúng trong top-K) / K
   ```

2. **Recall@K**: Tỷ lệ items liên quan được tìm thấy
   ```
   Recall@K = (số items đúng trong top-K) / (tổng số items đúng)
   ```

3. **HitRate@K**: Tỷ lệ users có ít nhất 1 item đúng trong top-K
   ```
   HitRate@K = (số users có hit) / (tổng số users)
   ```

4. **NDCG@K** (Normalized Discounted Cumulative Gain): Xem xét thứ tự ranking
   ```
   NDCG@K = DCG@K / IDCG@K
   ```

5. **MAP@K** (Mean Average Precision): Trung bình precision tại mọi vị trí có item đúng

## 🎯 Quy Trình Training

### Early Stopping Strategy

Training sử dụng **early stopping** dựa trên Recall@K trên validation set:

```python
# Pseudo-code
best_recall = 0
patience_counter = 0
max_patience = 3

for epoch in epochs:
    train_one_epoch()
    
    metrics_val = evaluate_on_validation()
    current_recall = metrics_val['recall']
    
    if current_recall > best_recall:
        best_recall = current_recall
        save_best_model()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= max_patience:
        print("Early stopping!")
        break
```

### Negative Sampling

Mỗi batch training:
```python
for each user in batch:
    - Chọn 1 positive item (đã tương tác)
    - Chọn 1 negative item (chưa tương tác, random)
    - Tính BPR loss
```

## 💡 Các Tính Năng Nâng Cao

### 1. Time-Decay Weighting 🕐

Tự động đổi tên checkpoint khi dùng time-decay:
- `lightgcn_hm_best.pt` → `lightgcn_hm_best_td.pt`
- `ngcf_hm_best.pt` → `ngcf_hm_best_td.pt`

### 2. Gradient Clipping

Tránh gradient explosion:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### 3. Checkpoint Management

Lưu 2 loại checkpoints:
- **Best model**: Model tốt nhất theo validation recall
- **Last model**: Model ở epoch cuối (có thể resume training)

## 📈 Kết Quả Mong Đợi

Với dataset H&M và cấu hình mặc định:

**LightGCN** (sau ~10-15 epochs):
- Recall@20: ~0.05 - 0.08
- NDCG@20: ~0.03 - 0.05

**NGCF** (sau ~15-20 epochs):
- Recall@20: ~0.04 - 0.07
- NDCG@20: ~0.025 - 0.045

*Lưu ý*: Kết quả có thể thay đổi tùy hyperparameters và random seed.

## 🔧 Hyperparameters Quan Trọng

### LightGCN
- `emb_dim`: Kích thước embedding vector (32, 64, 128)
- `n_layers`: Số layers GCN (1-4, thường là 2-3)
- `lr`: Learning rate (1e-4 đến 1e-3)
- `reg_weight`: L2 regularization (1e-5 đến 1e-3)
- `batch_size`: Batch size (4096 - 32768)

### NGCF
- `emb_dim`: Kích thước embedding ban đầu
- `layer_sizes`: Kích thước cho mỗi layer message passing
- `mess_dropout`: Dropout rate (0.0 - 0.3)
- `leaky_relu_slope`: Slope cho LeakyReLU (0.1 - 0.3)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Giảm batch_size
python train_lightGCN_v2.py --batch_size 8192 ...

# Hoặc dùng CPU
python train_lightGCN_v2.py --device cpu ...
```

### Training quá chậm
```bash
# Giảm steps_per_epoch
python train_lightGCN_v2.py --steps_per_epoch 400 ...

# Giảm số epochs (dùng early stopping)
python train_lightGCN_v2.py --epochs 20 --early_stop_patience 3 ...
```

## 📚 Tài Liệu Tham Khảo

### Papers

1. **LightGCN**: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)
   - Paper: https://arxiv.org/abs/2002.02126

2. **NGCF**: Wang et al. "Neural Graph Collaborative Filtering" (SIGIR 2019)
   - Paper: https://arxiv.org/abs/1905.08108

3. **BPR**: Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)

### Code References
- Original LightGCN: https://github.com/gusye1234/LightGCN-PyTorch
- Original NGCF: https://github.com/xiangwang1223/neural_graph_collaborative_filtering

## 🤝 Đóng Góp

Nếu bạn muốn đóng góp cho dự án:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

## 👨‍💻 Tác Giả

Repository được tạo bởi Nguyen3007

## 📞 Liên Hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên GitHub.

---

**Happy Coding! 🚀**
