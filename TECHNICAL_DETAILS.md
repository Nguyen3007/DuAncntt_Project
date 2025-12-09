# Chi Tiết Kỹ Thuật - Technical Deep Dive

## 📐 Kiến Trúc Hệ Thống

### 1. Tổng Quan Luồng Dữ Liệu

```
┌─────────────────┐
│  Raw Data       │
│  (train.txt)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  TxtCFDataLoader        │
│  - Parse user-item data │
│  - Create mappings      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  GraphBuilder           │
│  - Build adjacency      │
│  - Normalize: Â = D^½AD^½│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model (LightGCN/NGCF)  │
│  - Propagate embeddings │
│  - Compute BPR loss     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Optimizer              │
│  - Adam optimizer       │
│  - Gradient clipping    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Evaluation             │
│  - Top-K ranking        │
│  - Metrics calculation  │
└─────────────────────────┘
```

## 🔬 Phân Tích Chi Tiết DataLoader

### TxtCFDataLoader (`src/data_utils/dataloader.py`)

**Nhiệm vụ chính**:
1. Load 3 file: train.txt, val.txt, test.txt
2. Tự động infer num_users và num_items
3. Cung cấp API thuận tiện cho model

**Cấu trúc dữ liệu nội bộ**:
```python
self.train = {
    0: [20556, 5085, 132, ...],  # user 0 đã mua các items này
    1: [1, 2, 3, 84, ...],
    ...
}

self.val = {
    0: [41197],  # user 0 có 1 item validation
    1: [42332],
    ...
}

self.test = {
    0: [1233],   # user 0 có 1 item test
    1: [43218],
    ...
}
```

**API Methods**:
- `get_train_pos()`: Trả về dict user → items cho training
- `get_val_truth()`: Trả về dict user → single_item cho validation
- `get_test_truth()`: Trả về dict user → single_item cho test

**Lưu ý quan trọng**:
- Validation và test chỉ có 1 item/user (leave-last-1 strategy)
- Training có thể có nhiều items/user
- User IDs và Item IDs bắt đầu từ 0

## 🌐 Graph Construction - Chi Tiết

### Binary GraphBuilder

**Input**: Dictionary `train_user_items = {user: [items]}`

**Quá trình xây dựng đồ thị**:

```python
# Bước 1: Tạo bipartite graph (đồ thị hai phía)
# Nodes: [0...U-1] = users, [U...U+I-1] = items

for user in users:
    for item in user's items:
        add_edge(user, item + num_users)  # user → item
        add_edge(item + num_users, user)  # item → user (undirected)

# Bước 2: Tính degree của mỗi node
degree[i] = số edges kề với node i

# Bước 3: Symmetric normalization
for each edge (i, j) with weight w_ij:
    normalized_weight = w_ij / sqrt(degree[i] * degree[j])

# Bước 4: (Optional) Add self-loop cho NGCF
if add_self_loop:
    for each node i:
        add_edge(i, i, weight=1.0)
```

**Ma trận Adjacency chuẩn hóa**:

Công thức: Â = D^(-1/2) × A × D^(-1/2)

Ví dụ với đồ thị nhỏ:
```
Users: [0, 1]
Items: [0, 1, 2]  → Nodes: [2, 3, 4] (offset by num_users=2)

Interactions:
- User 0 bought items [0, 1]
- User 1 bought items [1, 2]

Adjacency matrix A (binary):
     0  1  2  3  4
  ┌─────────────────┐
0 │ 0  0  1  1  0  │  user 0
1 │ 0  0  0  1  1  │  user 1
2 │ 1  0  0  0  0  │  item 0
3 │ 1  1  0  0  0  │  item 1
4 │ 0  1  0  0  0  │  item 2
  └─────────────────┘

Degree matrix D:
D = diag([2, 2, 1, 2, 1])  # số edges của mỗi node

Normalized Â:
     0      1      2      3      4
  ┌──────────────────────────────────┐
0 │ 0      0     1/√2  1/2    0     │
1 │ 0      0      0    1/2   1/√2   │
2 │ 1/√2   0      0     0     0     │
3 │ 1/2   1/2     0     0     0     │
4 │ 0    1/√2     0     0     0     │
  └──────────────────────────────────┘
```

### Time-Decay GraphBuilder

**Điểm khác biệt**: Trọng số edges không phải là 1, mà phụ thuộc thời gian

**Input CSV** (`train_time_weights.csv`):
```csv
u,v,weight
0,20556,0.9523
0,5085,0.8891
0,132,0.7234
...
```

**Công thức time-decay**:
```python
# Giả sử có timestamp t_interaction
delta_t = t_current - t_interaction
weight = exp(-decay_rate * delta_t)

# Ví dụ:
# Item mua 1 ngày trước: weight ≈ 0.95
# Item mua 30 ngày trước: weight ≈ 0.74
# Item mua 90 ngày trước: weight ≈ 0.41
```

**Normalization với weights**:
```python
# Degree with weights
degree[i] = sum(weight_ij for all j connected to i)

# Symmetric normalization
normalized_weight_ij = weight_ij / sqrt(degree[i] * degree[j])
```

## 🧩 Kiến Trúc Model Chi Tiết

### LightGCN Architecture

**Layer-by-layer breakdown**:

```python
# Initialization
E_0 = Embedding_matrix  # Shape: [num_users + num_items, emb_dim]

# Layer 1
E_1 = Â @ E_0
# Ý nghĩa: Mỗi node nhận trung bình weighted embedding từ neighbors

# Layer 2
E_2 = Â @ E_1 = Â @ (Â @ E_0) = Â² @ E_0
# Ý nghĩa: Mỗi node nhận thông tin từ 2-hop neighbors

# Layer K
E_K = Â^K @ E_0
# Ý nghĩa: Thông tin từ K-hop neighbors

# Final Embedding
E_final = (E_0 + E_1 + E_2 + ... + E_K) / (K + 1)
```

**Tại sao không cần activation function?**
- Paper chứng minh rằng cho Collaborative Filtering, linear propagation đủ tốt
- Activation functions (ReLU, sigmoid) có thể làm mất information
- Đơn giản hơn = ít overfitting hơn

**Complexity Analysis**:
```
Time: O(K × |E| × d)  # K layers, |E| edges, d embedding dim
Space: O((U + I) × d)  # U users, I items
```

### NGCF Architecture

**Detailed Message Passing**:

```python
# Layer k message passing
for each layer k:
    # 1. Aggregate neighbors
    neighbor_sum = Â @ E_{k-1}  # Shape: [N, d]
    
    # 2. First-order propagation (Graph Convolution)
    msg_1 = LeakyReLU(neighbor_sum @ W_gc[k] + b_gc[k])
    # Transformation matrix W_gc học cách combine neighbor info
    
    # 3. Second-order propagation (Bi-Interaction)
    # Element-wise product: capture pairwise feature interactions
    bi_interaction = E_{k-1} ⊙ neighbor_sum
    msg_2 = LeakyReLU(bi_interaction @ W_bi[k] + b_bi[k])
    
    # 4. Combine messages
    E_k = msg_1 + msg_2
    
    # 5. Dropout (regularization)
    E_k = Dropout(E_k, p=mess_dropout)

# Final: Concatenate all layers
E_final = [E_0 || E_1 || E_2 || ... || E_K]
# Shape: [N, (K+1) × d]
```

**Bi-Interaction Term - Tại sao quan trọng?**

Ví dụ cụ thể:
```
User u có embedding: [0.5, 0.3, 0.8]
Item i có embedding:  [0.2, 0.7, 0.4]

# Không có bi-interaction:
Combined = W × (u + i) = linear combination

# Có bi-interaction:
Bi = u ⊙ i = [0.1, 0.21, 0.32]  # element-wise product
Combined = W1 × (u + i) + W2 × Bi
# Capture được tương tác giữa các features
```

**Complexity Analysis**:
```
Time: O(K × |E| × d + K × d²)  # Matrix multiplications
Space: O((U + I) × (K+1) × d)  # Lưu tất cả layers
```

## 🎯 BPR Loss - Mathematical Derivation

### Bayesian Personalized Ranking

**Intuition**: Minimize pairwise ranking loss

**Formulation**:

```
Given:
- User u
- Positive item i (user interacted)
- Negative item j (user NOT interacted)

Goal: Score(u, i) > Score(u, j)

Likelihood:
P(i >_u j | Θ) = σ(x_uij)

where:
x_uij = score(u, i) - score(u, j)
σ = sigmoid function

Maximum Likelihood:
max_Θ ∏_{u,i,j} σ(x_uij)

Log-likelihood:
max_Θ ∑_{u,i,j} log(σ(x_uij))

Minimize negative log-likelihood:
min_Θ -∑_{u,i,j} log(σ(x_uij))

With L2 regularization:
Loss = -∑_{u,i,j} log(σ(x_uij)) + λ||Θ||²
```

### Code Implementation

```python
def bpr_loss(user_emb, pos_emb, neg_emb, reg_weight):
    # Shape: [batch_size, emb_dim]
    
    # Scores
    pos_scores = (user_emb * pos_emb).sum(dim=1)  # [B]
    neg_scores = (user_emb * neg_emb).sum(dim=1)  # [B]
    
    # BPR: -mean(log(sigmoid(pos - neg)))
    diff = pos_scores - neg_scores
    bpr = -F.logsigmoid(diff).mean()
    
    # L2 regularization on ego embeddings
    # (only on initial embeddings, not propagated ones)
    reg = (user_emb.norm(2).pow(2) + 
           pos_emb.norm(2).pow(2) + 
           neg_emb.norm(2).pow(2)) / batch_size
    
    loss = bpr + reg_weight * reg
    return loss
```

### Negative Sampling Strategy

**Uniform Sampling**:
```python
def sample_negative(user, num_items, user_positive_set):
    while True:
        neg_item = random.randint(0, num_items - 1)
        if neg_item not in user_positive_set:
            return neg_item
```

**Tại sao uniform sampling?**
- Đơn giản, hiệu quả
- Paper chứng minh tốt hơn popularity-based sampling
- Tránh model bias về popular items

## 📊 Evaluation Metrics - Detailed Calculation

### Recall@K

```python
def recall_at_k(recommended_items, ground_truth_items, K):
    """
    recommended_items: Top-K items
    ground_truth_items: Items user actually interacted with
    """
    hits = len(set(recommended_items[:K]) & set(ground_truth_items))
    recall = hits / len(ground_truth_items)
    return recall

# Example:
recommended = [5, 2, 8, 1, 9]  # Top-5
ground_truth = [2, 7, 9]        # 3 relevant items

hits = {2, 9}  # 2 items in common
recall@5 = 2 / 3 = 0.667
```

### NDCG@K (Normalized Discounted Cumulative Gain)

**Intuition**: Items ranked higher should get more credit

```python
import numpy as np

def ndcg_at_k(recommended, ground_truth, K):
    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, item in enumerate(recommended[:K]):
        if item in ground_truth:
            rank = i + 1  # 1-indexed
            dcg += 1.0 / np.log2(rank + 1)
    
    # IDCG: Ideal DCG (best possible ranking)
    idcg = 0.0
    for i in range(min(len(ground_truth), K)):
        rank = i + 1
        idcg += 1.0 / np.log2(rank + 1)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

# Example:
recommended = [5, 2, 8, 1, 9]  # Positions: [1, 2, 3, 4, 5]
ground_truth = [2, 9]

# Item 2 at position 2: 1/log2(3) = 0.631
# Item 9 at position 5: 1/log2(6) = 0.387
DCG = 0.631 + 0.387 = 1.018

# Ideal: both items at positions 1 and 2
# Position 1: 1/log2(2) = 1.0
# Position 2: 1/log2(3) = 0.631
IDCG = 1.0 + 0.631 = 1.631

NDCG@5 = 1.018 / 1.631 = 0.624
```

### MAP@K (Mean Average Precision)

```python
def average_precision_at_k(recommended, ground_truth, K):
    if len(ground_truth) == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, item in enumerate(recommended[:K]):
        if item in ground_truth:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    ap = sum_precisions / min(len(ground_truth), K)
    return ap

# Example:
recommended = [5, 2, 8, 9, 1]
ground_truth = [2, 9]

# Item 2 at pos 2: precision = 1/2 = 0.5
# Item 9 at pos 4: precision = 2/4 = 0.5
AP@5 = (0.5 + 0.5) / 2 = 0.5
```

## ⚡ Optimization Techniques

### 1. Sparse Matrix Operations

```python
# PyTorch sparse matrix multiplication
adj_sparse = torch.sparse_coo_tensor(indices, values, size)

# Efficient: O(nnz × d) where nnz = number of non-zeros
result = torch.sparse.mm(adj_sparse, embeddings)

# vs Dense (very slow): O(N² × d)
# result = torch.mm(adj_dense, embeddings)
```

### 2. Batch Propagation

```python
# BAD: Propagate for each batch (slow)
for batch in batches:
    user_emb, item_emb = model.propagate(adj, batch)
    loss = compute_loss(user_emb, item_emb)
    
# GOOD: Propagate once for all nodes
all_emb = model.propagate(adj)  # Once per epoch
for batch in batches:
    user_emb = all_emb[batch.users]
    item_emb = all_emb[batch.items]
    loss = compute_loss(user_emb, item_emb)
```

### 3. Gradient Clipping

```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Why needed?
# - Deep propagation can cause gradient explosion
# - Especially in early epochs with random initialization
```

### 4. Mixed Precision Training (Future)

```python
# For faster training on modern GPUs
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in batches:
    with autocast():
        loss = model.forward(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 🔍 Debugging Tips

### Kiểm tra Graph Construction

```python
# Verify adjacency matrix
adj = gb.build_normalized_adj()

print(f"Shape: {adj.shape}")  # Should be [N, N]
print(f"Non-zeros: {adj._nnz()}")  # Number of edges
print(f"Is coalesced: {adj.is_coalesced()}")  # Should be True

# Check symmetry (for undirected graph)
adj_dense = adj.to_dense()
is_symmetric = torch.allclose(adj_dense, adj_dense.t())
print(f"Symmetric: {is_symmetric}")  # Should be True

# Check normalization (row sums should be consistent)
row_sums = torch.sparse.sum(adj, dim=1).to_dense()
print(f"Row sum stats: min={row_sums.min()}, max={row_sums.max()}")
```

### Monitor Training

```python
# Track metrics every epoch
metrics_history = {
    'train_loss': [],
    'val_recall': [],
    'val_ndcg': []
}

for epoch in epochs:
    train_loss = train_epoch()
    val_metrics = evaluate(split='val')
    
    metrics_history['train_loss'].append(train_loss)
    metrics_history['val_recall'].append(val_metrics['recall'])
    metrics_history['val_ndcg'].append(val_metrics['ndcg'])
    
    # Plot or save
    plot_metrics(metrics_history)
```

### Check Embedding Quality

```python
# After training
user_embs, item_embs = model.get_user_item_embeddings(adj)

# Check embedding norm
user_norms = user_embs.norm(dim=1)
print(f"User embedding norms: mean={user_norms.mean():.4f}, "
      f"std={user_norms.std():.4f}")

# Check for NaN/Inf
assert not torch.isnan(user_embs).any()
assert not torch.isinf(user_embs).any()

# Check similarity distribution
similarities = torch.matmul(user_embs, user_embs.t())
print(f"User-user similarity: min={similarities.min():.4f}, "
      f"max={similarities.max():.4f}, "
      f"mean={similarities.mean():.4f}")
```

## 🚀 Performance Tuning

### Memory Optimization

```python
# 1. Use smaller batch size
--batch_size 4096  # instead of 16384

# 2. Reduce embedding dimension
--emb_dim 32  # instead of 64

# 3. Fewer layers
--n_layers 1  # instead of 3

# 4. Use CPU if GPU memory is limited
--device cpu
```

### Speed Optimization

```python
# 1. Increase batch size (if memory allows)
--batch_size 32768

# 2. Reduce steps_per_epoch
--steps_per_epoch 400  # instead of 800

# 3. Use DataLoader with multiple workers (future improvement)
DataLoader(dataset, num_workers=4, pin_memory=True)

# 4. Profile code
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True) as prof:
    model.forward(batch)

print(prof.key_averages().table())
```

## 📈 Experiment Tracking (Recommended)

### Using TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/lightgcn_experiment_1')

for epoch in epochs:
    # Log training loss
    writer.add_scalar('Loss/train', train_loss, epoch)
    
    # Log validation metrics
    writer.add_scalar('Recall@20/val', val_recall, epoch)
    writer.add_scalar('NDCG@20/val', val_ndcg, epoch)
    
    # Log learning rate
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

writer.close()

# View: tensorboard --logdir=runs
```

### Using Weights & Biases (W&B)

```python
import wandb

wandb.init(project="recommendation-system", 
           config={
               "emb_dim": 64,
               "n_layers": 2,
               "lr": 5e-4,
           })

for epoch in epochs:
    wandb.log({
        "train_loss": train_loss,
        "val_recall": val_recall,
        "val_ndcg": val_ndcg,
        "epoch": epoch
    })
```

## 🎓 Advanced Topics

### 1. Cold Start Problem

**Vấn đề**: Users/items mới không có interaction history

**Giải pháp**:
```python
# 1. Use content features (if available)
user_content_emb = encode_user_profile(user_features)
final_emb = graph_emb + content_emb

# 2. Use popularity baseline for new items
if is_new_item:
    return top_popular_items()

# 3. Hybrid approach
score = α × graph_score + (1-α) × content_score
```

### 2. Temporal Dynamics

**Time-decay implementation**:
```python
# Option 1: Exponential decay
weight = exp(-λ × Δt)

# Option 2: Linear decay
weight = max(0, 1 - λ × Δt)

# Option 3: Logarithmic decay
weight = 1 / (1 + log(1 + Δt))
```

### 3. Multi-behavior Interactions

**Ví dụ**: view, cart, purchase
```python
# Build separate graphs
A_view = build_graph(view_interactions)
A_cart = build_graph(cart_interactions)
A_purchase = build_graph(purchase_interactions)

# Weighted combination
A_final = w1 × A_view + w2 × A_cart + w3 × A_purchase

# Or: Separate propagation
E_view = propagate(A_view)
E_cart = propagate(A_cart)
E_purchase = propagate(A_purchase)
E_final = concat([E_view, E_cart, E_purchase])
```

---

**Document này cung cấp hiểu biết sâu về implementation details. Đọc kèm README.md để có overview hoàn chỉnh!**
