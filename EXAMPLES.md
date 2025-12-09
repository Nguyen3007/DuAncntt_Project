# Hướng Dẫn Thực Hành - Practical Examples

## 🎯 Mục Tiêu

File này cung cấp các ví dụ thực tế, từng bước để giúp bạn:
1. Chạy training từ đầu
2. Đánh giá model
3. Sử dụng model cho inference
4. Thử nghiệm với different configurations

## 📚 Kịch Bản 1: Training LightGCN Cơ Bản

### Bước 1: Chuẩn bị môi trường

```bash
# Clone repository (nếu chưa có)
git clone <repository-url>
cd DuAncntt_Project

# Cài đặt dependencies
pip install torch numpy pandas

# Kiểm tra CUDA (nếu có GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Bước 2: Kiểm tra dữ liệu

```bash
# Xem structure của data
ls -lh data/h_m/

# Đếm số users
wc -l data/h_m/train.txt
wc -l data/h_m/val.txt
wc -l data/h_m/test.txt

# Xem vài dòng đầu tiên
head -3 data/h_m/train.txt
head -3 data/h_m/val.txt
```

### Bước 3: Training với cấu hình đơn giản

```bash
# Training với GPU, cấu hình nhỏ để test nhanh
python train_lightGCN_v2.py \
    --data_dir data/h_m \
    --emb_dim 32 \
    --n_layers 2 \
    --lr 1e-3 \
    --batch_size 4096 \
    --epochs 5 \
    --steps_per_epoch 100 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --early_stop_patience 2 \
    --val_K 20 \
    --seed 42

# Kết quả mong đợi:
# - Mỗi epoch mất khoảng 2-5 phút (tùy GPU)
# - Loss giảm dần
# - Validation metrics được in ra sau mỗi epoch
```

**Output ví dụ**:
```
=== TxtCFDataLoader ===
Data dir          : data/h_m
#Users            : 556884
#Items            : 43847
#Train interactions: 9845023
========================

[Graph] Building BINARY normalized adjacency...
[Graph] Done.

Epoch 01 | Step    1/ 100 | Loss 0.6931 (bpr=0.6931, reg=0.0000)
Epoch 01 | Step   50/ 100 | Loss 0.5234 (bpr=0.5201, reg=0.0033)
Epoch 01 | Step  100/ 100 | Loss 0.4523 (bpr=0.4489, reg=0.0034)

[Epoch 1] Avg train loss: 0.4523
[Epoch 1] VAL@20 | Precision: 0.001234 | Recall: 0.024680 | HitRate: 0.024680 | NDCG: 0.012345 | MAP: 0.011234
⭐ New best model at epoch 1 with Recall@20 = 0.024680
```

### Bước 4: Đánh giá model trên test set

```bash
# Evaluate best model
python evaluate_lightgcn.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/lightgcn_hm_best.pt \
    --split test \
    --K 20 \
    --device cuda \
    --batch_size 1024 \
    --seed 42
```

**Output ví dụ**:
```
========== EVAL RESULT ==========
Split:      test
K:          20
Precision@20: 0.0012
Recall@20:    0.0245
HitRate@20:   0.0245
NDCG@20:      0.0123
MAP@20:       0.0112
=================================
```

## 📚 Kịch Bản 2: Training NGCF với Time-Decay

### Bước 1: Tạo time-decay weights (giả sử)

```python
# create_time_weights.py
import pandas as pd
import numpy as np

# Giả sử bạn có file với timestamps
# transactions.csv: user_id, item_id, timestamp

df = pd.read_csv('data/h_m/transactions.csv')

# Tính time decay
max_time = df['timestamp'].max()
df['days_ago'] = (max_time - df['timestamp']) / (24 * 3600)
df['weight'] = np.exp(-0.01 * df['days_ago'])  # decay_rate = 0.01

# Lưu weights
output = df[['user_id', 'item_id', 'weight']]
output.columns = ['u', 'v', 'weight']
output.to_csv('data/h_m/train_time_weights.csv', index=False)
```

### Bước 2: Training với time-decay graph

```bash
python train_ngcf.py \
    --data_dir data/h_m \
    --use_time_decay \
    --time_weight_csv data/h_m/train_time_weights.csv \
    --emb_dim 64 \
    --layer_sizes 64 64 \
    --lr 1e-3 \
    --batch_size 4096 \
    --epochs 20 \
    --steps_per_epoch 800 \
    --device cuda \
    --mess_dropout 0.1 \
    --early_stop_patience 5

# Checkpoint sẽ tự động được đổi tên thành:
# - ngcf_hm_best_td.pt
# - ngcf_hm_last_td.pt
```

### Bước 3: So sánh binary vs time-decay

```bash
# Train binary NGCF
python train_ngcf.py --data_dir data/h_m --epochs 20 --device cuda

# Train time-decay NGCF
python train_ngcf.py --data_dir data/h_m --use_time_decay --epochs 20 --device cuda

# Evaluate cả hai
python evaluate_ngcf.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/ngcf_hm_best.pt \
    --split test --K 20

python evaluate_ngcf.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/ngcf_hm_best_td.pt \
    --split test --K 20 \
    --use_time_decay
```

## 📚 Kịch Bản 3: Hyperparameter Tuning

### Grid Search Script

```bash
#!/bin/bash
# hyperparameter_search.sh

EMB_DIMS=(32 64 128)
LEARNING_RATES=(1e-4 5e-4 1e-3)
N_LAYERS=(1 2 3)

for emb_dim in "${EMB_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for n_layers in "${N_LAYERS[@]}"; do
            echo "Training: emb_dim=$emb_dim, lr=$lr, n_layers=$n_layers"
            
            python train_lightGCN_v2.py \
                --data_dir data/h_m \
                --emb_dim $emb_dim \
                --lr $lr \
                --n_layers $n_layers \
                --epochs 30 \
                --device cuda \
                --checkpoint_dir "checkpoints/exp_${emb_dim}_${lr}_${n_layers}" \
                > "logs/exp_${emb_dim}_${lr}_${n_layers}.log" 2>&1
        done
    done
done
```

### Best Practices cho Hyperparameter Tuning

1. **Start small**: Test với 1 epoch trước
2. **One at a time**: Thay đổi 1 parameter một lúc
3. **Track everything**: Log tất cả experiments
4. **Use validation**: Chọn best model theo validation recall

## 📚 Kịch Bản 4: Inference và Recommendation

### Viết Script Inference

```python
# inference.py
import torch
import numpy as np
from src.data_utils.dataloader import TxtCFDataLoader
from src.data_utils.graph_builder import GraphBuilder
from src.models.LightGCN import LightGCN

def load_model(checkpoint_path, device='cuda'):
    """Load trained model"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = LightGCN(
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        emb_dim=ckpt["emb_dim"],
        n_layers=ckpt["n_layers"],
        reg_weight=ckpt["reg_weight"],
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def get_recommendations(model, adj, user_id, K=20, exclude_items=None):
    """
    Get top-K recommendations for a user
    
    Args:
        model: Trained model
        adj: Adjacency matrix
        user_id: User ID
        K: Number of recommendations
        exclude_items: Set of items to exclude (e.g., already purchased)
    
    Returns:
        List of (item_id, score) tuples
    """
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Propagate to get embeddings
        all_embs = model.propagate(adj)
        num_users = model.num_users
        
        user_emb = all_embs[user_id:user_id+1]  # [1, d]
        item_embs = all_embs[num_users:]        # [num_items, d]
        
        # Compute scores
        scores = torch.matmul(user_emb, item_embs.t()).squeeze()  # [num_items]
        
        # Exclude items
        if exclude_items is not None:
            scores[list(exclude_items)] = -float('inf')
        
        # Get top-K
        topk_scores, topk_items = torch.topk(scores, K)
        
        recommendations = [
            (item_id.item(), score.item()) 
            for item_id, score in zip(topk_items, topk_scores)
        ]
        
    return recommendations

# Usage
if __name__ == "__main__":
    device = 'cuda'
    
    # Load data
    loader = TxtCFDataLoader('data/h_m', verbose=True)
    
    # Build graph
    gb = GraphBuilder(
        num_users=loader.num_users,
        num_items=loader.num_items,
        train_user_items=loader.get_train_pos()
    )
    adj = gb.build_normalized_adj(device=device)
    
    # Load model
    model, ckpt = load_model('checkpoints/lightgcn_hm_best.pt', device=device)
    
    # Get recommendations for user 0
    user_id = 0
    purchased_items = set(loader.get_train_pos().get(user_id, []))
    
    recommendations = get_recommendations(
        model, adj, user_id, K=20, exclude_items=purchased_items
    )
    
    print(f"\nTop-20 recommendations for user {user_id}:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i:2d}. Item {item_id:5d} (score: {score:.4f})")
```

**Chạy inference**:
```bash
python inference.py

# Output:
# Top-20 recommendations for user 0:
#  1. Item 12345 (score: 0.8234)
#  2. Item 23456 (score: 0.7891)
#  3. Item 34567 (score: 0.7456)
# ...
```

## 📚 Kịch Bản 5: Batch Recommendation cho nhiều Users

```python
# batch_inference.py
import torch
import pandas as pd
from tqdm import tqdm

def batch_recommendations(model, adj, user_ids, K=20, batch_size=1024):
    """
    Get recommendations for multiple users efficiently
    """
    device = next(model.parameters()).device
    num_users = model.num_users
    
    # Propagate once
    with torch.no_grad():
        all_embs = model.propagate(adj)
        user_embs = all_embs[:num_users]
        item_embs = all_embs[num_users:]
    
    all_recommendations = {}
    
    for start in tqdm(range(0, len(user_ids), batch_size)):
        end = min(start + batch_size, len(user_ids))
        batch_users = user_ids[start:end]
        
        batch_users_t = torch.tensor(batch_users, dtype=torch.long, device=device)
        
        with torch.no_grad():
            u_emb = user_embs[batch_users_t]  # [B, d]
            scores = torch.matmul(u_emb, item_embs.t())  # [B, num_items]
            topk_scores, topk_items = torch.topk(scores, K, dim=1)
        
        topk_items_np = topk_items.cpu().numpy()
        topk_scores_np = topk_scores.cpu().numpy()
        
        for i, user_id in enumerate(batch_users):
            all_recommendations[user_id] = [
                (int(item), float(score)) 
                for item, score in zip(topk_items_np[i], topk_scores_np[i])
            ]
    
    return all_recommendations

# Usage
if __name__ == "__main__":
    from src.data_utils.dataloader import TxtCFDataLoader
    from src.data_utils.graph_builder import GraphBuilder
    from src.models.LightGCN import LightGCN
    
    device = 'cuda'
    loader = TxtCFDataLoader('data/h_m')
    
    gb = GraphBuilder(
        num_users=loader.num_users,
        num_items=loader.num_items,
        train_user_items=loader.get_train_pos()
    )
    adj = gb.build_normalized_adj(device=device)
    
    ckpt = torch.load('checkpoints/lightgcn_hm_best.pt')
    model = LightGCN(
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        emb_dim=ckpt["emb_dim"],
        n_layers=ckpt["n_layers"],
        reg_weight=ckpt["reg_weight"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Recommend for first 10000 users
    user_ids = list(range(10000))
    recommendations = batch_recommendations(model, adj, user_ids, K=20)
    
    # Save to CSV
    rows = []
    for user_id, recs in recommendations.items():
        for rank, (item_id, score) in enumerate(recs, 1):
            rows.append({
                'user_id': user_id,
                'rank': rank,
                'item_id': item_id,
                'score': score
            })
    
    df = pd.DataFrame(rows)
    df.to_csv('recommendations.csv', index=False)
    print(f"Saved {len(rows)} recommendations to recommendations.csv")
```

## 📚 Kịch Bản 6: Debugging và Troubleshooting

### Check Model Sanity

```python
# check_model.py
import torch
from src.models.LightGCN import LightGCN

model = LightGCN(num_users=100, num_items=50, emb_dim=16, n_layers=2)

# 1. Check parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Expected: {(100+50) * 16:,}")

# 2. Check forward pass
import numpy as np
from src.data_utils.graph_builder import GraphBuilder

dummy_train = {i: [j] for i in range(10) for j in range(5)}
gb = GraphBuilder(num_users=100, num_items=50, train_user_items=dummy_train)
adj = gb.build_normalized_adj()

all_embs = model.propagate(adj)
print(f"Embedding shape: {all_embs.shape}")  # Should be [150, 16]

# 3. Check loss computation
users = torch.tensor([0, 1, 2], dtype=torch.long)
pos_items = torch.tensor([0, 1, 2], dtype=torch.long)
neg_items = torch.tensor([3, 4, 5], dtype=torch.long)

loss, log_dict = model.bpr_loss_slow(users, pos_items, neg_items, adj)
print(f"Loss: {loss.item():.4f}")
print(f"Log dict: {log_dict}")

# 4. Check gradients
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")
```

### Monitor Memory Usage

```python
# monitor_memory.py
import torch
import gc

def print_memory_stats():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Before training
print("Before training:")
print_memory_stats()

# During training
# ... training code ...

# Clear cache
gc.collect()
torch.cuda.empty_cache()

print("\nAfter clearing cache:")
print_memory_stats()
```

## 📚 Kịch Bản 7: Analysis và Visualization

### Analyze Embeddings

```python
# analyze_embeddings.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load model
ckpt = torch.load('checkpoints/lightgcn_hm_best.pt')
embeddings = ckpt['model_state_dict']['embedding.weight'].cpu().numpy()

num_users = ckpt['num_users']
user_embs = embeddings[:num_users]
item_embs = embeddings[num_users:]

# 1. Embedding statistics
print(f"User embedding stats:")
print(f"  Mean: {user_embs.mean():.4f}")
print(f"  Std: {user_embs.std():.4f}")
print(f"  Min: {user_embs.min():.4f}")
print(f"  Max: {user_embs.max():.4f}")

# 2. t-SNE visualization (sample 1000 items)
sample_size = 1000
sample_items = item_embs[np.random.choice(len(item_embs), sample_size, replace=False)]

tsne = TSNE(n_components=2, random_state=42)
item_2d = tsne.fit_transform(sample_items)

plt.figure(figsize=(10, 10))
plt.scatter(item_2d[:, 0], item_2d[:, 1], alpha=0.5, s=10)
plt.title('Item Embeddings t-SNE Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('item_embeddings_tsne.png', dpi=150, bbox_inches='tight')
print("Saved visualization to item_embeddings_tsne.png")

# 3. Similarity analysis
# Find most similar items to item 0
item_0 = item_embs[0:1]
similarities = np.dot(item_embs, item_0.T).squeeze()
top_similar = np.argsort(similarities)[::-1][:10]

print(f"\nTop-10 items similar to item 0:")
for i, item_id in enumerate(top_similar, 1):
    print(f"{i:2d}. Item {item_id:5d} (similarity: {similarities[item_id]:.4f})")
```

## 🎓 Tips và Best Practices

### 1. Start Simple
```bash
# Luôn bắt đầu với config nhỏ
--emb_dim 32 --n_layers 1 --epochs 3 --steps_per_epoch 50
```

### 2. Monitor Everything
```python
# Log mọi thứ
import logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
```

### 3. Save Checkpoints Frequently
```python
# Save every N epochs
if epoch % 5 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
```

### 4. Use Version Control
```bash
# Tag experiments
git tag -a v1.0-lightgcn-baseline -m "Baseline LightGCN experiment"
git tag -a v1.1-lightgcn-timedecay -m "LightGCN with time decay"
```

### 5. Compare Results Systematically
```python
# results_comparison.py
results = {
    'LightGCN-baseline': {
        'recall@20': 0.0245,
        'ndcg@20': 0.0123,
    },
    'LightGCN-timedecay': {
        'recall@20': 0.0267,
        'ndcg@20': 0.0145,
    },
    'NGCF-baseline': {
        'recall@20': 0.0238,
        'ndcg@20': 0.0118,
    }
}

import pandas as pd
df = pd.DataFrame(results).T
print(df)
df.to_csv('results_comparison.csv')
```

---

**Các ví dụ này giúp bạn bắt đầu nhanh chóng và hiệu quả với repository!**
