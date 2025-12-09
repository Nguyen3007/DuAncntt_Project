# 🚀 BẮT ĐẦU NHANH - Quick Start

## Dự Án Này Là Gì?

Hệ thống gợi ý sản phẩm thời trang H&M sử dụng **Graph Neural Networks** (LightGCN & NGCF)

- 📦 **Dataset**: 556,884 users, 43,847 items (H&M fashion)
- 🧠 **Models**: LightGCN (simple & fast) + NGCF (complex & powerful)
- 🎯 **Task**: Recommend items dựa trên lịch sử mua hàng

## ⚡ Chạy Ngay (5 Phút)

```bash
# 1. Cài đặt
pip install torch numpy pandas

# 2. Training LightGCN (test nhanh - 3 phút)
python train_lightGCN_v2.py \
    --data_dir data/h_m \
    --emb_dim 32 \
    --epochs 3 \
    --steps_per_epoch 50 \
    --device cuda

# 3. Evaluate
python evaluate_lightgcn.py \
    --data_dir data/h_m \
    --checkpoint checkpoints/lightgcn_hm_best.pt \
    --split test \
    --K 20

# Done! 🎉
```

## 📚 Đọc Tài Liệu Gì?

### 🔰 Tôi mới bắt đầu
→ Đọc **README.md** (15-20 phút)

### 🎯 Tôi muốn chạy code
→ Đọc **EXAMPLES.md** → Kịch bản 1, 2, 4

### 🔬 Tôi muốn hiểu sâu
→ Đọc **TECHNICAL_DETAILS.md**

### 🗺️ Tôi không biết bắt đầu từ đâu
→ Đọc **GUIDE.md** để có lộ trình rõ ràng

## 📖 Tài Liệu Có Gì?

| File | Nội Dung | Đối Tượng |
|------|----------|-----------|
| **README.md** | Tổng quan, hướng dẫn sử dụng, giải thích mô hình | Tất cả mọi người ⭐ |
| **EXAMPLES.md** | 7 kịch bản thực hành có code | Người dùng thực tế 🎯 |
| **TECHNICAL_DETAILS.md** | Chi tiết kỹ thuật, toán học, optimization | Researchers 🔬 |
| **GUIDE.md** | Lộ trình học, navigation guide | Người mới 🗺️ |
| **START.md** | File này - Quick start | Bắt đầu nhanh ⚡ |

## 🎯 Use Cases Phổ Biến

### 1️⃣ Làm Đồ Án / Thesis
```
README.md → EXAMPLES.md (train & eval) → TECHNICAL_DETAILS.md (viết báo cáo) → Papers
⏱️ Thời gian: 1-2 tuần
```

### 2️⃣ Deploy Production
```
README.md → EXAMPLES.md (inference) → Viết API wrapper
⏱️ Thời gian: 3-5 ngày
```

### 3️⃣ Research & Extend
```
Tất cả docs → Source code → Papers → Implement new ideas
⏱️ Thời gian: 2-4 tuần
```

### 4️⃣ Demo Nhanh
```
EXAMPLES.md Kịch bản 1 → Copy commands → Run!
⏱️ Thời gian: 30 phút
```

## 🔥 Tính Năng Nổi Bật

✅ **2 mô hình state-of-the-art**: LightGCN & NGCF  
✅ **Time-decay weighting**: Tăng trọng số cho interactions gần đây  
✅ **Early stopping**: Tự động dừng khi overfitting  
✅ **Full evaluation metrics**: Precision, Recall, NDCG, MAP, HitRate  
✅ **GPU support**: Training nhanh với CUDA  
✅ **Batch inference**: Recommend cho nhiều users cùng lúc  
✅ **Documented code**: Comments chi tiết trong code  

## 🆘 Gặp Vấn Đề?

| Vấn đề | Giải pháp |
|--------|-----------|
| 🐛 **CUDA out of memory** | `--batch_size 2048` hoặc `--device cpu` |
| ⏰ **Training quá chậm** | `--steps_per_epoch 200` hoặc giảm `--emb_dim` |
| ❓ **Không biết bắt đầu** | Đọc **GUIDE.md** |
| 🔧 **Lỗi khi chạy** | Xem **README.md** → Troubleshooting |
| 📊 **Kết quả không tốt** | Tune hyperparameters (xem **README.md**) |

## 🎓 Tài Liệu Tham Khảo Nhanh

### Commands Quan Trọng

```bash
# Train LightGCN (full)
python train_lightGCN_v2.py --data_dir data/h_m --epochs 30 --device cuda

# Train NGCF (full)
python train_ngcf.py --data_dir data/h_m --epochs 20 --device cuda

# Train với time-decay
python train_lightGCN_v2.py --use_time_decay --data_dir data/h_m

# Evaluate
python evaluate_lightgcn.py --data_dir data/h_m --checkpoint <path> --split test

# Custom config
python train_lightGCN_v2.py \
    --emb_dim 64 \
    --n_layers 2 \
    --lr 5e-4 \
    --batch_size 16384 \
    --device cuda
```

### Hyperparameters Hay Dùng

**LightGCN**:
- `--emb_dim`: 32, 64, 128
- `--n_layers`: 1, 2, 3
- `--lr`: 1e-4, 5e-4, 1e-3
- `--batch_size`: 4096, 8192, 16384

**NGCF**:
- `--emb_dim`: 64
- `--layer_sizes`: 64 64 (hoặc 64 64 64)
- `--mess_dropout`: 0.0, 0.1, 0.2
- `--lr`: 1e-3

## 💡 Quick Tips

1. ⚡ **Start small**: Test với config nhỏ trước (3-5 epochs)
2. 📊 **Monitor validation**: Dùng early stopping
3. 💾 **Save checkpoints**: Luôn lưu best model
4. 🔍 **Debug systematically**: Check README troubleshooting
5. 📈 **Track experiments**: Log tất cả configurations và results

## 🏆 Kết Quả Mong Đợi

Với cấu hình mặc định trên H&M dataset:

| Model | Recall@20 | NDCG@20 | Training Time |
|-------|-----------|---------|---------------|
| LightGCN | ~0.05-0.08 | ~0.03-0.05 | ~2-3 giờ (30 epochs) |
| NGCF | ~0.04-0.07 | ~0.025-0.045 | ~3-4 giờ (20 epochs) |

*GPU: NVIDIA GPU với 8GB+ VRAM*

## 📞 Hỗ Trợ

- 📧 **Issues**: Tạo issue trên GitHub
- 📚 **Documentation**: Đọc các file .md
- 🔍 **Source code**: Comments trong code
- 📖 **Papers**: Links trong README.md

## ✅ Checklist Bắt Đầu

- [ ] Cài đặt dependencies (`pip install -r requirements.txt`)
- [ ] Kiểm tra data (`ls data/h_m/`)
- [ ] Chạy training test (`--epochs 3 --steps_per_epoch 50`)
- [ ] Đọc README.md (ít nhất phần "Tổng Quan" và "Hướng Dẫn Sử Dụng")
- [ ] Chạy training full
- [ ] Evaluate model
- [ ] Đọc thêm tài liệu chi tiết (nếu cần)

## 🚀 Next Steps

Sau khi chạy được training cơ bản:

1. 📖 **Hiểu sâu hơn**: Đọc TECHNICAL_DETAILS.md
2. 🎯 **Thử nghiệm**: EXAMPLES.md có 7 kịch bản
3. 🔧 **Tùy chỉnh**: Modify hyperparameters
4. 📊 **So sánh**: LightGCN vs NGCF, binary vs time-decay
5. 🚀 **Deploy**: Viết inference API (xem EXAMPLES.md kịch bản 4, 5)

---

**Prepared by**: Documentation Team  
**Version**: 1.0  
**Last Updated**: December 2025

**🎉 Chúc bạn thành công với dự án! Happy coding! 🚀**
