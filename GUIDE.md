# 📖 Hướng Dẫn Đọc Tài Liệu - Documentation Guide

## Chào mừng bạn đến với DuAncntt_Project!

Repository này triển khai hai mô hình Graph Neural Network (LightGCN & NGCF) cho hệ thống gợi ý thời trang H&M. Tài liệu được tổ chức theo nhiều cấp độ để phục vụ cho người dùng khác nhau.

## 🗺️ Lộ Trình Học Tập

### 🔰 Người Mới Bắt Đầu

**Nếu bạn chưa biết gì về dự án này:**

1. **Bắt đầu với README.md** (15-20 phút)
   - Hiểu tổng quan về dự án
   - Biết được mục đích và ứng dụng
   - Nắm được cấu trúc cơ bản

2. **Đọc EXAMPLES.md - Kịch Bản 1** (10 phút)
   - Chạy thử training đơn giản nhất
   - Xem kết quả thực tế
   - Làm quen với command line

3. **Quay lại README.md - Phần "Giải Thích Các Mô Hình"** (20 phút)
   - Hiểu cơ bản về LightGCN
   - Hiểu cơ bản về NGCF
   - Biết khác biệt giữa hai mô hình

**Tổng thời gian: ~1 giờ**

### 🎓 Người Dùng Trung Cấp

**Nếu bạn đã chạy được code và muốn hiểu sâu hơn:**

1. **README.md - Đọc toàn bộ** (30 phút)
   - Hiểu đầy đủ các tính năng
   - Nắm được các metrics
   - Biết cách tune hyperparameters

2. **EXAMPLES.md - Tất cả các kịch bản** (1 giờ)
   - Thử nghiệm với time-decay
   - Làm inference
   - Debug và troubleshoot

3. **TECHNICAL_DETAILS.md - Phần "Kiến Trúc Hệ Thống"** (45 phút)
   - Hiểu luồng dữ liệu
   - Biết cách graph được xây dựng
   - Hiểu BPR loss

**Tổng thời gian: ~2-3 giờ**

### 🚀 Người Dùng Nâng Cao

**Nếu bạn muốn nghiên cứu sâu hoặc modify code:**

1. **TECHNICAL_DETAILS.md - Đọc toàn bộ** (2-3 giờ)
   - Hiểu chi tiết implementation
   - Biết mathematical formulation
   - Học optimization techniques

2. **Đọc source code kèm comments** (3-4 giờ)
   - `src/models/LightGCN.py`
   - `src/models/NGCF.py`
   - `src/data_utils/graph_builder.py`

3. **Thử nghiệm advanced topics** (tùy thời gian)
   - Multi-behavior interactions
   - Cold start solutions
   - Custom modifications

**Tổng thời gian: 1-2 ngày**

## 📚 Tổ Chức Tài Liệu

### 1. README.md - Tài Liệu Chính ⭐
**Đối tượng**: Tất cả mọi người
**Nội dung**:
- ✅ Tổng quan dự án
- ✅ Cấu trúc thư mục
- ✅ Giải thích mô hình (high-level)
- ✅ Hướng dẫn sử dụng cơ bản
- ✅ Metrics và đánh giá
- ✅ Hyperparameters
- ✅ Troubleshooting

**Khi nào đọc**: Luôn luôn đọc đầu tiên!

### 2. EXAMPLES.md - Hướng Dẫn Thực Hành 🎯
**Đối tượng**: Người muốn chạy code thực tế
**Nội dung**:
- ✅ 7 kịch bản thực hành cụ thể
- ✅ Code examples đầy đủ
- ✅ Expected outputs
- ✅ Scripts để copy-paste
- ✅ Debugging tips

**Khi nào đọc**: 
- Sau khi đọc README
- Khi muốn chạy training/inference
- Khi gặp lỗi cần debug

### 3. TECHNICAL_DETAILS.md - Chi Tiết Kỹ Thuật 🔬
**Đối tượng**: Researchers, advanced users
**Nội dung**:
- ✅ Mathematical formulation
- ✅ Implementation details
- ✅ Complexity analysis
- ✅ Optimization techniques
- ✅ Advanced topics

**Khi nào đọc**:
- Khi muốn hiểu sâu về algorithm
- Khi cần modify code
- Khi viết paper/thesis
- Khi optimize performance

### 4. GUIDE.md (File này) - Lộ Trình 🗺️
**Đối tượng**: Tất cả mọi người
**Nội dung**:
- ✅ Hướng dẫn đọc tài liệu
- ✅ Lộ trình học tập
- ✅ Quick reference
- ✅ FAQs

## 🔍 Quick Reference

### Tìm Kiếm Theo Chủ Đề

**Q: Làm sao để train model?**
→ README.md - "Hướng Dẫn Sử Dụng" → EXAMPLES.md - Kịch bản 1

**Q: LightGCN và NGCF khác nhau như thế nào?**
→ README.md - "Giải Thích Các Mô Hình"

**Q: Làm sao để tune hyperparameters?**
→ README.md - "Hyperparameters Quan Trọng" → EXAMPLES.md - Kịch bản 3

**Q: Time-decay là gì và dùng như thế nào?**
→ README.md - "Xây Dựng Đồ Thị" → EXAMPLES.md - Kịch bản 2

**Q: Làm sao để inference và recommend cho users?**
→ EXAMPLES.md - Kịch bản 4 và 5

**Q: BPR loss hoạt động như thế nào?**
→ README.md - "BPR Loss" (cơ bản) → TECHNICAL_DETAILS.md - "BPR Loss" (chi tiết)

**Q: Graph được xây dựng ra sao?**
→ README.md - "Xây Dựng Đồ Thị" → TECHNICAL_DETAILS.md - "Graph Construction"

**Q: Code bị lỗi, debug như thế nào?**
→ README.md - "Troubleshooting" → EXAMPLES.md - Kịch bản 6

**Q: Metrics được tính như thế nào?**
→ README.md - "Metrics Đánh Giá" → TECHNICAL_DETAILS.md - "Evaluation Metrics"

**Q: Muốn optimize performance?**
→ TECHNICAL_DETAILS.md - "Optimization Techniques" và "Performance Tuning"

## 📋 Checklist Học Tập

### Mức Cơ Bản
- [ ] Đọc xong README.md
- [ ] Hiểu được LightGCN vs NGCF
- [ ] Chạy được training script
- [ ] Evaluate được model
- [ ] Hiểu các metrics cơ bản

### Mức Trung Cấp
- [ ] Thử được time-decay graph
- [ ] Làm được inference
- [ ] Tune được hyperparameters
- [ ] Debug được lỗi thường gặp
- [ ] Hiểu BPR loss

### Mức Nâng Cao
- [ ] Hiểu mathematical formulation
- [ ] Biết optimize performance
- [ ] Có thể modify code
- [ ] Hiểu complexity analysis
- [ ] Đọc được papers gốc

## 🎯 Use Cases

### Use Case 1: Tôi là sinh viên làm đồ án
**Mục tiêu**: Hiểu và chạy được code, viết được báo cáo

**Lộ trình**:
1. README.md (toàn bộ) - Hiểu tổng quan
2. EXAMPLES.md - Kịch bản 1, 2, 3 - Chạy experiments
3. TECHNICAL_DETAILS.md - "Kiến Trúc" và "BPR Loss" - Viết báo cáo
4. Papers gốc - Cite trong báo cáo

**Thời gian**: 1-2 tuần

### Use Case 2: Tôi là developer muốn deploy
**Mục tiêu**: Chạy được model, làm inference cho production

**Lộ trình**:
1. README.md - "Hướng Dẫn Sử Dụng"
2. EXAMPLES.md - Kịch bản 4, 5 - Inference code
3. TECHNICAL_DETAILS.md - "Optimization Techniques"
4. Viết API wrapper

**Thời gian**: 3-5 ngày

### Use Case 3: Tôi là researcher muốn extend
**Mục tiêu**: Hiểu sâu để modify và improve

**Lộ trình**:
1. README.md (toàn bộ)
2. TECHNICAL_DETAILS.md (toàn bộ)
3. Đọc source code chi tiết
4. Đọc papers gốc
5. Implement modifications

**Thời gian**: 2-4 tuần

### Use Case 4: Tôi chỉ cần chạy nhanh để demo
**Mục tiêu**: Chạy được asap

**Lộ trình**:
1. README.md - "Cài Đặt Dependencies"
2. EXAMPLES.md - Kịch bản 1 - Copy commands
3. Chạy!

**Thời gian**: 30 phút - 1 giờ

## 💡 Tips

### Khi Đọc Tài Liệu
1. **Đừng đọc tuần tự từ đầu đến cuối** - Nhảy đến phần bạn cần
2. **Thực hành ngay khi đọc** - Chạy code trong khi đọc
3. **Bookmark các phần quan trọng** - Bạn sẽ quay lại nhiều lần
4. **Làm notes** - Ghi lại những gì bạn học được

### Khi Chạy Code
1. **Start small** - Bắt đầu với config nhỏ nhất
2. **Test incrementally** - Chạy từng bước, đừng chạy toàn bộ ngay
3. **Save everything** - Logs, checkpoints, results
4. **Version control** - Git commit thường xuyên

### Khi Gặp Vấn Đề
1. **Check README troubleshooting** - Lỗi thường gặp
2. **Xem EXAMPLES debugging** - Kịch bản 6
3. **Google error messages** - Thường có người gặp tương tự
4. **Read source code** - Đôi khi câu trả lời nằm trong code

## 🎓 Resources Bổ Sung

### Papers (Nên Đọc)
1. **LightGCN** (SIGIR 2020)
   - https://arxiv.org/abs/2002.02126
   - Đọc để hiểu design principles

2. **NGCF** (SIGIR 2019)
   - https://arxiv.org/abs/1905.08108
   - Đọc để hiểu bi-interaction

3. **BPR** (UAI 2009)
   - Foundation cho implicit feedback

### Code References
- Original LightGCN: https://github.com/gusye1234/LightGCN-PyTorch
- Original NGCF: https://github.com/xiangwang1223/neural_graph_collaborative_filtering

### Online Resources
- PyTorch docs: https://pytorch.org/docs/
- Graph Neural Networks: https://distill.pub/2021/gnn-intro/
- RecSys tutorials: https://recsys.acm.org/tutorials/

## 📞 Hỗ Trợ

Nếu bạn:
- ❓ Có câu hỏi
- 🐛 Tìm thấy bug
- 💡 Có ý tưởng cải thiện
- 📝 Muốn contribute

→ Tạo issue trên GitHub hoặc contact repository owner

## 📝 Kết Luận

Tài liệu này được thiết kế để phục vụ nhiều đối tượng khác nhau:

- **README.md**: Cho mọi người
- **EXAMPLES.md**: Cho practitioners
- **TECHNICAL_DETAILS.md**: Cho researchers
- **GUIDE.md** (file này): Cho navigation

Chọn lộ trình phù hợp với mục tiêu của bạn và happy learning! 🚀

---

**Cập nhật lần cuối**: December 2025
**Maintainer**: Nguyen3007
