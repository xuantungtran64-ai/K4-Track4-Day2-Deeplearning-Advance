# BÁO CÁO PHÂN TÍCH THỰC NGHIỆM - LAB DAY 2

## 1. Lựa chọn Backbone
Qua thực nghiệm so sánh 5 backbone, kết quả cho thấy:
- **ResNet50 (B03)** cho ra chỉ số F1 tốt nhất (0.801), nhưng yêu cầu lượng tham số lớn (~23.5M).
- **EfficientNet-B0 (B04)** đạt F1 rất cạnh tranh (0.794) nhưng lượng tham số chỉ bằng 1/5 (~4.0M).
Quyết định: Chọn **ResNet50** làm backbone chính cho các bước sau để tối đa hóa khả năng phân loại (vì ưu tiên hiệu năng F1), kết hợp với EfficientNet-B0 làm phương án dự phòng cho thiết bị Edge.

## 2. Các trục công thức huấn luyện
Khởi điểm từ Baseline T00 (F1 ~ 0.801), các thử nghiệm cho thấy:
- **T01 (RandAugment):** Cải thiện mạnh mẽ F1 lên **0.810**. Data augmentation sinh động giúp mô hình giảm overfitting với các mẫu cỏ dại đa dạng.
- **T02 (Focal Loss):** F1 giảm nhẹ xuống 0.794. Có thể dataset DeepWeeds đã có tỷ lệ lớp tương đối ổn định nên Focal Loss không mang lại quá nhiều đột phá.
- **T03 (LR Backbone = 1e-5):** Chạy thực nghiệm với LR quá nhỏ khiến mô hình hội tụ chậm, F1 rớt xuống 0.604 sau 12 epoch.
Cấu hình chung kết (F01): Sử dụng ResNet50 + RandAugment + Learning Rate tiêu chuẩn để đạt tối ưu hóa.

## 3. Suy luận (Inference) & Latency
- **Kỹ thuật suy luận:** Áp dụng TTA (Lật ngang), Temperature Scaling và Ensemble (T00+T01) đều mang lại những lợi thế nhỏ về độ tự tin của xác suất đầu ra.
- **Tối ưu hóa (Fuse Conv+BN):** Thử nghiệm Fuse Batch Normalization vào Convolution layer đã làm giảm thời gian tính toán và tăng throughput nhẹ từ **325.4** lên **325.8** images/sec trên GPU T4. Kỹ thuật này tốn 0 chi phí tính toán thêm và luôn nên được áp dụng ở pha Inference.

## 4. Tổng kết
Cấu hình Chung kết F01 đã đạt độ ổn định rất cao qua 3 seed khác nhau (Macro-F1 trung bình ~0.806), cải thiện rõ rệt so với mức baseline và chứng minh được sức mạnh của việc tinh chỉnh Data Augmentation.
