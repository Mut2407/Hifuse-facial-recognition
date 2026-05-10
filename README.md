# 🎭 Nhận Diện Cảm Xúc Khuôn Mặt với HiFuse (Facial Expression Recognition)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)

Dự án này ứng dụng Deep Learning để giải quyết bài toán Nhận diện cảm xúc khuôn mặt (Facial Expression Recognition - FER). Hệ thống sử dụng mô hình **HiFuse (HiFuse_Small)** kết hợp với công nghệ nhận diện khuôn mặt **MTCNN** để tự động bóc tách và phân loại 7 trạng thái cảm xúc khác nhau của con người.

---

## 🌟 1. Giới thiệu ứng dụng

Nhận diện cảm xúc đóng vai trò quan trọng trong giao tiếp người - máy, phân tích tâm lý, và các hệ thống gợi ý cá nhân hóa. Dự án này được thiết kế để:
- Nhận diện khuôn mặt từ ảnh gốc có nhiều nhiễu.
- Tự động cắt (crop) và căn chỉnh vùng khuôn mặt.
- Phân loại biểu cảm khuôn mặt thành 7 nhãn cảm xúc cơ bản (Ví dụ: Tức giận, Ghê tởm, Sợ hãi, Vui vẻ, Buồn bã, Bất ngờ, Bình thường).

Tập dữ liệu mặc định được sử dụng và tối ưu trong dự án là **FER2013**.

---

## 🏗️ 2. Kiến trúc hệ thống

Hệ thống được thiết kế theo luồng quy trình (pipeline) khép kín từ khâu xử lý dữ liệu đến suy luận:

1. **Face Detection & Cropping (`pre_crop_faces.py`):** Sử dụng mạng **MTCNN** (`facenet_pytorch`) để quét ảnh, định vị chính xác khuôn mặt và cắt (crop) ra khung hình chuẩn (224x224), giúp loại bỏ các bối cảnh thừa (background noise).
2. **Data Pipeline (`my_dataset.py`):** Xây dựng lớp `MyDataSet` kế thừa từ `torch.utils.data.Dataset` để đọc dữ liệu ảnh và nhãn từ các thư mục động.
3. **Core Model (`main_model.py`):** Sử dụng cấu trúc mạng **HiFuse_Small**. Mô hình này lấy đầu vào là ảnh khuôn mặt đã căn chỉnh (224x224) và trích xuất các đặc trưng phân tầng.
4. **Training Strategy (`utils.py` & `train.py`):** - Sử dụng **Focal Loss** thay vì Cross-Entropy thông thường nhằm giải quyết triệt để tình trạng mất cân bằng lớp (Imbalanced data) đặc trưng của tập dữ liệu cảm xúc.
   - Kết hợp **Weighted Random Sampler** để lấy mẫu huấn luyện công bằng hơn.
   - Hỗ trợ huấn luyện với độ chính xác hỗn hợp tự động (**AMP - Automatic Mixed Precision**) để tăng tốc độ trên GPU.

---

## 💻 3. Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Deep Learning Framework:** PyTorch & Torchvision (Hỗ trợ huấn luyện phân tán và GPU CUDA).
- **Thị giác máy tính (Computer Vision):** - `facenet_pytorch` (Cung cấp MTCNN để nhận diện khuôn mặt).
  - `Pillow` (PIL) để xử lý ảnh đầu vào.
- **Trực quan hóa:** `matplotlib` để vẽ ảnh đầu ra kèm nhãn dự đoán và xác suất.
- **Tiện ích:** `tqdm` để hiển thị thanh tiến trình.

---

## ⚙️ 4. Nghiệp vụ tính năng

* **Tiền xử lý hàng loạt (Batch Preprocessing):** File `pre_crop_faces.py` cho phép quét toàn bộ tập dữ liệu gốc (train/val/test), tự động phát hiện khuôn mặt và lưu sang thư mục dữ liệu sạch (`fer2013_aligned`), giúp tăng độ chính xác khi huấn luyện.
* **Huấn luyện linh hoạt (Robust Training):** Hỗ trợ Data Augmentation (RandomResizedCrop, RandomHorizontalFlip, ColorJitter) trong quá trình huấn luyện nhằm chống Overfitting (`train.py`).
* **Đánh giá Đơn/Đa mục tiêu:**
  - `test.py`: Suy luận trên một bức ảnh duy nhất, hỗ trợ ảnh có cảnh vật xung quanh (hệ thống tự tìm và cắt khuôn mặt trước khi đoán).
  - `multi_test.py`: Đánh giá hàng loạt trên một thư mục ảnh (VD: `test10/surprise`), tính toán hàm Loss, xác suất (Accuracy) và tự động vẽ (plot) kết quả lưu vào thư mục `results_test`.

---

## 🚀 5. Hướng dẫn cài đặt và khởi chạy

### Bước 1: Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.8+. Cài đặt các thư viện phụ thuộc bằng lệnh sau:
```bash
pip install torch torchvision torchaudio facenet-pytorch matplotlib tqdm pillow
```

### Bước 2: Chuẩn bị dữ liệu
1. Tải tập dữ liệu FER2013 (hoặc tập dữ liệu của bạn) và đặt vào thư mục ./data/fer2013. Cấu trúc thư mục yêu cầu:
data/fer2013/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
├── val/
└── test/

2. Chạy lệnh tiền xử lý để cắt khuôn mặt:
```bash
python pre_crop_faces.py
```

### Bước 3: Huấn luyện mô hình
```bash
python train.py --train-data-path ./data/fer2013_aligned/train --val-data-path ./data/fer2013_aligned/val --num-classes 7 --epochs 80
```

### Bước 4: Kiểm thử và Suy luận
- Kiểm thử trên 1 ảnh đơn lẻ
```bash
python test.py
```

- Kiểm thử trên 1 thư mục ảnh
```bash
python multi_test.py
```
