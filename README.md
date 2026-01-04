# Movie Rating Prediction (Regression) — Stacking Ensemble (Ridge Meta-Model)

Dự án xây dựng mô hình **dự đoán điểm rating (1–5)** cho phim từ metadata (ví dụ: *budget, popularity, release_year, genre one-hot, …*).  
Trọng tâm của repo là **Stacking Ensemble**: kết hợp nhiều mô hình base (Linear Regression, KNN, Random Forest, GBDT) và **meta-model Ridge Regression** học cách “trộn” dự đoán để cải thiện RMSE.

---

## 1) Mục tiêu
- Bài toán: **Supervised Regression**  
- Input: đặc trưng phim (numeric + categorical đã encode)  
- Output: rating dự đoán (liên tục)  
- Metrics chính: **RMSE** (ưu tiên), kèm **MAE, R²**

---

## 2) Pipeline tổng quát
1. **Tiền xử lý dữ liệu**
   - Làm sạch cột, xử lý thiếu
   - Chuẩn hoá các cột số (vd: `budget`, `popularity`, `release_year`)
   - One-hot encoding cho cột phân loại (vd: `genre_*`)
2. **Chia dữ liệu 60/20/20**
   - 60%: train (dùng để tạo OOF cho stacking)
   - 20%: validation (tuning)
   - 20%: test (đánh giá cuối)
3. **Train các base models**
   - Linear Regression (baseline tuyến tính)
   - KNN Regression (dựa trên khoảng cách sau chuẩn hoá)
   - Random Forest Regression (phi tuyến mạnh, giảm overfit so với cây đơn)
   - Gradient Boosting Decision Trees (GBDT) (boosting theo hướng giảm lỗi)
4. **Stacking (Ridge meta-model)**
   - Tạo **OOF predictions** trên phần train (K-fold) cho từng base model
   - Dùng OOF làm feature đầu vào cho Ridge để học trọng số kết hợp
   - Có thể thêm **meta-features phi tuyến** (vd: polynomial / tương tác thủ công)

> **Note quan trọng:** Stacking “đúng” phải dùng **OOF** để tránh leakage (meta-model không được học từ dự đoán sinh ra bởi base model đã nhìn thấy chính mẫu đó trong train).

---

## 3) Mô hình sử dụng
### Base models
- **Linear Regression**: mô hình tuyến tính làm baseline, dễ diễn giải.
- **KNN Regression**: dự đoán bằng trung bình rating của *k* láng giềng gần nhất (cần chuẩn hoá để khoảng cách có ý nghĩa).
- **Random Forest Regression**: ensemble nhiều decision tree, mỗi cây học từ bootstrap + chọn ngẫu nhiên tập con feature ở mỗi node.
- **GBDT**: mô hình boosting, cộng dồn các cây nhỏ để giảm dần residual.

### Meta model (điểm nhấn của repo)
- **Ridge Regression (stacking meta-model)**: học cách kết hợp dự đoán của base models, ổn định hơn khi các base models có tương quan (regularization L2 giúp giảm overfit).

---

## 4) Kết quả (tóm tắt)
> Số liệu có thể thay đổi tuỳ lần chạy/tuning. Repo lưu các file dự đoán/OOF để tái lập kết quả.

**Random Forest (best theo Validation RMSE)**  
- Best params (val RMSE tốt nhất):  
  `n_estimators=95, max_depth=17, max_features≈0.2769, min_samples_split=8, min_samples_leaf=2, bootstrap=True, random_state=42`  
- Split: Train 49135 | Val 16378 | Test 16379  
- Metrics:
  - Train: RMSE=0.9242, R²=0.2352
  - Val: **RMSE=0.9560**, R²=0.1722
  - Test: RMSE=0.9743, R²=0.1641, MAE=0.7634

**Stacking Ridge (meta-model)**  
- Mục tiêu: giảm RMSE so với từng base model nhờ tận dụng “điểm mạnh” khác nhau của LR/KNN/Tree/Boosting.
- Repo có thể kèm **ablation** (bỏ bớt 1 base model) để thấy ảnh hưởng từng thành phần lên kết quả cuối.

---

## 5) Cấu trúc thư mục (gợi ý)
```
.
├─ data/
│  ├─ data_raw.csv
│  ├─ data_processed.csv
│  ├─ oof_lr_train.csv
│  ├─ oof_knn_train.csv
│  ├─ oof_rf_train.csv
│  ├─ oof_gbdt_train.csv
│  ├─ test_lr.csv
│  ├─ test_knn.csv
│  ├─ test_rf.csv
│  ├─ test_gbdt.csv
│  └─ pred_*.npz
├─ src/
│  ├─ preprocess.py
│  ├─ lr_train.py
│  ├─ knn_train.py
│  ├─ rf_tuned.py
│  ├─ gbdt_tuned.py
│  └─ meta_ridge_4base_poly_manual.py
├─ notebooks/
│  └─ eda.ipynb
├─ reports/
│  ├─ figures/
│  └─ report.pdf
└─ README.md
```

---

## 6) Cách chạy (Reproducibility)
### Yêu cầu
- Python 3.10+
- Các thư viện thường dùng: `numpy`, `pandas` (tuỳ script có thể cần thêm thư viện khác)

Cài đặt:
```bash
pip install -r requirements.txt
```

### (1) Tiền xử lý
```bash
python src/preprocess.py --input data/data_raw.csv --output data/data_processed.csv
```

### (2) Train base models + xuất OOF/test predictions
Ví dụ (tuỳ repo của bạn đặt tên file/flags):
```bash
python src/lr_train.py   --data data/data_processed.csv --k 5 --seed 42
python src/knn_train.py  --data data/data_processed.csv --k 5 --seed 42
python src/rf_tuned.py   --data data/data_processed.csv --k 5 --seed 42 --trials 15
python src/gbdt_tuned.py --data data/data_processed.csv --k 5 --seed 42 --trials 15
```

### (3) Train meta-model Ridge (stacking)
```bash
python src/meta_ridge_4base_poly_manual.py
```

---

## 7) Output / Artifacts
Sau khi chạy, bạn sẽ có (tuỳ cách implement):
- `data/oof_*_train.csv`: OOF predictions cho train (dùng để fit meta-model)
- `data/test_*.csv`: dự đoán trên test (phần 20%)
- `data/pred_*.npz`: gói dự đoán/metrics phục vụ báo cáo & vẽ biểu đồ

---

## 8) Ghi chú về Stacking & Leakage
- Nếu train meta-model trên dự đoán của base model được tạo từ **cùng dữ liệu mà base model đã fit**, meta-model sẽ bị “ảo tưởng” (leakage) → RMSE nhìn rất đẹp nhưng không tổng quát.
- Vì vậy, repo sử dụng **K-fold OOF** cho train → meta-model học trên dự đoán “out-of-fold” (base model không nhìn thấy mẫu đó khi train).

---

## 9) Thành viên / Tác giả

- Phạm Lê Anh Duy - 23162012
- Cao Minh Đạt - 23162015
- Nguyễn Hoàng Khang - 23162038
- Lê Quang Khải - 23162042

---

## 10) License
Dùng cho mục đích học tập / đồ án. (Nếu cần, hãy thêm MIT License hoặc license phù hợp.)

---

## 11) Acknowledgements
- Tài liệu/slide môn Machine Learning (Linear Regression, KNN, Decision Tree/Random Forest, …)
