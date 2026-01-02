import csv
import os
import math
import random
import heapq

# =========================
# CONFIG
# =========================
DATA_PATH = os.path.join("data", "data_KNN_new.csv")
OOF_OUT_PATH = os.path.join("data", "oof_knn_train_full.csv")

SEED = 42
FOLDS = 5

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

NUM_COLS = ["budget", "popularity", "release_year"]
TARGET_COL = "rating"

K_FIXED = 50   


# =========================
# Utilities
# =========================
def clean_col(name: str) -> str:
    return (name or "").lstrip("\ufeff").strip()

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def rmse(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    return math.sqrt(mse)

def mae(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

def r2_score(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    y_mean = sum(y_true) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


# =========================
# Load CSV -> (b, pop, year, mask, rating)
# =========================
def load_knn_bitmask_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_raw = next(reader)

        header = [clean_col(h) for h in header_raw]
        idx_map = {h.lower(): i for i, h in enumerate(header)}

        if TARGET_COL not in idx_map:
            raise ValueError(f"Không có cột '{TARGET_COL}' trong {path}. Header: {header}")

        for c in NUM_COLS:
            if c not in idx_map:
                raise ValueError(f"Thiếu cột '{c}' trong {path}. Header: {header}")

        b_i = idx_map["budget"]
        p_i = idx_map["popularity"]
        y_i = idx_map["release_year"]
        r_i = idx_map[TARGET_COL]

        genre_cols = [h for h in header if h.lower().startswith("genre_")]
        genre_indices = [idx_map[h.lower()] for h in genre_cols]

        budgets, pops, years, masks, ratings, orig_idx = [], [], [], [], [], []
        row_counter = -1

        for row in reader:
            row_counter += 1

            b = to_float(row[b_i]) if b_i < len(row) else None
            p = to_float(row[p_i]) if p_i < len(row) else None
            yr = to_float(row[y_i]) if y_i < len(row) else None
            rt = to_float(row[r_i]) if r_i < len(row) else None

            if b is None or p is None or yr is None or rt is None:
                continue

            m = 0
            for bit, gi in enumerate(genre_indices):
                if gi < len(row):
                    gv = to_float(row[gi])
                    if gv is not None and gv >= 0.5:
                        m |= (1 << bit)

            budgets.append(b)
            pops.append(p)
            years.append(yr)
            masks.append(m)
            ratings.append(rt)
            orig_idx.append(row_counter)

    if not budgets:
        raise ValueError("Không load được dữ liệu hợp lệ (có thể dữ liệu lỗi/NaN).")

    return budgets, pops, years, masks, ratings, orig_idx, genre_cols


# =========================
# Split 60/20/20
# =========================
def split_60_20_20(budgets, pops, years, masks, ratings, orig_idx, seed=42):
    n = len(ratings)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def take(idxs):
        return (
            [budgets[i] for i in idxs],
            [pops[i] for i in idxs],
            [years[i] for i in idxs],
            [masks[i] for i in idxs],
            [ratings[i] for i in idxs],
            [orig_idx[i] for i in idxs],
        )

    return take(train_idx), take(val_idx), take(test_idx)


# =========================
# Distance with bitmask
# dist2 = (db^2 + dp^2 + dy^2) + popcount(mask XOR qmask)
# =========================
def dist2_bitmask(tb, tp, ty, tm, qb, qp, qy, qm):
    db = tb - qb
    dp = tp - qp
    dy = ty - qy
    genre_dist = (tm ^ qm).bit_count()
    return db * db + dp * dp + dy * dy + genre_dist


# =========================
# KNN prediction (UNWEIGHTED)
# =========================
def knn_predict_one(b_tr, p_tr, y_tr, m_tr, r_tr, qb, qp, qy, qm, k):
    heap = []
    sum_y = 0.0

    hpush = heapq.heappush
    hreplace = heapq.heapreplace

    for i in range(len(r_tr)):
        d2 = dist2_bitmask(b_tr[i], p_tr[i], y_tr[i], m_tr[i], qb, qp, qy, qm)

        if len(heap) < k:
            hpush(heap, (-d2, r_tr[i]))
            sum_y += r_tr[i]
        else:
            worst_d2 = -heap[0][0]
            if d2 < worst_d2:
                popped = hreplace(heap, (-d2, r_tr[i]))
                sum_y += r_tr[i] - popped[1]

    return sum_y / len(heap)

def knn_predict_batch(b_tr, p_tr, y_tr, m_tr, r_tr, b_q, p_q, y_q, m_q, k):
    return [
        knn_predict_one(b_tr, p_tr, y_tr, m_tr, r_tr, b_q[i], p_q[i], y_q[i], m_q[i], k)
        for i in range(len(b_q))
    ]


# =========================
# K-fold indices (on TRAIN)
# =========================
def kfold_indices(n, folds=5, seed=42):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1

    folds_idx = []
    start = 0
    for fs in fold_sizes:
        folds_idx.append(idx[start:start + fs])
        start += fs
    return folds_idx


# =========================
# Compute OOF predictions for TRAIN (fold=5)
# =========================
def compute_oof_knn(b_tr, p_tr, y_tr, m_tr, r_tr, k=5, folds=5, seed=42):
    n = len(r_tr)
    oof = [None] * n
    folds_idx = kfold_indices(n, folds=folds, seed=seed)

    for i in range(folds):
        val_ids = folds_idx[i]
        val_set = set(val_ids)
        tr_ids = [j for j in range(n) if j not in val_set]

        bF = [b_tr[j] for j in tr_ids]
        pF = [p_tr[j] for j in tr_ids]
        yF = [y_tr[j] for j in tr_ids]
        mF = [m_tr[j] for j in tr_ids]
        rF = [r_tr[j] for j in tr_ids]

        for j in val_ids:
            oof[j] = knn_predict_one(bF, pF, yF, mF, rF, b_tr[j], p_tr[j], y_tr[j], m_tr[j], k)

    return oof


def export_oof(oof_pred, y_train, orig_train_idx, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_row_index", "oof_pred", "rating_true"])
        for i in range(len(oof_pred)):
            w.writerow([orig_train_idx[i], oof_pred[i], y_train[i]])


# =========================
# MAIN
# =========================
def main():
    budgets, pops, years, masks, ratings, orig_idx, genre_cols = load_knn_bitmask_csv(DATA_PATH)

    print(f"Loaded: {len(ratings)} rows")
    print(f"Features used: {NUM_COLS} + {len(genre_cols)} genre_* (bitmask)")
    print("Note: rating KHÔNG dùng làm feature (rating là target).")
    print(f"Split: train {int(TRAIN_RATIO*100)}% / val {int(VAL_RATIO*100)}% / test {int(TEST_RATIO*100)}%")
    print(f"K-fold on TRAIN: {FOLDS}")
    print(f"Mode: KNN thường (UNWEIGHTED) + Bitmask genres | K_FIXED = {K_FIXED}\n")

    (b_tr, p_tr, y_tr, m_tr, r_tr, id_tr), (b_va, p_va, y_va, m_va, r_va, id_va), (b_te, p_te, y_te, m_te, r_te, id_te) = \
        split_60_20_20(budgets, pops, years, masks, ratings, orig_idx, seed=SEED)

    print(f"Train: {len(r_tr)} | Val: {len(r_va)} | Test: {len(r_te)}\n")

    # ----- Export OOF for TRAIN -----
    print("=== Tạo & xuất OOF predictions (TRAIN, fold=5) ===")
    oof = compute_oof_knn(b_tr, p_tr, y_tr, m_tr, r_tr, k=K_FIXED, folds=FOLDS, seed=SEED)
    export_oof(oof, r_tr, id_tr, OOF_OUT_PATH)
    print(f"Đã xuất: {OOF_OUT_PATH}")
    print(f"OOF Train RMSE={rmse(r_tr, oof):.4f} | MAE={mae(r_tr, oof):.4f} | R2={r2_score(r_tr, oof):.4f}\n")

    # ----- Validate (train=TRAIN, eval=VAL) -----
    print("=== Đánh giá trên VALIDATION (train=TRAIN) ===")
    pred_val = knn_predict_batch(b_tr, p_tr, y_tr, m_tr, r_tr, b_va, p_va, y_va, m_va, K_FIXED)
    print(f"Val RMSE={rmse(r_va, pred_val):.4f} | Val MAE={mae(r_va, pred_val):.4f} | Val R2={r2_score(r_va, pred_val):.4f}\n")

    # ----- Final test (train=TRAIN+VAL, eval=TEST) -----
    print("=== Đánh giá cuối trên TEST (train=TRAIN+VAL) ===")
    b_final = b_tr + b_va
    p_final = p_tr + p_va
    y_final = y_tr + y_va
    m_final = m_tr + m_va
    r_final = r_tr + r_va

    pred_test = knn_predict_batch(b_final, p_final, y_final, m_final, r_final, b_te, p_te, y_te, m_te, K_FIXED)
    print(f"Test RMSE={rmse(r_te, pred_test):.4f} | Test MAE={mae(r_te, pred_test):.4f} | Test R2={r2_score(r_te, pred_test):.4f}")


if __name__ == "__main__":
    main()
