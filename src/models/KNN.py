
import sys
import os
import heapq
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# Utils: find file like LR
# =========================
def find_data_csv(filename: str) -> Path:
    p = Path(filename)
    if p.exists():
        return p.resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        cand = parent / "data" / filename
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Không tìm thấy file: '{filename}'.\n"
        f"- Hãy đặt file vào thư mục 'data/' của project hoặc truyền đường dẫn đầy đủ.\n"
        f"- Vị trí code hiện tại: {here}"
    )


# =========================
# Metrics (tương tự)
# =========================
def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


# =========================
# Split 60/20/20 (giống LR)
# =========================
def split_60_20_20(n, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.60 * n)
    n_val = int(0.20 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# =========================
# Export CSV (KHÔNG làm tròn)
# =========================
def export_pred_csv(path: Path, original_idx, y_pred, y_true, pred_col_name="pred_knn"):
    out = pd.DataFrame(
        {
            "original_row_index": np.asarray(original_idx, dtype=int),
            pred_col_name: np.asarray(y_pred, dtype=float),
            "rating_true": np.asarray(y_true, dtype=float),
        }
    ).sort_values("original_row_index", kind="mergesort")

    # CÁCH 1: KHÔNG float_format => giữ full precision
    out.to_csv(path, index=False)

    print(f"\nSaved: {path}")
    print(out.head(10).to_string(index=False))


# =========================
# Load + clean giống LR/RF
# =========================
def load_selected_features_for_knn(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    target = "rating"
    required_numeric = ["budget", "popularity", "release_year"]
    genre_cols = sorted([c for c in df.columns if c.startswith("genre_")])
    feature_cols = required_numeric + genre_cols

    missing = [c for c in required_numeric + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}\nHiện có: {list(df.columns)[:40]} ...")

    for c in feature_cols + [target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=[target]).copy()

    for c in required_numeric:
        df[c] = df[c].fillna(df[c].median())
    for c in genre_cols:
        df[c] = df[c].fillna(0.0)

    budgets = df["budget"].to_numpy(dtype=float)
    pops    = df["popularity"].to_numpy(dtype=float)
    years   = df["release_year"].to_numpy(dtype=float)
    y       = df[target].to_numpy(dtype=float)

    # genre -> bitmask python int (để dùng .bit_count())
    if genre_cols:
        G = (df[genre_cols].to_numpy(dtype=float) >= 0.5).astype(np.int64)
        powers = (1 << np.arange(len(genre_cols), dtype=np.int64))
        masks_np = (G * powers).sum(axis=1).astype(np.int64)
        masks = [int(x) for x in masks_np]  # ép sang python int
    else:
        masks = [0] * len(df)

    return df, budgets, pops, years, masks, y, feature_cols


# =========================
# KNN Regressor (bitmask)
# =========================
class KNNRegressorBitmask:
    def __init__(self, n_neighbors=50):
        self.k = int(n_neighbors)
        self.b_tr = None
        self.p_tr = None
        self.y_tr = None
        self.m_tr = None
        self.r_tr = None

    @staticmethod
    def dist2_bitmask(tb, tp, ty, tm, qb, qp, qy, qm):
        db = tb - qb
        dp = tp - qp
        dy = ty - qy
        genre_dist = (tm ^ qm).bit_count()  # tm,qm là python int
        return db*db + dp*dp + dy*dy + genre_dist

    def fit(self, b_tr, p_tr, y_tr, m_tr, r_tr):
        self.b_tr = list(b_tr)
        self.p_tr = list(p_tr)
        self.y_tr = list(y_tr)
        self.m_tr = list(m_tr)
        self.r_tr = list(r_tr)
        return self

    def predict_one(self, qb, qp, qy, qm):
        n = len(self.r_tr)
        if n == 0:
            return 0.0
        k = self.k if self.k <= n else n

        heap = []
        sum_y = 0.0
        hpush = heapq.heappush
        hreplace = heapq.heapreplace

        for i in range(n):
            d2 = self.dist2_bitmask(
                self.b_tr[i], self.p_tr[i], self.y_tr[i], self.m_tr[i],
                qb, qp, qy, qm
            )
            if len(heap) < k:
                hpush(heap, (-d2, self.r_tr[i]))
                sum_y += self.r_tr[i]
            else:
                worst_d2 = -heap[0][0]
                if d2 < worst_d2:
                    popped = hreplace(heap, (-d2, self.r_tr[i]))
                    sum_y += self.r_tr[i] - popped[1]

        return sum_y / len(heap)

    def predict(self, b_q, p_q, y_q, m_q):
        b_q = list(b_q)
        p_q = list(p_q)
        y_q = list(y_q)
        m_q = list(m_q)

        out = []
        for i in range(len(b_q)):
            out.append(self.predict_one(b_q[i], p_q[i], y_q[i], m_q[i]))
        return np.asarray(out, dtype=float)

    def cross_validate_oof(self, b, p, y_year, m, r, n_folds=5, shuffle=True, random_state=42, verbose=True):
        b = np.asarray(b, dtype=float)
        p = np.asarray(p, dtype=float)
        y_year = np.asarray(y_year, dtype=float)
        r = np.asarray(r, dtype=float).reshape(-1)
        m = list(m)

        n = len(r)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(idx)

        fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
        fold_sizes[: n % n_folds] += 1

        oof = np.zeros(n, dtype=float)
        fold_r2, fold_mse = [], []

        cur = 0
        for f in range(n_folds):
            vs = cur
            ve = cur + fold_sizes[f]
            val_idx = idx[vs:ve]
            tr_idx = np.concatenate([idx[:vs], idx[ve:]])

            b_tr = b[tr_idx].tolist()
            p_tr = p[tr_idx].tolist()
            y_tr = y_year[tr_idx].tolist()
            m_tr = [m[i] for i in tr_idx.tolist()]
            r_tr = r[tr_idx].tolist()

            b_val = b[val_idx].tolist()
            p_val = p[val_idx].tolist()
            y_val = y_year[val_idx].tolist()
            m_val = [m[i] for i in val_idx.tolist()]
            r_val = r[val_idx]

            model = KNNRegressorBitmask(n_neighbors=self.k).fit(b_tr, p_tr, y_tr, m_tr, r_tr)
            pred = model.predict(b_val, p_val, y_val, m_val)

            oof[val_idx] = pred
            fold_r2.append(r2(r_val, pred))
            fold_mse.append(mse(r_val, pred))

            cur = ve

        if verbose:
            for i, (r2v, msev) in enumerate(zip(fold_r2, fold_mse), start=1):
                print(f"Fold {i}: R² = {r2v:.4f} | MSE = {msev:.4f}")
            print("=" * 50)
            print(f"Mean R²: {float(np.mean(fold_r2)):.4f} (+/- {float(np.std(fold_r2)):.4f})")
            print("=" * 50)
            print(f"OOF MSE: {mse(r, oof):.4f}")
            print(f"OOF R²:  {r2(r, oof):.4f}")

        return oof, fold_r2, fold_mse


# =========================
# Runner giống LR
# =========================
def run_export_pred_knn_for_stacking(
    data_filename="data_KNN_new.csv",
    n_folds=5,
    seed=42,
    knn_k=50
):
    csv_path = find_data_csv(data_filename)
    df, budgets, pops, years, masks, y, feature_cols = load_selected_features_for_knn(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Samples: {len(df)} | Features: {len(feature_cols)}")
    print("First cols:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    print(f"KNN neighbors (k) = {knn_k} | OOF folds = {n_folds}")

    train_idx, val_idx, test_idx = split_60_20_20(len(y), seed=seed)

    # TRAIN subset
    b_train = budgets[train_idx]
    p_train = pops[train_idx]
    y_train_year = years[train_idx]
    m_train = [masks[i] for i in train_idx.tolist()]
    y_train = y[train_idx]

    # VAL subset
    b_val = budgets[val_idx]
    p_val = pops[val_idx]
    y_val_year = years[val_idx]
    m_val = [masks[i] for i in val_idx.tolist()]
    y_val = y[val_idx]

    # TEST subset
    b_test = budgets[test_idx]
    p_test = pops[test_idx]
    y_test_year = years[test_idx]
    m_test = [masks[i] for i in test_idx.tolist()]
    y_test = y[test_idx]

    # OOF on TRAIN (giống LR)
    model_for_oof = KNNRegressorBitmask(n_neighbors=knn_k)
    oof_train, _, _ = model_for_oof.cross_validate_oof(
        b_train, p_train, y_train_year, m_train, y_train,
        n_folds=n_folds, shuffle=True, random_state=seed, verbose=True
    )

    # Fit trên TRAIN rồi predict TRAIN/VAL/TEST (giống LR)
    model = KNNRegressorBitmask(n_neighbors=knn_k).fit(
        b_train.tolist(), p_train.tolist(), y_train_year.tolist(), m_train, y_train.tolist()
    )
    pred_train = model.predict(b_train.tolist(), p_train.tolist(), y_train_year.tolist(), m_train)
    pred_val   = model.predict(b_val.tolist(), p_val.tolist(), y_val_year.tolist(), m_val)
    pred_test  = model.predict(b_test.tolist(), p_test.tolist(), y_test_year.tolist(), m_test)

    train_mse = mse(y_train, pred_train)
    val_mse   = mse(y_val, pred_val)
    test_mse  = mse(y_test, pred_test)

    train_r2 = r2(y_train, pred_train)
    val_r2   = r2(y_val, pred_val)
    test_r2  = r2(y_test, pred_test)

    print(f"Train samples: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Train MSE={train_mse:.4f} | R²={train_r2:.4f}")
    print(f"Val   MSE={val_mse:.4f} | R²={val_r2:.4f}")
    print(f"Test  MSE={test_mse:.4f} | R²={test_r2:.4f}")

    # In sample (chỉ hiển thị, file vẫn full precision)
    print(f"{'i':<4} {'y_true':>8} {'y_pred':>12} {'abs_err':>10}")
    print("-" * 40)
    for i in range(min(10, len(y_test))):
        ae = abs(float(y_test[i]) - float(pred_test[i]))
        print(f"{i:<4} {float(y_test[i]):>8.2f} {float(pred_test[i]):>12.6f} {ae:>10.2f}")
    print("-" * 40)

    # export giống LR
    out_dir = csv_path.parent
    export_pred_csv(out_dir / "oof_knn_train.csv", train_idx, oof_train, y_train, pred_col_name="oof_pred_knn")
    export_pred_csv(out_dir / "val_knn.csv", val_idx, pred_val, y_val, pred_col_name="pred_knn")
    export_pred_csv(out_dir / "test_knn.csv", test_idx, pred_test, y_test, pred_col_name="pred_knn")

    out_npz = out_dir / "pred_knn_60_20_20.npz"
    np.savez(
        out_npz,
        oof_train=oof_train,
        pred_train=pred_train,
        pred_val=pred_val,
        pred_test=pred_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_cols=np.array(feature_cols, dtype=object),
        knn_k=int(knn_k),
        oof_folds=int(n_folds),
    )
    print(f"\nSaved NPZ: {out_npz}")
    print("=" * 60)


if __name__ == "__main__":
    data_file = "data_KNN_new.csv"
    folds = 5
    seed = 42
    knn_k = 50

    args = sys.argv[1:]
    if "--data" in args:
        data_file = args[args.index("--data") + 1]
    if "--k" in args:  # giữ giống script LR: --k là số folds
        folds = int(args[args.index("--k") + 1])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--knn_k" in args:  # thêm option riêng cho KNN neighbors
        knn_k = int(args[args.index("--knn_k") + 1])

    run_export_pred_knn_for_stacking(data_filename=data_file, n_folds=folds, seed=seed, knn_k=knn_k)
