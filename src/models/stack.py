
import os
import numpy as np
import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_BASE_FILES = [
    DATA_DIR / "oof_lr_train.csv",
    DATA_DIR / "oof_rf_train.csv",
    DATA_DIR / "oof_knn_train.csv",
    DATA_DIR / "oof_gbdt_train.csv",
]

TEST_BASE_FILES = [
    DATA_DIR / "test_lr.csv",
    DATA_DIR / "test_rf.csv",
    DATA_DIR / "test_knn.csv",
    DATA_DIR / "test_gbdt.csv",
]

OUT_TEST_PRED = DATA_DIR / "stack_ridge_poly_test_pred.csv"

INDEX_COL = "original_row_index"
TRUE_COL = "rating_true"

CLIP_RANGE = (1.0, 5.0)   

ALPHAS = np.logspace(-4, 4, 60)
CV_FOLDS = 5
SEED = 42

USE_SQUARES = True
USE_INTERACTIONS = True
USE_ABS_DIFFS = True
USE_STATS = True

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df

def _detect_pred_col(df: pd.DataFrame, index_col: str, true_col: str | None) -> str:
    forbid = {index_col}
    if true_col is not None and true_col in df.columns:
        forbid.add(true_col)

    candidates = [c for c in df.columns if c not in forbid]
    if len(candidates) == 1:
        return candidates[0]

    keys = ["oof_pred", "pred", "y_pred", "prediction", "rating_pred"]
    for kw in keys:
        hits = [c for c in candidates if kw in c.lower()]
        if len(hits) == 1:
            return hits[0]

    raise ValueError(f"Không nhận ra cột pred. Candidates={candidates}")

def _infer_tag_from_text(text: str, fallback_i: int) -> str:
    s = text.lower()
    if "gbdt" in s or "gbrt" in s or "boost" in s:
        return "gbdt"
    if "knn" in s:
        return "knn"
    if "rf" in s or "forest" in s:
        return "rf"
    if "lr" in s or "linear" in s:
        return "lr"
    return f"m{fallback_i}"

def _load_one_base(path: Path, index_col: str, true_col: str | None, i: int) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Không thấy file: {path}")

    df = _clean_cols(pd.read_csv(path))
    if index_col not in df.columns:
        raise ValueError(f"[{path}] thiếu cột {index_col}")

    pred_col = _detect_pred_col(df, index_col, true_col)

    tag = _infer_tag_from_text(pred_col + " " + path.name, i)
    new_pred = f"pred_{tag}"

    keep = [index_col, pred_col]
    if true_col is not None and true_col in df.columns:
        keep.append(true_col)

    df = df[keep].rename(columns={pred_col: new_pred})

    if df[index_col].duplicated().any():
        ex = df[df[index_col].duplicated()][index_col].head(5).tolist()
        raise ValueError(f"[{path}] {index_col} bị trùng. Ví dụ: {ex}")

    df[new_pred] = pd.to_numeric(df[new_pred], errors="coerce")
    if df[new_pred].isna().any():
        df[new_pred] = df[new_pred].fillna(df[new_pred].mean())

    if true_col is not None and true_col in df.columns:
        df[true_col] = pd.to_numeric(df[true_col], errors="coerce")
        if df[true_col].isna().any():
            raise ValueError(f"[{path}] rating_true có NaN")

    return df

def merge_base_files(paths: list[Path], index_col: str, true_col: str | None) -> pd.DataFrame:
    frames = []
    for i, p in enumerate(paths):
        f = _load_one_base(p, index_col, true_col, i)
        frames.append(f)

    print("Base file sizes:", [len(x) for x in frames])

    merged = frames[0]
    for f in frames[1:]:

        if true_col is not None and true_col in f.columns:
            f = f.drop(columns=[true_col])
        merged = merged.merge(f, on=index_col, how="inner")

    merged = merged.sort_values(index_col).reset_index(drop=True)

    if merged[index_col].nunique() != len(merged):
        raise ValueError("Merge xong bị trùng original_row_index (không nên).")

    if true_col is not None:
        if true_col not in merged.columns:
            raise ValueError(f"Thiếu {true_col} sau merge.")
        if merged[true_col].isna().any():
            raise ValueError("rating_true bị NaN sau merge.")

    print("Merged size:", len(merged), "| unique idx:", merged[index_col].nunique())

    min_len = min(len(x) for x in frames)
    if len(merged) < min_len:
        print("[WARN] Merge bị rớt dòng (inner join). Kiểm tra original_row_index giữa các file có khớp không.")

    return merged

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    folds = []
    cur = 0
    for fs in fold_sizes:
        va = idx[cur:cur + fs]
        tr = np.concatenate([idx[:cur], idx[cur + fs:]])
        folds.append((tr, va))
        cur += fs
    return folds

def build_meta_features(P: np.ndarray):
    """
    P: (n, m) base preds, m = số base models (4)
    Returns X: (n, d)
    """
    P = np.asarray(P, float)
    n, m = P.shape
    feats = [P]  

    if USE_SQUARES:
        feats.append(P ** 2)

    if USE_INTERACTIONS:
        inter = []
        for i in range(m):
            for j in range(i + 1, m):
                inter.append((P[:, i] * P[:, j]).reshape(-1, 1))
        if inter:
            feats.append(np.hstack(inter))

    if USE_ABS_DIFFS:
        dif = []
        for i in range(m):
            for j in range(i + 1, m):
                dif.append(np.abs(P[:, i] - P[:, j]).reshape(-1, 1))
        if dif:
            feats.append(np.hstack(dif))

    if USE_STATS:
        pmin = np.min(P, axis=1, keepdims=True)
        pmax = np.max(P, axis=1, keepdims=True)
        pmean = np.mean(P, axis=1, keepdims=True)
        pstd = np.std(P, axis=1, keepdims=True)
        feats.append(np.hstack([pmin, pmax, pmean, pstd]))

    return np.hstack(feats)

def _standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd

def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def _ridge_fit_from_gram(G: np.ndarray, c: np.ndarray, alpha: float):
    A = G.copy()

    d = A.shape[0] - 1
    diag = np.diag_indices(d + 1)
    A[diag] += alpha
    A[0, 0] -= alpha  
    w = np.linalg.solve(A, c)
    return w

def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    mu, sd = _standardize_fit(X)
    Xs = _standardize_apply(X, mu, sd)
    n = Xs.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), Xs], axis=1)

    G = Xb.T @ Xb
    c = Xb.T @ y
    w = _ridge_fit_from_gram(G, c, float(alpha))

    return {"mu": mu, "sd": sd, "w": w, "alpha": float(alpha)}

def predict_ridge(model, X: np.ndarray):
    X = np.asarray(X, float)
    Xs = _standardize_apply(X, model["mu"], model["sd"])
    n = Xs.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), Xs], axis=1)
    return Xb @ model["w"]

def tune_alpha_cv(X: np.ndarray, y: np.ndarray, alphas, k=5, seed=42):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    folds = kfold_indices(len(y), k, seed)
    best_alpha = None
    best_score = float("inf")

    for a in alphas:
        scores = []
        for tr, va in folds:
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]

            mu, sd = _standardize_fit(Xtr)
            Xtr_s = _standardize_apply(Xtr, mu, sd)
            Xva_s = _standardize_apply(Xva, mu, sd)

            Xtr_b = np.concatenate([np.ones((Xtr_s.shape[0], 1)), Xtr_s], axis=1)
            Xva_b = np.concatenate([np.ones((Xva_s.shape[0], 1)), Xva_s], axis=1)

            G = Xtr_b.T @ Xtr_b
            c = Xtr_b.T @ ytr
            w = _ridge_fit_from_gram(G, c, float(a))

            pred = Xva_b @ w
            scores.append(rmse(yva, pred))

        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = float(a)

    return best_alpha, best_score

def main():

    train_df = merge_base_files(TRAIN_BASE_FILES, INDEX_COL, TRUE_COL)
    pred_cols = [c for c in train_df.columns if c.startswith("pred_")]


    prefer = ["pred_lr", "pred_rf", "pred_knn", "pred_gbdt"]
    pred_cols = [c for c in prefer if c in pred_cols] + [c for c in pred_cols if c not in prefer]

    if len(pred_cols) < 2:
        raise ValueError(f"Cần >=2 cột pred_*. Hiện có: {pred_cols}")

    P_train = train_df[pred_cols].to_numpy(float)
    y_train = train_df[TRUE_COL].to_numpy(float)

    X_train = build_meta_features(P_train)

    best_alpha, cv_rm = tune_alpha_cv(X_train, y_train, ALPHAS, k=CV_FOLDS, seed=SEED)
    model = fit_ridge_closed_form(X_train, y_train, alpha=best_alpha)

    yhat_tr = predict_ridge(model, X_train)
    if CLIP_RANGE is not None:
        yhat_tr = np.clip(yhat_tr, CLIP_RANGE[0], CLIP_RANGE[1])

    print("\n===== RIDGE META (4 BASE, POLY FEATURES, MANUAL) =====")
    print("Base cols:", pred_cols)
    print("Meta feature dim:", X_train.shape[1])
    print(f"Best alpha: {best_alpha:.6g} | CV-RMSE(mean): {cv_rm:.6f}")
    print(f"Train RMSE: {rmse(y_train, yhat_tr):.6f} | MAE: {mae(y_train, yhat_tr):.6f} | R2: {r2(y_train, yhat_tr):.6f}")

    try:
        test_df = merge_base_files(TEST_BASE_FILES, INDEX_COL, TRUE_COL)
        has_y = True
    except Exception:
        test_df = merge_base_files(TEST_BASE_FILES, INDEX_COL, true_col=None)
        has_y = False

    for c in pred_cols:
        if c not in test_df.columns:
            raise ValueError(f"Test thiếu cột {c}. Test có: {[x for x in test_df.columns if x.startswith('pred_')]}")

    P_test = test_df[pred_cols].to_numpy(float)
    X_test = build_meta_features(P_test)

    yhat_te = predict_ridge(model, X_test)
    if CLIP_RANGE is not None:
        yhat_te = np.clip(yhat_te, CLIP_RANGE[0], CLIP_RANGE[1])

    out = pd.DataFrame({
        INDEX_COL: test_df[INDEX_COL].to_numpy(int),
        "rating_pred": yhat_te
    }).sort_values(INDEX_COL, kind="mergesort")

    if has_y and TRUE_COL in test_df.columns:
        y_test = test_df[TRUE_COL].to_numpy(float)

        # META metrics
        meta_rmse = rmse(y_test, yhat_te)
        meta_mae  = mae(y_test, yhat_te)
        meta_r2   = r2(y_test, yhat_te)

        # BASE metrics (RMSE/MAE/R2)
        base_rows = []
        for c in pred_cols:
            yb = test_df[c].to_numpy(float)
            base_rows.append((c, rmse(y_test, yb), mae(y_test, yb), r2(y_test, yb)))

        base_rows.sort(key=lambda x: x[1])  # sort theo RMSE
        best_name, best_rmse, best_mae, best_r2 = base_rows[0]

        improve = best_rmse - meta_rmse
        improve_pct = (improve / best_rmse * 100.0) if best_rmse != 0 else 0.0

        print("\n================ TOTAL PERFORMANCE ================")
        print("META Ridge (Stacking) | 4 base + poly features")
        print("Base cols:", pred_cols)
        print(f"Best alpha: {best_alpha:.6g} | CV-RMSE(mean): {cv_rm:.6f}")
        print(f"Train RMSE: {rmse(y_train, yhat_tr):.6f} | MAE: {mae(y_train, yhat_tr):.6f} | R2: {r2(y_train, yhat_tr):.6f}")

        print("\n--- TEST (FINAL / META) ---")
        print(f"RMSE: {meta_rmse:.6f} | MAE: {meta_mae:.6f} | R2: {meta_r2:.6f}")

        print("\n--- BEST BASE ON TEST ---")
        print(f"{best_name} -> RMSE: {best_rmse:.6f} | MAE: {best_mae:.6f} | R2: {best_r2:.6f}")
        print(f"GAIN (RMSE): {improve:.6f} ({improve_pct:.2f}%) vs best base")

        print("\n--- BASE MODELS (TEST) ---")
        print(f"{'model':<14} {'rmse':>10} {'mae':>10} {'r2':>10}")
        for name, r, m, rr in base_rows:
            print(f"{name:<14} {r:>10.6f} {m:>10.6f} {rr:>10.6f}")

        out[TRUE_COL] = y_test

    out.to_csv(OUT_TEST_PRED, index=False)
    p = (1-meta_rmse/5)*100;
    print(f"Performance: {p}")
    print(f"\nSaved: {OUT_TEST_PRED}")


if __name__ == "__main__":
    main()
