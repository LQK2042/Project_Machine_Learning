import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load as joblib_load

# QUAN TRỌNG: import class custom trước khi unpickle LR
from Linear_Regression import LinearRegressionGD  # noqa: F401

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = DATA_DIR / "artifacts"

LR_PATH    = ART_DIR / "lr.pkl"
RF_PATH    = ART_DIR / "rf.pkl"
KNN_PATH   = ART_DIR / "knn.pkl"
GBDT_PATH  = ART_DIR / "gbdt.pkl"
RIDGE_PATH = ART_DIR / "ridge.pkl"

STD_PATH = ART_DIR / "standardization_params.json"
FEATURE_COLS_PATH = ART_DIR / "feature_cols.json"

CLIP_RANGE = (1.0, 5.0)

# ============================================================
# UTILS
# ============================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clip_rating(y):
    return np.clip(y, CLIP_RANGE[0], CLIP_RANGE[1])

def _peek_bytes(path: Path, n=16) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_joblib(path: Path):
    return joblib_load(path)

def load_model_auto(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Không thấy artifact {name}: {path}")

    if path.suffix.lower() == ".joblib":
        return _load_joblib(path)

    try:
        return _load_pickle(path)
    except Exception as e1:
        try:
            return _load_joblib(path)
        except Exception as e2:
            head = _peek_bytes(path, 16)
            raise RuntimeError(
                f"[{name}] Không load được artifact: {path}\n"
                f"- pickle error: {repr(e1)}\n"
                f"- joblib error: {repr(e2)}\n"
                f"- first16bytes: {head}\n"
                f"👉 File có thể không phải pickle/joblib hoặc bị ghi đè."
            )

def load_knn_with_fallback(primary_path: Path):
    candidates = [
        primary_path,
        primary_path.with_suffix(".joblib"),
        primary_path.with_name("knn.joblib"),
        primary_path.with_name("knn_model.joblib"),
        primary_path.with_name("model_knn.joblib"),
        primary_path.with_name("knn.pkl"),
    ]
    last_err = None
    for p in candidates:
        if not p.exists():
            continue
        try:
            return load_model_auto(p, "KNN"), p
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError(f"Không tìm thấy artifact KNN. Đã thử: {[str(x) for x in candidates]}")

# ============================================================
# STANDARDIZATION (theo schema num_mean/num_std/num_fill)
# ============================================================
def standardize_raw(df, params, feature_cols):
    df = df.copy()

    numeric_cols = params.get("numeric_cols", [])
    binary_cols  = params.get("binary_cols", [])

    num_mean = params.get("num_mean", {})
    num_std  = params.get("num_std", {})
    num_fill = params.get("num_fill", {})
    bin_fill = params.get("bin_fill", {})

    # numeric
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = num_fill.get(c, 0.0)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(num_fill.get(c, num_mean.get(c, 0.0)))

    # binary
    for c in binary_cols:
        if c not in df.columns:
            df[c] = bin_fill.get(c, 0.0)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(bin_fill.get(c, 0.0))
        df[c] = (df[c] >= 0.5).astype(float)

    # standardize numeric
    for c in numeric_cols:
        mu = float(num_mean.get(c, 0.0))
        sd = float(num_std.get(c, 1.0))
        if sd == 0:
            sd = 1.0
        df[c] = (df[c] - mu) / sd

    # ensure feature_cols
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    return df[feature_cols].to_numpy(float)

# ============================================================
# META FEATURES
# ============================================================
def build_meta_features(P, use_squares, use_interactions, use_abs_diffs, use_stats):
    feats = [P]

    if use_squares:
        feats.append(P ** 2)

    if use_interactions:
        inter = []
        m = P.shape[1]
        for i in range(m):
            for j in range(i + 1, m):
                inter.append((P[:, i] * P[:, j]).reshape(-1, 1))
        feats.append(np.hstack(inter) if inter else np.empty((P.shape[0], 0)))

    if use_abs_diffs:
        dif = []
        m = P.shape[1]
        for i in range(m):
            for j in range(i + 1, m):
                dif.append(np.abs(P[:, i] - P[:, j]).reshape(-1, 1))
        feats.append(np.hstack(dif) if dif else np.empty((P.shape[0], 0)))

    if use_stats:
        feats.append(
            np.hstack([
                np.min(P, axis=1, keepdims=True),
                np.max(P, axis=1, keepdims=True),
                np.mean(P, axis=1, keepdims=True),
                np.std(P, axis=1, keepdims=True),
            ])
        )

    return np.hstack(feats)

# ============================================================
# RIDGE PREDICT
# ============================================================
def predict_ridge(model, X):
    mu, sd, w = model["mu"], model["sd"], model["w"]
    Xs = (X - mu) / sd
    Xb = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    return Xb @ w

# ============================================================
# METRICS
# ============================================================
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst != 0 else float("nan")
    return rmse, mae, r2

# ============================================================
# MAIN
# ============================================================
def predict_from_raw(raw_csv: str):
    print("=== LOAD ARTIFACTS ===")

    feature_cols = load_json(FEATURE_COLS_PATH)
    std_params = load_json(STD_PATH)
    target_col = std_params.get("target_col", "rating")

    lr = load_model_auto(LR_PATH, "LR")
    rf = load_model_auto(RF_PATH, "RF")
    gbdt = load_model_auto(GBDT_PATH, "GBDT")
    knn, knn_used_path = load_knn_with_fallback(KNN_PATH)

    ridge_pack = load_model_auto(RIDGE_PATH, "RIDGE_PACK")
    ridge_model = ridge_pack["model"]
    use_squares = ridge_pack.get("use_squares", False)
    use_interactions = ridge_pack.get("use_interactions", False)
    use_abs_diffs = ridge_pack.get("use_abs_diffs", False)
    use_stats = ridge_pack.get("use_stats", False)

    print("[OK] Models loaded:",
          f"KNN={Path(knn_used_path).name}")

    print("=== LOAD INPUT CSV ===")
    df = pd.read_csv(raw_csv, encoding="utf-8-sig")

    has_y = target_col in df.columns
    y_true = None
    if has_y:
        y_true = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)

    X = standardize_raw(df, std_params, feature_cols)

    print("=== PREDICT ===")
    P = np.column_stack([
        lr.predict(X),
        rf.predict(X),
        knn.predict(X),
        gbdt.predict(X),
    ])

    X_meta = build_meta_features(P, use_squares, use_interactions, use_abs_diffs, use_stats)
    y_pred = clip_rating(predict_ridge(ridge_model, X_meta))

    # OUTPUT chỉ gồm rating_pred (+ original_row_index nếu có)
    out_path = DATA_DIR / "final_prediction.csv"

    if has_y and y_true is not None:
        mask = ~np.isnan(y_true)
        if mask.sum() > 0:
            rmse, mae, r2 = metrics(y_true[mask], y_pred[mask])
            print("=== METRICS ===")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE : {mae:.6f}")
            print(f"R2  : {r2:.6f}")
        else:
            print(f"[WARN] Có cột '{target_col}' nhưng toàn NaN -> không tính được metric.")

    # save + print predictions (gọn)
    out_cols = []
    if "original_row_index" in df.columns:
        out_cols.append("original_row_index")

    out = df[out_cols].copy() if out_cols else pd.DataFrame()
    out["rating_pred"] = y_pred
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    if not has_y:
        print("=== PREDICTIONS (first 20) ===")
        print(out.head(20).to_string(index=False))

    print(f"[DONE] Saved → {out_path}")
    return out
def load_all_artifacts():
    feature_cols = load_json(FEATURE_COLS_PATH)
    std_params = load_json(STD_PATH)

    lr = load_model_auto(LR_PATH, "LR")
    rf = load_model_auto(RF_PATH, "RF")
    gbdt = load_model_auto(GBDT_PATH, "GBDT")
    knn, knn_used_path = load_knn_with_fallback(KNN_PATH)

    ridge_pack = load_model_auto(RIDGE_PATH, "RIDGE_PACK")
    ridge_model = ridge_pack["model"]
    flags = dict(
        use_squares=ridge_pack.get("use_squares", False),
        use_interactions=ridge_pack.get("use_interactions", False),
        use_abs_diffs=ridge_pack.get("use_abs_diffs", False),
        use_stats=ridge_pack.get("use_stats", False),
    )

    return {
        "feature_cols": feature_cols,
        "std_params": std_params,
        "lr": lr, "rf": rf, "knn": knn, "gbdt": gbdt,
        "ridge_model": ridge_model,
        "flags": flags,
        "knn_path": str(knn_used_path)
    }


def build_one_row(budget, popularity, release_year, genres, std_params):
    """
    genres: list[str] ví dụ ["Action", "Drama"] hoặc ["Science Fiction"]
    """
    binary_cols = std_params.get("binary_cols", [])

    row = {
        "budget": float(budget),
        "popularity": float(popularity),
        "release_year": float(release_year),
    }

    # set tất cả genre = 0
    for c in binary_cols:
        row[c] = 0.0

    # bật những genre được chọn
    for g in (genres or []):
        g = str(g).strip().replace("-", " ").replace("_", " ")
        g = " ".join(g.split()).title().replace(" ", "_")
        if g == "Tv_Movie":
            g = "TV_Movie"
        if g == "Sci_Fi":
            g = "Science_Fiction"

        col = f"genre_{g}"
        if col in row:
            row[col] = 1.0

    return pd.DataFrame([row])


def predict_df(df_raw: pd.DataFrame, artifacts: dict):
    X = standardize_raw(df_raw, artifacts["std_params"], artifacts["feature_cols"])

    P = np.column_stack([
        artifacts["lr"].predict(X),
        artifacts["rf"].predict(X),
        artifacts["knn"].predict(X),
        artifacts["gbdt"].predict(X),
    ])

    f = artifacts["flags"]
    X_meta = build_meta_features(
        P,
        f["use_squares"], f["use_interactions"], f["use_abs_diffs"], f["use_stats"]
    )

    y_pred = clip_rating(predict_ridge(artifacts["ridge_model"], X_meta))
    return y_pred

# ============================================================
if __name__ == "__main__":
    import sys
    raw_file = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "data_raw.csv")
    predict_from_raw(raw_file)
