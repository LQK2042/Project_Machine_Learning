# Lê Quang khải - 23162042 
import numpy as np
import pandas as pd
from pathlib import Path

def train_test_split(df, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - test_size))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))

def norm_genre_name(s: str) -> str:

    return s.strip().replace(" ", "_")

def build_X_y_scaled(df: pd.DataFrame,
                     target_col="rating",
                     numeric_cols=("budget", "popularity", "release_year"),
                     genre_prefix="genre_",
                     clip_scaled=True,
                     clip_lo=-5.0,
                     clip_hi=5.0):

    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in list(numeric_cols) + [target_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    genre_cols = [c for c in df.columns if c.startswith(genre_prefix)]
    if not genre_cols:
        raise ValueError(f"No genre_* columns found with prefix '{genre_prefix}'.")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[list(numeric_cols)] = df[list(numeric_cols)].fillna(0)

    # genre -> 0/1
    df[genre_cols] = df[genre_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_num = df[list(numeric_cols)].to_numpy(dtype=np.float32)
    if clip_scaled:
        X_num = np.clip(X_num, clip_lo, clip_hi)

    X_gen = df[genre_cols].to_numpy(dtype=np.float32)
    X = np.hstack([X_num, X_gen]).astype(np.float32)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return X, y, list(numeric_cols), genre_cols

class KNNRegressorChunked:
    def __init__(self, k=50, weighted=True, eps=1e-8,
                 batch_size=256, train_chunk_size=20000):
        self.k = int(k)
        self.weighted = bool(weighted)
        self.eps = float(eps)
        self.batch_size = int(batch_size)
        self.train_chunk_size = int(train_chunk_size)

        self.X_train = None
        self.y_train = None
        self.train_norm = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.float32).reshape(-1)
        self.train_norm = np.sum(self.X_train * self.X_train, axis=1)
        return self

    def predict(self, Xq):
        Xq = np.asarray(Xq, dtype=np.float32)
        n_train = self.X_train.shape[0]
        k = min(self.k, n_train)

        preds = np.empty((Xq.shape[0],), dtype=np.float32)

        for start in range(0, Xq.shape[0], self.batch_size):
            end = min(start + self.batch_size, Xq.shape[0])
            Q = Xq[start:end]
            b = Q.shape[0]

            q_norm = np.sum(Q * Q, axis=1, keepdims=True)

            best_d2 = np.full((b, k), np.inf, dtype=np.float32)
            best_y  = np.zeros((b, k), dtype=np.float32)

            for j in range(0, n_train, self.train_chunk_size):
                T = self.X_train[j:j + self.train_chunk_size]
                yT = self.y_train[j:j + self.train_chunk_size]
                t_norm = self.train_norm[j:j + self.train_chunk_size][None, :]

                d2 = q_norm + t_norm - 2.0 * (Q @ T.T)
                d2 = np.maximum(d2, 0.0)

                c = d2.shape[1]
                kk = min(k, c) 

                idx = np.argpartition(d2, kth=kk-1, axis=1)[:, :kk]
                d2_small = np.take_along_axis(d2, idx, axis=1)
                y_small  = yT[idx]

                merged_d2 = np.concatenate([best_d2, d2_small], axis=1)
                merged_y  = np.concatenate([best_y,  y_small],  axis=1)

                sel = np.argpartition(merged_d2, kth=k-1, axis=1)[:, :k]
                best_d2 = np.take_along_axis(merged_d2, sel, axis=1)
                best_y  = np.take_along_axis(merged_y,  sel, axis=1)

            if not self.weighted:
                preds[start:end] = np.mean(best_y, axis=1)
            else:
                best_d = np.sqrt(best_d2)
                w = 1.0 / (best_d + self.eps)
                preds[start:end] = np.sum(w * best_y, axis=1) / np.sum(w, axis=1)

        return preds

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    csv_path = BASE_DIR / "data" / "data.csv" 

    print("Reading:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    df_train, df_test = train_test_split(df, test_size=0.2, seed=42)

    X_train, y_train, num_cols, gen_cols = build_X_y_scaled(df_train, clip_scaled=True, clip_lo=-5, clip_hi=5)
    X_test,  y_test,  _,       _        = build_X_y_scaled(df_test,  clip_scaled=True, clip_lo=-5, clip_hi=5)

    print("Using numeric columns (scaled):", num_cols)
    print("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)

    model = KNNRegressorChunked(k=50, weighted=True, batch_size=256, train_chunk_size=20000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"RMSE: {rmse(y_test, y_pred):.4f}")
    print(f"MAE : {mae(y_test, y_pred):.4f}")

    genre_to_idx = {g.replace("genre_", ""): i for i, g in enumerate(gen_cols)}
    available_genres = sorted(genre_to_idx.keys())

    print("\n===== PREDICT ONE MOVIE (SCALED INPUT) =====")
    print("Nhập từng thông số (đều là SCALED như data.csv). Gõ 'q' để thoát.\n")
    print("Genres hợp lệ:", ", ".join(available_genres), "\n")

    while True:
        s = input("budget (scaled): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            b = float(s)
        except ValueError:
            print("❌ budget không hợp lệ.\n")
            continue

        s = input("popularity (scaled): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            p = float(s)
        except ValueError:
            print("❌ popularity không hợp lệ.\n")
            continue

        s = input("release_year (scaled): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            y = float(s)
        except ValueError:
            print("❌ release_year không hợp lệ.\n")
            continue

        genres_in = input("genres (comma, vd: Action,Drama): ").strip()
        genres = [norm_genre_name(x) for x in genres_in.split(",") if x.strip()] if genres_in else []

        x_num = np.clip(np.array([b, p, y], dtype=np.float32), -5.0, 5.0)

        x_gen = np.zeros((len(gen_cols),), dtype=np.float32)
        unknown = []
        for g in genres:
            if g in genre_to_idx:
                x_gen[genre_to_idx[g]] = 1.0
            else:
                unknown.append(g)

        if unknown:
            print("⚠️ Genre không có trong dataset (bỏ qua):", unknown)

        X_one = np.hstack([x_num, x_gen])[None, :].astype(np.float32)
        pred = float(model.predict(X_one)[0])
        pred_clamped = max(1.0, min(5.0, pred))
        print(f"✅ Predicted rating: {pred:.4f} | (clamped 1..5): {pred_clamped:.4f}\n")
