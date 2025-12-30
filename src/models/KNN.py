#LQK

# knn_rating_manual_no_scaling.py
# KNN Regression (manual) - assumes numeric features are already normalized/scaled

import numpy as np
import pandas as pd

def train_test_split(df, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - test_size))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

class KNNRegressorManual:
    def __init__(self, k=50, weighted=True, metric="euclidean", eps=1e-9):
        self.k = int(k)
        self.weighted = bool(weighted)
        self.metric = metric
        self.eps = float(eps)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float).reshape(-1)
        return self

    def _dist(self, Xq):
        if self.metric == "euclidean":
            diff = Xq[:, None, :] - self.X_train[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=2))
        elif self.metric == "manhattan":
            diff = np.abs(Xq[:, None, :] - self.X_train[None, :, :])
            return np.sum(diff, axis=2)
        else:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")

    def predict(self, Xq):
        Xq = np.asarray(Xq, dtype=float)
        dists = self._dist(Xq)  # (m, n)

        k = min(self.k, self.X_train.shape[0])
        knn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]  # (m, k)

        neigh_d = np.take_along_axis(dists, knn_idx, axis=1)  # (m, k)
        neigh_y = self.y_train[knn_idx]                       # (m, k)

        if not self.weighted:
            return np.mean(neigh_y, axis=1)

        w = 1.0 / (neigh_d + self.eps)
        return np.sum(w * neigh_y, axis=1) / np.sum(w, axis=1)

def build_X_y(df: pd.DataFrame,
              numeric_cols=("budget", "runtime", "popularity", "release_year"),
              genre_prefix="genre_",
              target_col="rating"):
    numeric_cols = list(numeric_cols)
    for c in numeric_cols + [target_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    genre_cols = [c for c in df.columns if c.startswith(genre_prefix)]
    if not genre_cols:
        raise ValueError(f"No genre_* columns found with prefix '{genre_prefix}'.")

    X_num = df[numeric_cols].astype(float).to_numpy()
    X_gen = df[genre_cols].astype(float).to_numpy()
    X = np.hstack([X_num, X_gen])
    y = df[target_col].astype(float).to_numpy()
    return X, y, numeric_cols, genre_cols

if __name__ == "__main__":
    csv_path = "data/data.csv"     # hoặc "data\\data.csv" trên Windows
    df = pd.read_csv(csv_path)

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, seed=42)

    # Build features
    X_train, y_train, num_cols, gen_cols = build_X_y(df_train)
    X_test, y_test, _, _ = build_X_y(df_test)

    # (Optional) sanity check: xem numeric có thật đang scale không
    mins = df_train[num_cols].min()
    maxs = df_train[num_cols].max()
    print("Numeric min:\n", mins)
    print("Numeric max:\n", maxs)

    # Train & predict
    model = KNNRegressorManual(k=50, weighted=True, metric="euclidean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"RMSE: {rmse(y_test, y_pred):.4f}")
    print(f"MAE : {mae(y_test, y_pred):.4f}")
