# Phạm Lê Anh Duy - 20162012

import sys
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# Utils: find data file
# =========================
def find_data_csv(filename: str) -> Path:
    """
    Ưu tiên:
    1) Nếu filename là path tồn tại -> dùng luôn
    2) Nếu không, tìm theo cấu trúc project: <root>/data/<filename>
    """
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
# Linear Regression (GD)
# =========================
class LinearRegressionGD:
    def __init__(
        self,
        learning_rate=0.05,
        n_iterations=10_000,
        standardize=False,   # file bạn nói đã z-score -> để False
        numeric_idx=None,
        early_stopping=True,
        tol=1e-8,
        patience=200,
        clip_grad=5.0,
        random_state=42,
    ):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.standardize = bool(standardize)
        self.numeric_idx = None if numeric_idx is None else np.array(numeric_idx, dtype=int)

        self.early_stopping = bool(early_stopping)
        self.tol = float(tol)
        self.patience = int(patience)
        self.clip_grad = float(clip_grad)

        self.random_state = int(random_state)

        self.weights = None
        self.bias = 0.0

        self.x_mean_ = None
        self.x_std_ = None

        self.history_ = {"loss": [], "iterations": 0}

    def _fit_scaler(self, X: np.ndarray):
        if not self.standardize:
            self.x_mean_, self.x_std_ = None, None
            return

        if self.numeric_idx is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            self.x_mean_ = mean
            self.x_std_ = std
        else:
            mean = np.zeros(X.shape[1], dtype=float)
            std = np.ones(X.shape[1], dtype=float)

            sub = X[:, self.numeric_idx]
            m = sub.mean(axis=0)
            s = sub.std(axis=0)
            s[s == 0] = 1.0

            mean[self.numeric_idx] = m
            std[self.numeric_idx] = s
            self.x_mean_ = mean
            self.x_std_ = std

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return X
        if self.x_mean_ is None or self.x_std_ is None:
            raise ValueError("Scaler chưa fit. Hãy gọi fit() trước.")
        return (X - self.x_mean_) / self.x_std_

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        self._fit_scaler(X)
        Xb = self._transform(X)

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        self.history_ = {"loss": [], "iterations": 0}
        best_loss = float("inf")
        wait = 0

        for i in range(self.n_iterations):
            y_pred = Xb @ self.weights + self.bias
            err = y_pred - y
            loss = float(np.mean(err ** 2))
            self.history_["loss"].append(loss)

            dw = (2.0 / n_samples) * (Xb.T @ err)
            db = (2.0 / n_samples) * float(np.sum(err))

            if self.clip_grad is not None and self.clip_grad > 0:
                gnorm = float(np.linalg.norm(dw))
                if gnorm > self.clip_grad:
                    dw = dw * (self.clip_grad / (gnorm + 1e-12))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.early_stopping:
                if best_loss - loss > self.tol:
                    best_loss = loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        self.history_["iterations"] = i + 1
                        break
        else:
            self.history_["iterations"] = self.n_iterations

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._transform(X)
        return Xb @ self.weights + self.bias

    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    def cross_validate_oof(self, X, y, n_folds=5, shuffle=True, random_state=42, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[0]

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

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            m = LinearRegressionGD(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                standardize=self.standardize,
                numeric_idx=self.numeric_idx,
                early_stopping=self.early_stopping,
                tol=self.tol,
                patience=self.patience,
                clip_grad=self.clip_grad,
                random_state=self.random_state,
            )
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)

            oof[val_idx] = pred
            fold_r2.append(self.r2(y_val, pred))
            fold_mse.append(self.mse(y_val, pred))

            cur = ve

        if verbose:
            for i, (r2v, msev) in enumerate(zip(fold_r2, fold_mse), start=1):
                print(f"Fold {i}: R² = {r2v:.4f} | MSE = {msev:.4f}")
            print("=" * 50)
            print(f"Mean R²: {float(np.mean(fold_r2)):.4f} (+/- {float(np.std(fold_r2)):.4f})")
            print("=" * 50)
            print(f"OOF MSE: {self.mse(y, oof):.4f}")
            print(f"OOF R²:  {self.r2(y, oof):.4f}")

        return oof, fold_r2, fold_mse


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


def export_pred_csv(path: Path, original_idx, y_pred, y_true, pred_col_name="pred_lr"):
    out = pd.DataFrame(
        {
            "original_row_index": original_idx.astype(int),
            pred_col_name: np.asarray(y_pred, dtype=float),
            "rating_true": np.asarray(y_true, dtype=float),
        }
    ).sort_values("original_row_index", kind="mergesort")
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    print(out.head(10).to_string(index=False))

def load_selected_features(csv_path: Path):
    df = pd.read_csv(csv_path)

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

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    return df, X, y, feature_cols


def run_export_pred_lr_for_stacking(data_filename="data_KNN_new.csv", n_folds=5, seed=42):
    csv_path = find_data_csv(data_filename)
    df, X, y, feature_cols = load_selected_features(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Samples: {len(df)} | Features: {len(feature_cols)}")
    print("First cols:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")

    train_idx, val_idx, test_idx = split_60_20_20(len(X), seed=seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = LinearRegressionGD(
        learning_rate=0.05,
        n_iterations=10000,
        standardize=False,
        early_stopping=True,
        tol=1e-8,
        patience=200,
        clip_grad=5.0,
        random_state=seed,
    )

    oof_train, _, _ = model.cross_validate_oof(
        X_train, y_train, n_folds=n_folds, shuffle=True, random_state=seed, verbose=True
    )

    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    train_mse = LinearRegressionGD.mse(y_train, pred_train)
    val_mse = LinearRegressionGD.mse(y_val, pred_val)
    test_mse = LinearRegressionGD.mse(y_test, pred_test)

    train_r2 = LinearRegressionGD.r2(y_train, pred_train)
    val_r2 = LinearRegressionGD.r2(y_val, pred_val)
    test_r2 = LinearRegressionGD.r2(y_test, pred_test)

    print(f"Train samples: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Train MSE={train_mse:.4f} | R²={train_r2:.4f}")
    print(f"Val   MSE={val_mse:.4f} | R²={val_r2:.4f}")
    print(f"Test  MSE={test_mse:.4f} | R²={test_r2:.4f}")
    print(f"Used iterations: {model.history_['iterations']}")

    print(f"{'i':<4} {'y_true':>8} {'y_pred':>10} {'abs_err':>10}")
    print("-" * 36)
    for i in range(min(10, len(y_test))):
        ae = abs(y_test[i] - pred_test[i])
        print(f"{i:<4} {y_test[i]:>8.2f} {pred_test[i]:>10.2f} {ae:>10.2f}")
    print("-" * 36)

    # export
    out_dir = csv_path.parent
    export_pred_csv(out_dir / "oof_lr_train.csv", train_idx, oof_train, y_train, pred_col_name="oof_pred_lr")
    export_pred_csv(out_dir / "val_lr.csv", val_idx, pred_val, y_val, pred_col_name="pred_lr")
    export_pred_csv(out_dir / "test_lr.csv", test_idx, pred_test, y_test, pred_col_name="pred_lr")

    out_npz = out_dir / "pred_lr_60_20_20.npz"
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
    )
    print(f"\nSaved NPZ: {out_npz}")
    print("=" * 60)


if __name__ == "__main__":
    data_file = "data_KNN_new.csv"
    k = 5
    seed = 42

    args = sys.argv[1:]
    if "--data" in args:
        data_file = args[args.index("--data") + 1]
    if "--k" in args:
        k = int(args[args.index("--k") + 1])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])

    run_export_pred_lr_for_stacking(data_filename=data_file, n_folds=k, seed=seed)
