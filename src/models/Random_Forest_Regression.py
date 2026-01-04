import sys
import numpy as np
import pandas as pd
from pathlib import Path

def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def find_data_csv(filename: str) -> Path:
    p = Path(filename)
    if p.exists():
        return p.resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        cand = parent / "data" / filename
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"Không tìm thấy file: '{filename}'.")


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


def export_pred_csv(path: Path, original_idx, y_pred, y_true, pred_col_name="pred_rf"):
    out = pd.DataFrame(
        {
            "original_row_index": np.asarray(original_idx, dtype=int),
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
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

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

class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressorCustom:
    def __init__(
        self,
        max_depth=None,
        max_features=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_thresholds=16,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_thresholds = int(n_thresholds)
        self.random_state = int(random_state)

        self.root = None
        self._rng = np.random.default_rng(self.random_state)

        self.X_ = None
        self.y_ = None
        self.n_features_ = 0

    @staticmethod
    def _sse(y):
        n = y.size
        if n == 0:
            return 0.0
        s1 = float(np.sum(y))
        s2 = float(np.sum(y * y))
        return s2 - (s1 * s1) / n

    def _candidate_thresholds(self, col: np.ndarray):
        uniq = np.unique(col)
        if uniq.size <= 1:
            return np.array([], dtype=float)

        if uniq.size <= 20:
            thr = (uniq[:-1] + uniq[1:]) / 2.0
            return thr.astype(float)

        qs = np.linspace(0.05, 0.95, num=self.n_thresholds)
        thr = np.quantile(col, qs)
        thr = np.unique(thr)
        return thr.astype(float)

    def _best_split(self, idx: np.ndarray, depth: int):
        X = self.X_
        y = self.y_
        n = idx.size

        if self.max_features is None or self.max_features >= self.n_features_:
            feat_idx = np.arange(self.n_features_)
        else:
            feat_idx = self._rng.choice(self.n_features_, size=self.max_features, replace=False)

        best_feature = None
        best_threshold = None
        best_sse = float("inf")
        best_left = None
        best_right = None

        y_node = y[idx]
        sse_node = self._sse(y_node)
        if sse_node <= 1e-12:
            return None

        for f in feat_idx:
            col = X[idx, f]
            thresholds = self._candidate_thresholds(col)
            if thresholds.size == 0:
                continue

            for t in thresholds:
                mask_left = col <= t
                n_left = int(np.sum(mask_left))
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_idx = idx[mask_left]
                right_idx = idx[~mask_left]

                sse_left = self._sse(y[left_idx])
                sse_right = self._sse(y[right_idx])
                sse_split = sse_left + sse_right

                if sse_split < best_sse:
                    best_sse = sse_split
                    best_feature = int(f)
                    best_threshold = float(t)
                    best_left = left_idx
                    best_right = right_idx

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_left, best_right

    def _build(self, idx: np.ndarray, depth: int):
        y = self.y_[idx]
        node_value = float(np.mean(y)) if y.size > 0 else 0.0

        if self.max_depth is not None and depth >= self.max_depth:
            return _TreeNode(value=node_value)

        if idx.size < self.min_samples_split:
            return _TreeNode(value=node_value)

        if idx.size < 2 * self.min_samples_leaf:
            return _TreeNode(value=node_value)

        split = self._best_split(idx, depth)
        if split is None:
            return _TreeNode(value=node_value)

        f, t, left_idx, right_idx = split
        left_node = self._build(left_idx, depth + 1)
        right_node = self._build(right_idx, depth + 1)
        return _TreeNode(feature=f, threshold=t, left=left_node, right=right_node, value=None)

    def fit(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray | None = None):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_ = self.X_.shape[1]

        if idx is None:
            idx = np.arange(self.X_.shape[0])
        else:
            idx = np.asarray(idx, dtype=int)

        self.root = self._build(idx, depth=0)
        return self

    def _predict_node(self, node: _TreeNode, X: np.ndarray, rows: np.ndarray, out: np.ndarray):
        if node.value is not None:
            out[rows] = node.value
            return

        f = node.feature
        t = node.threshold
        col = X[rows, f]
        mask_left = col <= t
        left_rows = rows[mask_left]
        right_rows = rows[~mask_left]

        if left_rows.size > 0:
            self._predict_node(node.left, X, left_rows, out)
        if right_rows.size > 0:
            self._predict_node(node.right, X, right_rows, out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros(X.shape[0], dtype=float)
        rows = np.arange(X.shape[0], dtype=int)
        self._predict_node(self.root, X, rows, out)
        return out

class RandomForestRegressorCustom:
    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        max_features=0.33,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
        n_thresholds=16,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.bootstrap = bool(bootstrap)
        self.random_state = int(random_state)
        self.n_thresholds = int(n_thresholds)

        self.trees_ = []
        self._rng = np.random.default_rng(self.random_state)
        self._max_features_int = None

        self.X_ = None
        self.y_ = None
        self.n_features_ = 0

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, (int, np.integer)):
            return max(1, min(int(mf), n_features))
        mf = float(mf)
        if mf <= 0:
            return 1
        if mf <= 1:
            return max(1, min(int(round(mf * n_features)), n_features))
        return max(1, min(int(mf), n_features))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        n = self.X_.shape[0]
        self.n_features_ = self.X_.shape[1]
        self._max_features_int = self._resolve_max_features(self.n_features_)

        self.trees_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_idx = self._rng.integers(0, n, size=n, endpoint=False)
            else:
                sample_idx = np.arange(n, dtype=int)

            tree_seed = int(self._rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                max_features=self._max_features_int,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_thresholds=self.n_thresholds,
                random_state=tree_seed,
            )
            tree.fit(self.X_, self.y_, idx=sample_idx)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.trees_:
            raise ValueError("Model chưa fit.")
        preds = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees_:
            preds += tree.predict(X)
        preds /= len(self.trees_)
        return preds

def cross_validate_oof_rf(
    X, y,
    n_folds=5,
    shuffle=True,
    random_state=42,
    rf_params=None,
    verbose=True
):
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

        params = dict(rf_params or {})
        params["random_state"] = int(params.get("random_state", random_state) + f)

        model = RandomForestRegressorCustom(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        oof[val_idx] = pred
        fold_r2.append(r2_score(y_val, pred))
        fold_mse.append(mse(y_val, pred))

        cur = ve

    if verbose:
        for i, (r2v, msev) in enumerate(zip(fold_r2, fold_mse), start=1):
            print(f"Fold {i}: R² = {r2v:.4f} | MSE = {msev:.4f}")
        print("=" * 50)
        print(f"Mean R²: {float(np.mean(fold_r2)):.4f} (+/- {float(np.std(fold_r2)):.4f})")
        print("=" * 50)
        print(f"OOF MSE: {mse(y, oof):.4f}")
        print(f"OOF R²:  {r2_score(y, oof):.4f}")

    return oof, fold_r2, fold_mse

def tune_rf_by_validation(X_train, y_train, X_val, y_val, seed=42, trials=15):
    """
    Random search trên một số hyperparams phổ biến.
    Fit train -> chấm RMSE trên validation -> chọn best.
    """
    rng = np.random.default_rng(seed)

    best_params = None
    best_val_rmse = float("inf")

    print("\n[TUNE RF] Auto-tune RF params by VALIDATION (RMSE) ...")
    print(f"- Trials: {trials}")

    for t in range(1, trials + 1):

        params = dict(
            n_estimators=int(rng.integers(30, 121)),               
            max_depth=int(rng.integers(6, 21)),                 
            max_features=float(rng.uniform(0.20, 0.80)),          
            min_samples_split=int(rng.choice([2, 4, 8])),
            min_samples_leaf=int(rng.choice([1, 2, 4])),
            bootstrap=True,
            random_state=int(seed),
            n_thresholds=int(rng.choice([16, 32])),
        )

        model = RandomForestRegressorCustom(**params)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)

        cur_rmse = rmse(y_val, pred_val)
        cur_r2 = r2_score(y_val, pred_val)

        print(
            f"  Trial {t:02d}/{trials} | Val RMSE={cur_rmse:.4f} | R²={cur_r2:.4f} | "
            f"n_est={params['n_estimators']}, depth={params['max_depth']}, mf={params['max_features']:.2f}, "
            f"split={params['min_samples_split']}, leaf={params['min_samples_leaf']}, nth={params['n_thresholds']}"
        )

        if cur_rmse < best_val_rmse:
            best_val_rmse = cur_rmse
            best_params = params

    print("\n[TUNE RF] BEST PARAMS (by Val RMSE):")
    print(best_params)
    print(f"[TUNE RF] BEST Val RMSE = {best_val_rmse:.4f}\n")

    return best_params

def run_export_pred_rf_for_stacking(data_filename="data_KNN_new.csv", n_folds=5, seed=42, trials=15):
    csv_path = find_data_csv(data_filename)
    df, X, y, feature_cols = load_selected_features(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Samples: {len(df)} | Features: {len(feature_cols)}")
    print("First cols:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")

    train_idx, val_idx, test_idx = split_60_20_20(len(X), seed=seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    best_rf_params = tune_rf_by_validation(X_train, y_train, X_val, y_val, seed=seed, trials=trials)

    oof_train, _, _ = cross_validate_oof_rf(
        X_train, y_train,
        n_folds=n_folds,
        shuffle=True,
        random_state=seed,
        rf_params=best_rf_params,
        verbose=True
    )

    model = RandomForestRegressorCustom(**best_rf_params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    train_rmse = rmse(y_train, pred_train)
    val_rmse = rmse(y_val, pred_val)
    test_rmse = rmse(y_test, pred_test)

    train_r2 = r2_score(y_train, pred_train)
    val_r2 = r2_score(y_val, pred_val)
    test_r2 = r2_score(y_test, pred_test)

    test_mae = mae(y_test, pred_test)

    print(f"\nTrain samples: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Train RMSE={train_rmse:.4f} | R²={train_r2:.4f}")
    print(f"Val   RMSE={val_rmse:.4f} | R²={val_r2:.4f}")
    print(f"Test  RMSE={test_rmse:.4f} | R²={test_r2:.4f} | MAE={test_mae:.4f}")

    print(f"\n{'i':<4} {'y_true':>8} {'y_pred':>10} {'abs_err':>10}")
    print("-" * 36)
    for i in range(min(10, len(y_test))):
        ae = abs(y_test[i] - pred_test[i])
        print(f"{i:<4} {y_test[i]:>8.2f} {pred_test[i]:>10.2f} {ae:>10.2f}")
    print("-" * 36)

    out_dir = csv_path.parent
    export_pred_csv(out_dir / "oof_rf_train.csv", train_idx, oof_train, y_train, pred_col_name="oof_pred_rf")
    export_pred_csv(out_dir / "val_rf.csv", val_idx, pred_val, y_val, pred_col_name="pred_rf")
    export_pred_csv(out_dir / "test_rf.csv", test_idx, pred_test, y_test, pred_col_name="pred_rf")

    out_npz = out_dir / "pred_rf_60_20_20.npz"
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
        rf_best_params=np.array([str(best_rf_params)], dtype=object),
        rf_oof_folds=int(n_folds),
        rf_tune_trials=int(trials),
    )
    print(f"\nSaved NPZ: {out_npz}")
    print("=" * 60)

if __name__ == "__main__":
    data_file = "data_KNN_new.csv"
    k = 5
    seed = 42
    trials = 15

    args = sys.argv[1:]
    if "--data" in args:
        data_file = args[args.index("--data") + 1]
    if "--k" in args:
        k = int(args[args.index("--k") + 1])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--trials" in args:
        trials = int(args[args.index("--trials") + 1])

    run_export_pred_rf_for_stacking(data_filename=data_file, n_folds=k, seed=seed, trials=trials)
