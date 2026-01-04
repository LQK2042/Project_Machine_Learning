
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

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

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

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

def export_pred_csv(path: Path, original_idx, y_pred, y_true, pred_col_name="pred_gbdt"):
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

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    return df, X, y, feature_cols

class _TreeNode:
    __slots__ = ("is_leaf", "value", "feat", "thr", "left", "right")
    def __init__(self, is_leaf=True, value=0.0, feat=-1, thr=0.0, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = float(value)
        self.feat = int(feat)
        self.thr = float(thr)
        self.left = left
        self.right = right

class RegressionTree:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_splits_per_feature=32,
        min_gain=1e-12,
        random_state=42,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_splits_per_feature = int(max_splits_per_feature)
        self.min_gain = float(min_gain)
        self.random_state = int(random_state)
        self.root = None

    @staticmethod
    def _sse(sum_y, sum_y2, n):

        if n <= 0:
            return 0.0
        return float(sum_y2 - (sum_y * sum_y) / n)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        n = len(y)
        idx = np.arange(n, dtype=int)
        self.root = self._build(X, y, idx, depth=0)
        return self

    def _best_split_for_feature(self, Xf, y, idx):

        x = Xf[idx]
        r = y[idx]
        n = len(idx)
        if n < self.min_samples_split or n < 2 * self.min_samples_leaf:
            return (0.0, None)

        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        r_sorted = r[order]

        pref = np.cumsum(r_sorted)
        pref2 = np.cumsum(r_sorted * r_sorted)

        total_sum = float(pref[-1])
        total_sum2 = float(pref2[-1])
        base_sse = self._sse(total_sum, total_sum2, n)

        left_min = self.min_samples_leaf
        right_min = self.min_samples_leaf
        lo = left_min
        hi = n - right_min
        if lo >= hi:
            return (0.0, None)

        k = min(self.max_splits_per_feature, hi - lo)
        if k <= 0:
            return (0.0, None)

        cand = np.linspace(lo, hi - 1, num=k, dtype=int)
        cand = np.unique(cand)

        best_gain = 0.0
        best_thr = None

        for s in cand:

            if s <= 0 or s >= n:
                continue
            if x_sorted[s - 1] == x_sorted[s]:
                continue

            nL = s
            nR = n - s

            sumL = float(pref[s - 1])
            sumL2 = float(pref2[s - 1])

            sumR = total_sum - sumL
            sumR2 = total_sum2 - sumL2

            sseL = self._sse(sumL, sumL2, nL)
            sseR = self._sse(sumR, sumR2, nR)

            gain = base_sse - (sseL + sseR)
            if gain > best_gain:
                best_gain = gain
                best_thr = float((x_sorted[s - 1] + x_sorted[s]) * 0.5)

        return (best_gain, best_thr)

    def _build(self, X, y, idx, depth):

        r = y[idx]
        node_value = float(np.mean(r)) if len(r) else 0.0

        if depth >= self.max_depth:
            return _TreeNode(is_leaf=True, value=node_value)
        if len(idx) < self.min_samples_split:
            return _TreeNode(is_leaf=True, value=node_value)

        n, d = X.shape
        best_gain = 0.0
        best_feat = None
        best_thr = None

        for f in range(d):
            gain, thr = self._best_split_for_feature(X[:, f], y, idx)
            if thr is None:
                continue
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thr = thr

        if best_feat is None or best_gain < self.min_gain:
            return _TreeNode(is_leaf=True, value=node_value)

        xnode = X[idx, best_feat]
        left_mask = xnode <= best_thr
        right_mask = ~left_mask

        left_idx = idx[left_mask]
        right_idx = idx[right_mask]

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return _TreeNode(is_leaf=True, value=node_value)

        left_child = self._build(X, y, left_idx, depth + 1)
        right_child = self._build(X, y, right_idx, depth + 1)

        return _TreeNode(is_leaf=False, value=node_value, feat=best_feat, thr=best_thr,
                         left=left_child, right=right_child)

    def _predict_one(self, x, node: _TreeNode):
        while not node.is_leaf:
            if x[node.feat] <= node.thr:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            out[i] = self._predict_one(X[i], self.root)
        return out

class GBDTRegressorManual:
    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=5,
        subsample=1.0,
        max_splits_per_feature=32,
        min_gain=1e-12,
        random_state=42,
        clip_pred=(1.0, 5.0),
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.max_splits_per_feature = int(max_splits_per_feature)
        self.min_gain = float(min_gain)
        self.random_state = int(random_state)
        self.clip_pred = clip_pred

        self.init_ = 0.0
        self.trees_ = []

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=0, verbose=False):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)

        rng = np.random.default_rng(self.random_state)

        self.init_ = float(np.mean(y))
        pred = np.full(len(y), self.init_, dtype=float)

        self.trees_ = []
        best_val_rmse = float("inf")
        best_iter = -1
        patience = 0

        for t in range(self.n_estimators):
            resid = y - pred

            if self.subsample < 1.0:
                m = max(2, int(len(y) * self.subsample))
                samp_idx = rng.choice(len(y), size=m, replace=False)
                X_fit = X[samp_idx]
                r_fit = resid[samp_idx]
            else:
                X_fit = X
                r_fit = resid

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_splits_per_feature=self.max_splits_per_feature,
                min_gain=self.min_gain,
                random_state=int(rng.integers(0, 1_000_000_000)),
            ).fit(X_fit, r_fit)

            update = tree.predict(X)
            pred = pred + self.learning_rate * update
            self.trees_.append(tree)

            if X_val is not None and y_val is not None and early_stopping_rounds and early_stopping_rounds > 0:
                val_pred = self.predict(X_val)
                cur = rmse(y_val, val_pred)
                if cur + 1e-12 < best_val_rmse:
                    best_val_rmse = cur
                    best_iter = t
                    patience = 0
                else:
                    patience += 1
                if verbose and (t % 20 == 0 or t == self.n_estimators - 1):
                    print(f"[GBDT] iter={t+1}/{self.n_estimators} val_RMSE={cur:.6f} best={best_val_rmse:.6f} patience={patience}")

                if patience >= early_stopping_rounds:
                    if best_iter >= 0:
                        self.trees_ = self.trees_[: best_iter + 1]
                    break
            else:
                if verbose and (t % 50 == 0 or t == self.n_estimators - 1):
                    print(f"[GBDT] iter={t+1}/{self.n_estimators}")

        return self

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)

        if self.clip_pred is not None:
            lo, hi = self.clip_pred
            pred = np.clip(pred, lo, hi)
        return pred

    def cross_validate_oof(self, X, y, n_folds=5, shuffle=True, random_state=42, verbose=True):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)

        n = len(y)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(idx)

        fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
        fold_sizes[: n % n_folds] += 1

        oof = np.zeros(n, dtype=float)
        fold_rmse = []

        cur = 0
        for f in range(n_folds):
            vs = cur
            ve = cur + fold_sizes[f]
            val_idx = idx[vs:ve]
            tr_idx = np.concatenate([idx[:vs], idx[ve:]])

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            model = GBDTRegressorManual(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                max_splits_per_feature=self.max_splits_per_feature,
                min_gain=self.min_gain,
                random_state=(random_state + 12345 + f),
                clip_pred=self.clip_pred,
            )

            model.fit(X_tr, y_tr)
            pred_va = model.predict(X_va)
            oof[val_idx] = pred_va

            fr = rmse(y_va, pred_va)
            fold_rmse.append(fr)

            if verbose:
                print(f"Fold {f+1}: RMSE={fr:.6f} | R2={r2(y_va, pred_va):.6f}")

            cur = ve

        if verbose:
            print("=" * 50)
            print(f"Mean RMSE: {float(np.mean(fold_rmse)):.6f} (+/- {float(np.std(fold_rmse)):.6f})")
            print(f"OOF  RMSE: {rmse(y, oof):.6f}")
            print(f"OOF  R2:   {r2(y, oof):.6f}")
            print("=" * 50)

        return oof, fold_rmse

def tune_gbdt_on_validation(
    X_train, y_train,
    X_val, y_val,
    grid,
    tune_val_size=3000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    n_val = len(y_val)

    if tune_val_size is None or tune_val_size < 0 or tune_val_size >= n_val:
        sub = np.arange(n_val)
    else:
        sub = rng.choice(n_val, size=int(tune_val_size), replace=False)

    Xv = X_val[sub]
    yv = y_val[sub]

    best_cfg = None
    best_rm = float("inf")

    print("\n[TUNE GBDT] Tuning on validation:")
    print(f"- Validation used: {len(sub)}/{n_val}")
    print(f"- Candidates: {len(grid)} combos\n")

    for i, cfg in enumerate(grid, start=1):
        model = GBDTRegressorManual(**cfg, random_state=seed)
        model.fit(X_train, y_train)
        pred = model.predict(Xv)
        cur_rm = rmse(yv, pred)
        cur_r2 = r2(yv, pred)
        print(f"[{i:02d}] cfg={cfg} => RMSE={cur_rm:.6f} | R2={cur_r2:.6f}")

        if cur_rm < best_rm:
            best_rm = cur_rm
            best_cfg = cfg

    print(f"\n[TUNE GBDT] Best cfg = {best_cfg} | RMSE={best_rm:.6f}\n")
    return best_cfg

def run_export_pred_gbdt_for_stacking(
    data_filename="data_KNN_new.csv",
    n_folds=5,
    seed=42,
    tune_gbdt=True,
    tune_val_size=3000,
):
    csv_path = find_data_csv(data_filename)
    df, X, y, feature_cols = load_selected_features(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Samples: {len(df)} | Features: {len(feature_cols)}")
    print("First cols:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")

    train_idx, val_idx, test_idx = split_60_20_20(len(y), seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    grid = [
        dict(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=10, subsample=1.0, max_splits_per_feature=32, clip_pred=(1.0, 5.0)),
        dict(n_estimators=400, learning_rate=0.05, max_depth=3, min_samples_leaf=10, subsample=0.8, max_splits_per_feature=32, clip_pred=(1.0, 5.0)),
        dict(n_estimators=400, learning_rate=0.03, max_depth=3, min_samples_leaf=10, subsample=0.8, max_splits_per_feature=32, clip_pred=(1.0, 5.0)),
        dict(n_estimators=600, learning_rate=0.03, max_depth=4, min_samples_leaf=10, subsample=0.8, max_splits_per_feature=32, clip_pred=(1.0, 5.0)),
        dict(n_estimators=800, learning_rate=0.02, max_depth=4, min_samples_leaf=20, subsample=0.8, max_splits_per_feature=32, clip_pred=(1.0, 5.0)),
    ]

    if tune_gbdt:
        best_cfg = tune_gbdt_on_validation(
            X_train, y_train, X_val, y_val,
            grid=grid,
            tune_val_size=tune_val_size,
            seed=seed
        )
    else:
        best_cfg = grid[0]
        print("[INFO] --no_tune => dùng cfg mặc định:", best_cfg)

    print(f"\n[OOF] Build OOF with cfg={best_cfg}, folds={n_folds}")
    oof_model = GBDTRegressorManual(**best_cfg, random_state=seed)
    oof_train, _ = oof_model.cross_validate_oof(
        X_train, y_train, n_folds=n_folds, shuffle=True, random_state=seed, verbose=True
    )

    model = GBDTRegressorManual(**best_cfg, random_state=seed)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    print(f"\nTrain samples: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Train RMSE={rmse(y_train, pred_train):.4f} | R²={r2(y_train, pred_train):.4f} | MAE={mae(y_train, pred_train):.4f}")
    print(f"Val   RMSE={rmse(y_val, pred_val):.4f} | R²={r2(y_val, pred_val):.4f} | MAE={mae(y_val, pred_val):.4f}")
    print(f"Test  RMSE={rmse(y_test, pred_test):.4f} | R²={r2(y_test, pred_test):.4f} | MAE={mae(y_test, pred_test):.4f}")

    out_dir = csv_path.parent
    export_pred_csv(out_dir / "oof_gbdt_train.csv", train_idx, oof_train, y_train, pred_col_name="oof_pred_gbdt")
    export_pred_csv(out_dir / "val_gbdt.csv", val_idx, pred_val, y_val, pred_col_name="pred_gbdt")
    export_pred_csv(out_dir / "test_gbdt.csv", test_idx, pred_test, y_test, pred_col_name="pred_gbdt")

    out_npz = out_dir / "pred_gbdt_60_20_20.npz"
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
        best_cfg=str(best_cfg),
        oof_folds=int(n_folds),
    )
    print(f"\nSaved NPZ: {out_npz}")
    print("=" * 60)

if __name__ == "__main__":
    data_file = "data_KNN_new.csv"
    folds = 5
    seed = 42
    tune_gbdt = True
    tune_val_size = 3000

    args = sys.argv[1:]

    if "--data" in args:
        data_file = args[args.index("--data") + 1]
    if "--folds" in args:
        folds = int(args[args.index("--folds") + 1])
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])
    if "--no_tune" in args:
        tune_gbdt = False
    if "--tune_val_size" in args:
        tune_val_size = int(args[args.index("--tune_val_size") + 1])

    run_export_pred_gbdt_for_stacking(
        data_filename=data_file,
        n_folds=folds,
        seed=seed,
        tune_gbdt=tune_gbdt,
        tune_val_size=tune_val_size,
    )
