# Cao Minh Đạt - 23162015

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
        # f"- Hãy đặt file vào thư mục 'data/' của project hoặc truyền đường dẫn đầy đủ.\n"
        # f"- Vị trí code hiện tại: {here}"
    )


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

        model = RandomForestRegressor(**(rf_params or {}))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        oof[val_idx] = pred
        fold_r2.append(r2_score(y_val, pred))
        fold_mse.append(mean_squared_error(y_val, pred))

        cur = ve

    if verbose:
        for i, (r2v, msev) in enumerate(zip(fold_r2, fold_mse), start=1):
            print(f"Fold {i}: R² = {r2v:.4f} | MSE = {msev:.4f}")
        print("=" * 50)
        print(f"Mean R²: {float(np.mean(fold_r2)):.4f} (+/- {float(np.std(fold_r2)):.4f})")
        print("=" * 50)
        print(f"OOF MSE: {mean_squared_error(y, oof):.4f}")
        print(f"OOF R²:  {r2_score(y, oof):.4f}")

    return oof, fold_r2, fold_mse


def run_export_pred_rf_for_stacking(data_filename="data_KNN_new.csv", n_folds=5, seed=42):
    csv_path = find_data_csv(data_filename)
    df, X, y, feature_cols = load_selected_features(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Samples: {len(df)} | Features: {len(feature_cols)}")
    print("First cols:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")

    train_idx, val_idx, test_idx = split_60_20_20(len(X), seed=seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    rf_params = dict(
        n_estimators=1000,
        max_depth=None,
        max_features=0.33,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False,
    )

    oof_train, _, _ = cross_validate_oof_rf(
        X_train, y_train,
        n_folds=n_folds,
        shuffle=True,
        random_state=seed,
        rf_params=rf_params,
        verbose=True
    )

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, pred_train)
    val_mse = mean_squared_error(y_val, pred_val)
    test_mse = mean_squared_error(y_test, pred_test)

    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    test_rmse = np.sqrt(test_mse)

    train_r2 = r2_score(y_train, pred_train)
    val_r2 = r2_score(y_val, pred_val)
    test_r2 = r2_score(y_test, pred_test)

    test_mae = mean_absolute_error(y_test, pred_test)

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
        rf_params=np.array([str(rf_params)], dtype=object),
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

    run_export_pred_rf_for_stacking(data_filename=data_file, n_folds=k, seed=seed)
