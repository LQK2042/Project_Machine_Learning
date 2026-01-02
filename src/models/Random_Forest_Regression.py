# Cao Minh Đạt - 23162015 

from __future__ import annotations

import re
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def find_csv_upwards(start: Path, rel_path: str = "data/data.csv", max_up: int = 6) -> Path:
    """
    Try to find rel_path by walking up from 'start' directory.
    """
    cur = start
    for _ in range(max_up + 1):
        candidate = cur / rel_path
        if candidate.exists():
            return candidate
        cur = cur.parent
    raise FileNotFoundError(
        f"Không tìm thấy '{rel_path}' khi dò từ: {start}\n"
        f"Gợi ý: hãy truyền đúng đường dẫn hoặc đặt file theo cấu trúc: <project>/data/data.csv"
    )


def canonical_genre_key(s: str) -> str:
    """
    Normalize a genre string so input is case-insensitive and tolerant to spaces/hyphens.
    Examples:
      'Action' -> 'action'
      'Sci-Fi' -> 'sci_fi'
      'Science Fiction' -> 'science_fiction'
    """
    s = s.strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)              
    s = re.sub(r"[^a-z0-9_]", "", s)        
    s = re.sub(r"_+", "_", s).strip("_")    
    return s


def build_feature_schema(
    df: pd.DataFrame,
    target_col: str = "rating",
    numeric_cols: tuple[str, ...] = ("budget", "popularity", "release_year"),
    genre_prefix: str = "genre_",
) -> tuple[list[str], list[str], list[str]]:
    """
    Decide which columns to use, and their exact order.
    Returns: (feature_cols, numeric_cols_list, genre_cols)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    genre_cols = [c for c in df.columns if c.startswith(genre_prefix)]
    if not genre_cols:
        raise ValueError(f"No columns found with prefix '{genre_prefix}'")

  
    num_cols = []
    for c in numeric_cols:
        if c not in df.columns:
            raise ValueError(f"Missing numeric column: {c}")
        num_cols.append(c)

    feature_cols = num_cols + sorted(genre_cols) 
    return feature_cols, num_cols, sorted(genre_cols)


def prepare_X_y(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    genre_cols: list[str],
    target_col: str = "rating",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean + build X, y with consistent columns/order.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(np.float32).to_numpy()

    df = df.reindex(columns=feature_cols, fill_value=0)

    for c in numeric_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        med = float(col.median()) if col.notna().any() else 0.0
        df[c] = col.fillna(med)

    for c in genre_cols:
        col = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = col.clip(0, 1)

    X = df[feature_cols].astype(np.float32).to_numpy()
    return X, y


def build_one_movie_vector(
    budget: float,
    popularity: float,
    release_year: float,
    genres_in: list[str],
    feature_cols: list[str],
    numeric_cols: list[str],
    genre_cols: list[str],
    genre_prefix: str = "genre_",
) -> np.ndarray:
    """
    Create a single-row X vector following the trained schema.
    """
    row = {c: 0.0 for c in feature_cols}
    row[numeric_cols[0]] = float(budget)
    row[numeric_cols[1]] = float(popularity)
    row[numeric_cols[2]] = float(release_year)

    canon_to_col = {}
    for col in genre_cols:
        k = canonical_genre_key(col[len(genre_prefix):])
        canon_to_col[k] = col

    unknown = []
    for g in genres_in:
        kg = canonical_genre_key(g)
        if kg in canon_to_col:
            row[canon_to_col[kg]] = 1.0
        else:
            unknown.append(g)

    if unknown:
        candidates = sorted(canon_to_col.keys())
        suggestions = {}
        for g in unknown:
            kg = canonical_genre_key(g)
            close = difflib.get_close_matches(kg, candidates, n=5, cutoff=0.6)
            suggestions[g] = close

        print(" Genre không có trong dataset:", unknown)
        for g, close in suggestions.items():
            if close:
                print(f"   ↳ Gợi ý cho '{g}': {', '.join(close)}")

    X_one = np.array([row[c] for c in feature_cols], dtype=np.float32)[None, :]
    return X_one


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    csv_path = find_csv_upwards(script_dir, rel_path="data/data.csv", max_up=8)

    print("Reading:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    feature_cols, num_cols, gen_cols = build_feature_schema(
        df,
        target_col="rating",
        numeric_cols=("budget", "popularity", "release_year"),
        genre_prefix="genre_",
    )

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    X_train, y_train = prepare_X_y(df_train, feature_cols, num_cols, gen_cols, target_col="rating")
    X_test, y_test = prepare_X_y(df_test, feature_cols, num_cols, gen_cols, target_col="rating")

    print("Numeric features:", num_cols)
    print("Number of genre features:", len(gen_cols))
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    
    model = RandomForestRegressor(
        n_estimators=10000,
        max_depth=10,
        max_features=0.2,      
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        oob_score=False,
        bootstrap=True,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    print(f"RMSE (TRAIN): {rmse(y_train, y_pred_train):.4f}")

    y_pred_test = model.predict(X_test)
    print(f"RMSE (TEST) : {rmse(y_test, y_pred_test):.4f}")

    y_pred = model.predict(X_test)
    print(f"MAE : {mae(y_test, y_pred):.4f}")
    print(f"R2  : {r2_score(y_test, y_pred):.4f}")


    valid_genres = [canonical_genre_key(c.replace("genre_", "")) for c in gen_cols]
    valid_genres = sorted(set(valid_genres))

    print("\n===== PREDICT ONE MOVIE =====")
    print("Genres hợp lệ :")
    print(", ".join(valid_genres), "\n")

    while True:
        s = input("budget (q để thoát): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            budget = float(s)
        except ValueError:
            print(" budget không hợp lệ\n")
            continue

        s = input("popularity: ").strip()
        try:
            popularity = float(s)
        except ValueError:
            print(" popularity không hợp lệ\n")
            continue

        s = input("release_year: ").strip()
        try:
            release_year = float(s)
        except ValueError:
            print(" release_year không hợp lệ\n")
            continue

        genres_in = input("genres (vd: Action,Drama hoặc sci-fi, science fiction): ").strip()
        genres = [g.strip() for g in genres_in.split(",") if g.strip()]

        X_one = build_one_movie_vector(
            budget=budget,
            popularity=popularity,
            release_year=release_year,
            genres_in=genres,
            feature_cols=feature_cols,
            numeric_cols=num_cols,
            genre_cols=gen_cols,
            genre_prefix="genre_",
        )

        pred = float(model.predict(X_one)[0])
        pred_clamped = max(1.0, min(5.0, pred))
        print(f" Predicted rating: {pred:.4f} \n")
