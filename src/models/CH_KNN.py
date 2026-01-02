import csv
import os
import math
from collections import Counter

INPUT_PATH = os.path.join("data", "data_raw.csv")
OUTPUT_PATH = os.path.join("data", "data_KNN_new.csv")

DROP_COLS = {"userId", "movieId"}
TARGET_COL = "rating"
BUDGET_COL = "budget"   # <<< thêm: cột budget để lọc budget = 0


def clean_col(name: str) -> str:
    return (name or "").lstrip("\ufeff").strip()

def norm_col(name: str) -> str:
    return clean_col(name).lower()

def to_float(x: str):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0

def mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 1.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd == 0:
        sd = 1.0
    return mu, sd

def is_binary_column(non_missing_values):
    if not non_missing_values:
        return False
    return set(non_missing_values).issubset({0.0, 1.0})


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH}")

    with open(INPUT_PATH, "r", newline="", encoding="utf-8") as f:
        reader0 = csv.reader(f)
        raw_header = next(reader0)
        cleaned_header = [clean_col(h) for h in raw_header]
        norm_header = [norm_col(h) for h in cleaned_header]

        norm_to_clean = {nh: ch for nh, ch in zip(norm_header, cleaned_header)}

        target_norm = norm_col(TARGET_COL)
        if target_norm not in norm_to_clean:
            raise ValueError(
                f"File không có cột '{TARGET_COL}'.\n"
                f"Các cột đang có: {cleaned_header}\n"
                f"(Lưu ý: code đã tự bỏ BOM/space và không phân biệt hoa thường.)"
            )

        budget_norm = norm_col(BUDGET_COL)
        if budget_norm not in norm_to_clean:
            raise ValueError(
                f"File không có cột '{BUDGET_COL}' để lọc budget=0.\n"
                f"Các cột đang có: {cleaned_header}"
            )

        target_clean = norm_to_clean[target_norm]
        budget_clean = norm_to_clean[budget_norm]

        drop_norms = {norm_col(c) for c in DROP_COLS}

        feature_cols = [
            c for c in cleaned_header
            if norm_col(c) not in drop_norms and norm_col(c) != target_norm
        ]
        out_cols = feature_cols + [target_clean]

        dict_reader = csv.DictReader(f, fieldnames=cleaned_header)

        rows = []
        col_values = {c: [] for c in feature_cols}

        skipped_missing_y = 0
        skipped_budget0 = 0

        for row in dict_reader:
            # 1) bỏ dòng thiếu rating
            y = to_float(row.get(target_clean, ""))
            if y is None:
                skipped_missing_y += 1
                continue

            # 2) bỏ dòng budget = 0 (chỉ bỏ khi budget đọc được và đúng bằng 0)
            b = to_float(row.get(budget_clean, ""))
            if b is not None and b == 0.0:
                skipped_budget0 += 1
                continue

            new_row = {}
            for c in feature_cols:
                v = to_float(row.get(c, ""))
                new_row[c] = v
                if v is not None:
                    col_values[c].append(v)

            new_row[target_clean] = y
            rows.append(new_row)

    if not rows:
        raise ValueError("Không còn dòng dữ liệu hợp lệ sau khi lọc (rating/budget).")

    binary_cols = [c for c in feature_cols if is_binary_column(col_values[c])]
    numeric_cols = [c for c in feature_cols if c not in binary_cols]

    # Impute
    num_fill = {c: median(col_values[c]) for c in numeric_cols}

    bin_fill = {}
    for c in binary_cols:
        if not col_values[c]:
            bin_fill[c] = 0.0
        else:
            cnt = Counter(col_values[c])
            bin_fill[c] = 0.0 if cnt.get(0.0, 0) >= cnt.get(1.0, 0) else 1.0

    # Fit scaler (Standardization) trên dữ liệu sau khi impute
    num_mean, num_std = {}, {}
    for c in numeric_cols:
        imputed_vals = [(r[c] if r[c] is not None else num_fill[c]) for r in rows]
        mu, sd = mean_std(imputed_vals)
        num_mean[c], num_std[c] = mu, sd

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        writer.writeheader()

        for r in rows:
            out = {}

            for c in numeric_cols:
                x = r[c] if r[c] is not None else num_fill[c]
                out[c] = (x - num_mean[c]) / num_std[c]

            for c in binary_cols:
                x = r[c] if r[c] is not None else bin_fill[c]
                out[c] = 1.0 if x >= 0.5 else 0.0

            out[target_clean] = r[target_clean]
            writer.writerow(out)

    print(f"Đã xuất: {OUTPUT_PATH}")
    print(f"Số dòng còn lại: {len(rows)}")
    print(f"Đã loại (thiếu rating): {skipped_missing_y}")
    print(f"Đã loại (budget = 0): {skipped_budget0}")
    print(f"Numeric scaled ({len(numeric_cols)}): {numeric_cols}")
    print(f"Binary kept ({len(binary_cols)}): {binary_cols}")


if __name__ == "__main__":
    main()
