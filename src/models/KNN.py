# Lê Quang Khải - 23162042
import csv
import os
import math
import heapq
import tkinter as tk
from tkinter import ttk, messagebox

RAW_PATH = os.path.join("data", "data_raw.csv")
KNN_PATH = os.path.join("data", "data_KNN.csv")

NUM_COLS = ["budget", "popularity", "release_year"]
TARGET_COL = "rating"
EPS = 1e-9

def clean_col(name: str) -> str:
    return (name or "").lstrip("\ufeff").strip()

def norm_col(name: str) -> str:
    return clean_col(name).lower()

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(vals[mid]) if n % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2.0

def mean_std(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 1.0
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n  
    sd = math.sqrt(var)
    if sd == 0:
        sd = 1.0
    return mu, sd

def squared_distance(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def get_feature_cols_from_knn(knn_path):
    with open(knn_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    header = [clean_col(h) for h in header]
    header_norm = [h.lower() for h in header]

    if TARGET_COL not in header_norm:
        raise ValueError(f"Không thấy cột '{TARGET_COL}' trong data_KNN.csv")

    lower_to_clean = {h.lower(): h for h in header}

    for c in NUM_COLS:
        if c not in lower_to_clean:
            raise ValueError(f"data_KNN.csv thiếu cột '{c}'")

    genre_cols = [h for h in header if norm_col(h).startswith("genre_")]
    feature_cols = [lower_to_clean[c] for c in NUM_COLS] + genre_cols
    target_clean = header[header_norm.index(TARGET_COL)]
    return feature_cols, genre_cols, target_clean


def compute_scaler_from_raw(raw_path):

    with open(raw_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [clean_col(h) for h in header]
        header_norm = [h.lower() for h in header]
        lower_to_clean = {h.lower(): h for h in header}

        missing = [c for c in NUM_COLS if c not in header_norm]
        if missing:
            raise ValueError(f"data_raw.csv thiếu cột: {missing}")

        num_clean = {c: lower_to_clean[c] for c in NUM_COLS}
        dict_reader = csv.DictReader(f, fieldnames=header)

        num_vals = {c: [] for c in NUM_COLS}
        rows_num = []

        for row in dict_reader:
            item = {}
            for c in NUM_COLS:
                v = to_float(row.get(num_clean[c], ""))
                item[c] = v
                if v is not None:
                    num_vals[c].append(v)
            rows_num.append(item)

    fills = {c: median(num_vals[c]) for c in NUM_COLS}
    scaler = {}
    for c in NUM_COLS:
        imputed = [(r[c] if r[c] is not None else fills[c]) for r in rows_num]
        mu, sd = mean_std(imputed)
        scaler[c] = (mu, sd, fills[c]) 
    return scaler


def knn_predict_from_knn_csv(knn_path, feature_cols, target_col, xq, k=5, weighted=True):

    heap = []  

    with open(knn_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("data_KNN.csv không có header")

        orig_fields = list(reader.fieldnames)
        clean_fields = [clean_col(h) for h in orig_fields]
        clean_to_orig = {}
        for o, c in zip(orig_fields, clean_fields):
            if c not in clean_to_orig:
                clean_to_orig[c] = o

        target_key = clean_to_orig.get(target_col, target_col)
        feature_keys = [clean_to_orig.get(c, c) for c in feature_cols]

        for row in reader:
            y = to_float(row.get(target_key))
            if y is None:
                continue

            x = []
            ok = True
            for fk in feature_keys:
                v = to_float(row.get(fk))
                if v is None:
                    ok = False
                    break
                x.append(v)
            if not ok:
                continue

            dist2 = squared_distance(x, xq)

            if len(heap) < k:
                heapq.heappush(heap, (-dist2, y))
            else:
                worst_dist2 = -heap[0][0]
                if dist2 < worst_dist2:
                    heapq.heapreplace(heap, (-dist2, y))

    if not heap:
        raise ValueError("Không tìm được neighbor hợp lệ trong data_KNN.csv")

    neighbors = [(-dneg, y) for (dneg, y) in heap]  # (dist2, y)

    if not weighted:
        return sum(y for _, y in neighbors) / len(neighbors)

    num = 0.0
    den = 0.0
    for dist2, y in neighbors:
        w = 1.0 / (dist2 + EPS)
        num += w * y
        den += w
    return num / den


def build_query_vector_from_raw_inputs(feature_cols, genre_cols, scaler, budget, popularity, release_year, chosen_genres_set):
    # raw -> z-score for NUM_COLS; genres 0/1
    raw_num = {"budget": budget, "popularity": popularity, "release_year": release_year}

    vec = []
    for col in feature_cols:
        cn = norm_col(col)
        if cn in NUM_COLS:
            mu, sd, fill = scaler[cn]
            x = raw_num[cn]
            if x is None:
                x = fill
            vec.append((x - mu) / sd)
        elif cn.startswith("genre_"):
            vec.append(1.0 if col in chosen_genres_set else 0.0)
        else:
            vec.append(0.0)
    return vec

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)


class KNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KNN Rating Predictor")
        self.geometry("900x600")

        try:
            if not os.path.exists(RAW_PATH):
                raise FileNotFoundError(f"Không thấy {RAW_PATH}")
            if not os.path.exists(KNN_PATH):
                raise FileNotFoundError(f"Không thấy {KNN_PATH}")

            self.feature_cols, self.genre_cols, self.target_col = get_feature_cols_from_knn(KNN_PATH)
            self.scaler = compute_scaler_from_raw(RAW_PATH)
        except Exception as e:
            messagebox.showerror("Lỗi load dữ liệu", str(e))
            self.destroy()
            return

        self.selected = set() 
        self.genre_buttons = {} 

        self._build_ui()

    def _build_ui(self):

        left = ttk.Frame(self, padding=12)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Nhập thông tin phim", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Tên phim:").pack(anchor="w")
        self.movie_name = ttk.Entry(left, width=35)
        self.movie_name.pack(anchor="w", pady=(0, 10))

        self.budget_var = tk.StringVar()
        self.pop_var = tk.StringVar()
        self.year_var = tk.StringVar()

        ttk.Label(left, text="Budget:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.budget_var, width=35).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Popularity:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.pop_var, width=35).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Release year:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.year_var, width=35).pack(anchor="w", pady=(0, 10))

        self.k_var = tk.StringVar(value="5")
        self.weighted_var = tk.BooleanVar(value=True)

        row_k = ttk.Frame(left)
        row_k.pack(anchor="w", pady=(0, 8))
        ttk.Label(row_k, text="k:").pack(side="left")
        ttk.Entry(row_k, textvariable=self.k_var, width=6).pack(side="left", padx=(6, 0))

        ttk.Checkbutton(left, text="Weighted KNN", variable=self.weighted_var)\
            .pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Dự đoán rating", command=self.on_predict).pack(anchor="w", fill="x")

        self.result_lbl = ttk.Label(left, text="Rating dự đoán: -", font=("Segoe UI", 13, "bold"))
        self.result_lbl.pack(anchor="w", pady=(14, 0))

        right = ttk.Frame(self, padding=12)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(right, text="Chọn thể loại", font=("Segoe UI", 14, "bold"))\
            .pack(anchor="w", pady=(0, 10))

        sf = ScrollableFrame(right)
        sf.pack(fill="both", expand=True)


        cols_per_row = 4
        for idx, gc in enumerate(sorted(self.genre_cols)):
            name = gc[len("genre_"):] if gc.lower().startswith("genre_") else gc

            btn = tk.Button(
                sf.inner,
                text=name,
                relief="raised",
                bd=2,
                padx=10,
                pady=8,
                command=lambda col=gc: self.toggle_genre(col),
                wraplength=160
            )
            r = idx // cols_per_row
            c = idx % cols_per_row
            btn.grid(row=r, column=c, sticky="ew", padx=6, pady=6)
            sf.inner.grid_columnconfigure(c, weight=1)
            self.genre_buttons[gc] = btn

    def toggle_genre(self, genre_col):
        if genre_col in self.selected:
            self.selected.remove(genre_col)
            self._set_button_state(self.genre_buttons[genre_col], selected=False)
        else:
            self.selected.add(genre_col)
            self._set_button_state(self.genre_buttons[genre_col], selected=True)

    def _set_button_state(self, btn: tk.Button, selected: bool):

        if selected:
            btn.config(relief="sunken")
        else:
            btn.config(relief="raised")

    def on_predict(self):

        budget = to_float(self.budget_var.get())
        popularity = to_float(self.pop_var.get())
        year = to_float(self.year_var.get())

        try:
            k = int(self.k_var.get().strip() or "5")
            if k <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Input lỗi", "k phải là số nguyên dương.")
            return

        weighted = bool(self.weighted_var.get())

        try:
            xq = build_query_vector_from_raw_inputs(
                self.feature_cols, self.genre_cols, self.scaler,
                budget, popularity, year,
                self.selected
            )
            pred = knn_predict_from_knn_csv(
                KNN_PATH, self.feature_cols, self.target_col,
                xq, k=k, weighted=weighted
            )
            self.result_lbl.config(text=f"Rating dự đoán: {pred:.4f}")
        except Exception as e:
            messagebox.showerror("Lỗi dự đoán", str(e))


if __name__ == "__main__":
    app = KNNApp()
    app.mainloop()
