# Phạm Lê Anh Duy - 20162012

import os
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

def find_data_csv(filename="data_raw.csv") -> Path:
    here = Path(__file__).resolve()

    for p in [here.parent] + list(here.parents):
        cand = p / "data" / filename
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Không tìm thấy data/{filename} từ vị trí: {here}")

class Linear_Regression:
    def __init__(self, learning_rate=0.01, n_iterations=10_000, use_standardize=True,
                 early_stopping=False, tol=1e-4, patience=10):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.use_standardize = bool(use_standardize)

        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience

        self.weights = None
        self.bias = None
        self.x_mean_ = None
        self.x_std_ = None
        self.history_ = {"loss": [], "iterations": 0}

    def _fit_scaler(self, X):
        self.x_mean_ = np.mean(X, axis=0)
        self.x_std_ = np.std(X, axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0

    def _transform(self, X):
        if not self.use_standardize:
            return X
        if self.x_mean_ is None or self.x_std_ is None:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        return (X - self.x_mean_) / self.x_std_

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        if self.use_standardize:
            self._fit_scaler(X)
            X_train = self._transform(X)
        else:
            X_train = X

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        self.history_ = {"loss": [], "iterations": 0}
        best_loss = float("inf")
        patience_counter = 0

        for i in range(self.n_iterations):
            y_pred = X_train @ self.weights + self.bias
            error = y_pred - y

            current_loss = float(np.mean(error ** 2))
            self.history_["loss"].append(current_loss)

            dw = (2.0 / n_samples) * (X_train.T @ error)
            db = (2.0 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.early_stopping:
                if best_loss - current_loss > self.tol:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.history_["iterations"] = i + 1
                        break
        else:
            self.history_["iterations"] = self.n_iterations

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xp = self._transform(X) if self.use_standardize else X
        return Xp @ self.weights + self.bias

    def evaluate(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return float(1.0 - u / v) if v != 0 else 0.0


class App(tk.Tk):
    def __init__(self, csv_path: str):
        super().__init__()
        self.title("Dự đoán rating phim - Linear Regression")
        self.geometry("980x420")
        self.resizable(False, False)

        self.csv_path = str(csv_path)

        self.data = None
        self.model = None
        self.feature_cols = []
        self.base_cols = []
        self.genre_cols = []
        self.feature_median = {}

        self.genre_state = {}    
        self.genre_buttons = {}   

        self.status_var = tk.StringVar(value="Đang load dữ liệu & train model...")
        self.pred_var = tk.StringVar(value="Rating dự đoán: --")

        self._build_ui()
        self._train_async()

    def _build_ui(self):
        pad = 12

        top = ttk.Frame(self)
        top.pack(fill="x", padx=pad, pady=(pad, 6))

        ttk.Label(top, text="Nhập thong tin phim", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(top, text="Chon the loai", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(220, 0))
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", padx=pad)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=pad, pady=(8, 8))

        # Left panel
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Tên phim:").grid(row=0, column=0, sticky="w", pady=6)
        self.name_entry = ttk.Entry(left, width=26)
        self.name_entry.grid(row=0, column=1, sticky="w", pady=6)

        ttk.Label(left, text="Budget:").grid(row=1, column=0, sticky="w", pady=6)
        self.budget_entry = ttk.Entry(left, width=26)
        self.budget_entry.grid(row=1, column=1, sticky="w", pady=6)

        ttk.Label(left, text="Popularity:").grid(row=2, column=0, sticky="w", pady=6)
        self.pop_entry = ttk.Entry(left, width=26)
        self.pop_entry.grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(left, text="Release year:").grid(row=3, column=0, sticky="w", pady=6)
        self.year_entry = ttk.Entry(left, width=26)
        self.year_entry.grid(row=3, column=1, sticky="w", pady=6)

        self.predict_btn = ttk.Button(left, text="Dự đoán rating", command=self.on_predict, state="disabled")
        self.predict_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 6))

        self.clear_btn = ttk.Button(left, text="Clear", command=self.on_clear, state="disabled")
        self.clear_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(40, 0))

        self.genre_grid = ttk.Frame(right)
        self.genre_grid.pack(anchor="nw")

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=pad, pady=(0, pad))
        ttk.Label(bottom, textvariable=self.pred_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")

    def _build_genre_buttons(self):

        for w in self.genre_grid.winfo_children():
            w.destroy()

        self.genre_buttons.clear()
        self.genre_state = {gc: 0 for gc in self.genre_cols}

        cols = 4
        btn_w = 14
        btn_h = 2

        for i, gc in enumerate(self.genre_cols):
            r = i // cols
            c = i % cols

            display = gc.replace("genre_", "").replace("_", " ")
            b = tk.Button(
                self.genre_grid,
                text=display,
                width=btn_w,
                height=btn_h,
                relief="raised",
                command=lambda g=gc: self._toggle_genre(g)
            )
            b.grid(row=r, column=c, padx=8, pady=6, sticky="w")

            self.genre_buttons[gc] = b

    def _toggle_genre(self, genre_col: str):
        cur = self.genre_state.get(genre_col, 0)
        new = 0 if cur == 1 else 1
        self.genre_state[genre_col] = new

        b = self.genre_buttons[genre_col]
        b.config(relief="sunken" if new == 1 else "raised")

    def _train_async(self):
        t = threading.Thread(target=self._train_model, daemon=True)
        t.start()

    def _train_model(self):
        try:
            df = pd.read_csv(self.csv_path)

            drop_ids = {"userId", "movieId"}  
            target = "rating"
            feature_cols = [c for c in df.columns if c != target and c not in drop_ids]

            df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
            df[target] = pd.to_numeric(df[target], errors="coerce")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna(subset=[target]).copy()
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

            X = df[feature_cols].to_numpy(dtype=float)
            y = df[target].to_numpy(dtype=float)

            model = Linear_Regression(
                learning_rate=0.1,
                n_iterations=10000,
                use_standardize=True,
                early_stopping=True,
                tol=1e-8,
                patience=200
            )
            model.fit(X, y)

            self.data = df
            self.model = model
            self.feature_cols = feature_cols
            self.genre_cols = [c for c in feature_cols if c.startswith("genre_")]
            self.base_cols = [c for c in feature_cols if not c.startswith("genre_")]
            self.feature_median = df[feature_cols].median(numeric_only=True).to_dict()

            self.after(0, self._on_train_done)

        except Exception as e:
            self.after(0, self._on_train_fail, e)

    def _on_train_done(self):

        if "budget" in self.base_cols:
            self.budget_entry.insert(0, str(self.feature_median.get("budget", 0)))
        else:
            self.budget_entry.insert(0, "0")
            self.budget_entry.configure(state="disabled")

        if "popularity" in self.base_cols:
            self.pop_entry.insert(0, str(self.feature_median.get("popularity", 0)))
        else:
            self.pop_entry.insert(0, "0")
            self.pop_entry.configure(state="disabled")

        if "release_year" in self.base_cols:
            self.year_entry.insert(0, str(int(self.feature_median.get("release_year", 2000))))
        else:
            self.year_entry.insert(0, "2000")
            self.year_entry.configure(state="disabled")

        self._build_genre_buttons()

        mse = self.model.evaluate(self.data[self.feature_cols].to_numpy(float), self.data["rating"].to_numpy(float))
        r2 = self.model.score(self.data[self.feature_cols].to_numpy(float), self.data["rating"].to_numpy(float))
        iters = self.model.history_.get("iterations", 0)

        self.status_var.set(f"samples={len(self.data)} | features={len(self.feature_cols)} | iters={iters} | MSE={mse:.4f} | R²={r2:.4f}")
        self.predict_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")

    def _on_train_fail(self, e: Exception):
        self.status_var.set("Train thất bại.")
        messagebox.showerror("Error", f"Không train được model:\n{e}")

    def _read_float(self, entry: ttk.Entry, col: str):
        s = entry.get().strip()
        if s == "":
            return float(self.feature_median.get(col, 0))
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Giá trị '{col}' không hợp lệ: {s}")

    def _read_int(self, entry: ttk.Entry, col: str):
        s = entry.get().strip()
        if s == "":
            return int(self.feature_median.get(col, 2000))
        try:
            return int(float(s))
        except ValueError:
            raise ValueError(f"Giá trị '{col}' không hợp lệ: {s}")

    def on_predict(self):
        if self.model is None:
            return
        try:
            row = {}

            # base cols
            for c in self.base_cols:
                if c == "budget":
                    row[c] = self._read_float(self.budget_entry, "budget")
                elif c == "popularity":
                    row[c] = self._read_float(self.pop_entry, "popularity")
                elif c == "release_year":
                    row[c] = self._read_int(self.year_entry, "release_year")
                else:
                    row[c] = float(self.feature_median.get(c, 0))

            for gc in self.genre_cols:
                row[gc] = int(self.genre_state.get(gc, 0))

            x_input = np.array([[row[c] for c in self.feature_cols]], dtype=float)

            pred = float(self.model.predict(x_input)[0])
            pred_clip = float(np.clip(pred, 0.5, 5.0))

            self.pred_var.set(f"Rating dự đoán: {pred_clip:.4f}")

        except Exception as e:
            messagebox.showwarning("Input error", str(e))

    def on_clear(self):
        self.name_entry.delete(0, "end")

        if self.budget_entry.cget("state") != "disabled":
            self.budget_entry.delete(0, "end")
            self.budget_entry.insert(0, str(self.feature_median.get("budget", 0)))

        if self.pop_entry.cget("state") != "disabled":
            self.pop_entry.delete(0, "end")
            self.pop_entry.insert(0, str(self.feature_median.get("popularity", 0)))

        if self.year_entry.cget("state") != "disabled":
            self.year_entry.delete(0, "end")
            self.year_entry.insert(0, str(int(self.feature_median.get("release_year", 2000))))

        for gc in self.genre_cols:
            self.genre_state[gc] = 0
            self.genre_buttons[gc].config(relief="raised")

        self.pred_var.set("Rating dự đoán: --")


if __name__ == "__main__":
    csv_path = find_data_csv("data_raw.csv")

    app = App(csv_path)
    app.mainloop()
