import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import sys
from Linear_Regression import LinearRegressionGD

# Inject class vào __main__ (vì khi chạy GUI, __main__ chính là tinker.py)
setattr(sys.modules["__main__"], "LinearRegressionGD", LinearRegressionGD)
# import các hàm từ code_full.py
from code_full import load_all_artifacts, build_one_row, predict_df

def run_gui():
    artifacts = load_all_artifacts()
    std_params = artifacts["std_params"]
    binary_cols = std_params.get("binary_cols", [])

    root = tk.Tk()
    root.title("Movie Rating Predictor (Stacking Ridge)")

    frm = ttk.Frame(root, padding=12)
    frm.grid()

    # inputs
    ttk.Label(frm, text="budget").grid(column=0, row=0, sticky="w")
    e_budget = ttk.Entry(frm, width=25)
    e_budget.grid(column=1, row=0)

    ttk.Label(frm, text="popularity").grid(column=0, row=1, sticky="w")
    e_pop = ttk.Entry(frm, width=25)
    e_pop.grid(column=1, row=1)

    ttk.Label(frm, text="release_year").grid(column=0, row=2, sticky="w")
    e_year = ttk.Entry(frm, width=25)
    e_year.grid(column=1, row=2)

    # genres (checkbox)
    ttk.Label(frm, text="genres").grid(column=0, row=3, sticky="nw")
    genre_frame = ttk.Frame(frm)
    genre_frame.grid(column=1, row=3, sticky="w")

    vars_genre = {}
    # hiển thị 20 genre thành 2 cột cho gọn
    for i, col in enumerate(binary_cols):
        gname = col.replace("genre_", "")
        v = tk.IntVar(value=0)
        vars_genre[gname] = v
        cb = ttk.Checkbutton(genre_frame, text=gname, variable=v)
        cb.grid(column=i % 2, row=i // 2, sticky="w", padx=4)

    lbl_out = ttk.Label(frm, text="rating_pred = ?", font=("Segoe UI", 12, "bold"))
    lbl_out.grid(column=0, row=4, columnspan=2, pady=(10, 0))

    def on_predict():
        try:
            budget = float(e_budget.get().strip())
            pop = float(e_pop.get().strip())
            year = float(e_year.get().strip())
            genres = [g for g, v in vars_genre.items() if v.get() == 1]

            df_one = build_one_row(budget, pop, year, genres, std_params)
            y = predict_df(df_one, artifacts)[0]
            lbl_out.config(text=f"rating_pred = {y:.4f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    btn = ttk.Button(frm, text="Predict", command=on_predict)
    btn.grid(column=0, row=5, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
