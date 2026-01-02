import csv
import os
import random

INPUT_PATH = os.path.join("data", "data_KNN_new.csv")          # file 100k dòng
OUTPUT_PATH = os.path.join("data", "data_KNN_10k.csv")     # file mới 10k dòng

N_ROWS = 10_000   # số dòng muốn lấy (không tính header)
SEED = 42

def sample_csv_reservoir(input_path, output_path, n_rows, seed=42):
    rng = random.Random(seed)

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        reservoir = []
        total = 0

        for row in reader:
            total += 1
            if len(reservoir) < n_rows:
                reservoir.append(row)
            else:
                j = rng.randrange(total)
                if j < n_rows:
                    reservoir[j] = row

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo)
        writer.writerow(header)
        writer.writerows(reservoir)

    print(f"Total rows in input (excluding header): {total}")
    print(f"Sampled rows: {len(reservoir)}")
    print(f"Saved -> {output_path}")

if __name__ == "__main__":
    sample_csv_reservoir(INPUT_PATH, OUTPUT_PATH, N_ROWS, seed=SEED)
