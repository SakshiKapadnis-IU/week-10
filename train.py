# train.py
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_1_PATH = BASE_DIR / "model_1.pickle"
MODEL_2_PATH = BASE_DIR / "model_2.pickle"
CSV_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

def main():
    # Load dataset
    df = pd.read_csv(CSV_URL)

    # Keep only rows with needed columns
    df = df.dropna(subset=["100g_USD", "rating", "roast"])

    # ---------- Exercise 1: Linear Regression ----------
    X1 = df[["100g_USD"]].astype(float)  # ensure float
    y = df["rating"].astype(float)
    model_1 = LinearRegression()
    model_1.fit(X1, y)

    with open(MODEL_1_PATH, "wb") as f:
        pickle.dump(model_1, f)
    print(f"✅ Saved {MODEL_1_PATH.name}")

    # ---------- Exercise 2: Decision Tree Regressor ----------
    # Normalize roast
    df["roast_norm"] = df["roast"].astype(str).str.strip().str.title()
    unique_roasts = df["roast_norm"].unique()
    roast_map = {r: i for i, r in enumerate(unique_roasts)}

    df["roast_num"] = df["roast_norm"].map(roast_map)
    X2 = df[["100g_USD", "roast_num"]].astype(float)
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(X2, y)

    # Save as dict with model + roast_map
    with open(MODEL_2_PATH, "wb") as f:
        pickle.dump({"model": model_2, "roast_map": roast_map}, f)
    print(f"✅ Saved {MODEL_2_PATH.name}")

if __name__ == "__main__":
    main()