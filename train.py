# train.py
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Paths for pickle files
BASE_DIR = Path(__file__).resolve().parent
MODEL_1_PATH = BASE_DIR / "model_1.pickle"
MODEL_2_PATH = BASE_DIR / "model_2.pickle"

# Dataset URL
CSV_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

def main():
    # Load data
    df = pd.read_csv(CSV_URL)

    # Drop rows missing required columns
    df = df.dropna(subset=["100g_USD", "rating", "roast"]).copy()

    # ---------- Exercise 1: LinearRegression on 100g_USD ----------
    X1 = df[["100g_USD"]].astype(float)
    y = df["rating"].astype(float)

    model_1 = LinearRegression()
    model_1.fit(X1, y)

filename = 'model_1.pickle'
try:
    with open(filename, 'wb') as f:
        pickle.dump(model_1, f)
    print(f"Model successfully saved to {filename}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
    


    with open(MODEL_1_PATH, "wb") as f:
        pickle.dump(model_1, f)
    print(f"✅ Saved {MODEL_1_PATH.name}")

    # ---------- Exercise 2: DecisionTreeRegressor on 100g_USD + roast ----------
    # Normalize roast strings
    df["roast_norm"] = df["roast"].astype(str).str.strip().str.title()
    unique_roasts = df["roast_norm"].unique()

    # Create mapping from roast text to numeric code
    roast_map = {r: i for i, r in enumerate(unique_roasts)}

    df["roast_num"] = df["roast_norm"].map(roast_map)

    X2 = df[["100g_USD", "roast_num"]].astype(float)
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(X2, y)

    # Save DecisionTree as dict with model + roast_map
    with open(MODEL_2_PATH, "wb") as f:
        pickle.dump({"model": model_2, "roast_map": roast_map}, f)
    print(f"✅ Saved {MODEL_2_PATH.name} (contains model + roast_map)")

if __name__ == "__main__":
    main()