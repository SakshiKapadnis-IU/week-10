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
    df = pd.read_csv(CSV_URL)

    # keep only rows with required columns
    df = df.dropna(subset=["100g_USD", "rating", "roast"]).copy()

    # ensure roast strings are normalized (matches apputil.title-cased lookup)
    df["roast_norm"] = df["roast"].astype(str).str.strip().str.title()

    # ---------- Model 1: LinearRegression on 100g_USD ----------
    X1 = df[["100g_USD"]]
    y = df["rating"]
    model_1 = LinearRegression()
    model_1.fit(X1, y)

    with open(MODEL_1_PATH, "wb") as f:
        pickle.dump(model_1, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved {MODEL_1_PATH.name}")

    # ---------- Model 2: DecisionTreeRegressor on 100g_USD + roast ----------
    # Build roast_map with Title-cased keys to match apputil normalization
    unique_roasts = df["roast_norm"].unique()
    roast_map = {str(r).strip().title(): int(i) for i, r in enumerate(unique_roasts)}

    # map roast to numeric codes
    df["roast_num"] = df["roast_norm"].map(roast_map)

    X2 = df[["100g_USD", "roast_num"]]
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(X2, y)

    with open(MODEL_2_PATH, "wb") as f:
        pickle.dump({"model": model_2, "roast_map": roast_map}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved {MODEL_2_PATH.name} (contains model + roast_map)")


if __name__ == "__main__":
    main()