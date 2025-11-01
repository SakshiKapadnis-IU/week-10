# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_1_PATH = BASE_DIR / "model_1.pickle"
MODEL_2_PATH = BASE_DIR / "model_2.pickle"

# ---------- Load dataset ----------
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)
df = df.dropna(subset=["100g_USD", "rating", "roast"])

# ---------- Exercise 1: Linear Regression ----------
X1 = df[["100g_USD"]].copy()  # Ensure it's a DataFrame
y = df["rating"].copy()

model_1 = LinearRegression()
model_1.fit(X1, y)

# Save model_1
with open(MODEL_1_PATH, "wb") as f:
    pickle.dump(model_1, f)
print(f"✅ Saved {MODEL_1_PATH.name}")

# ---------- Exercise 2: Decision Tree Regressor ----------
# Map roast categories to numbers
roast_map = {cat: i for i, cat in enumerate(sorted(df["roast"].unique()), start=1)}
df["roast_num"] = df["roast"].map(roast_map)

X2 = df[["100g_USD", "roast_num"]].copy()
model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X2, y)

# Save model_2 with roast_map
with open(MODEL_2_PATH, "wb") as f:
    pickle.dump({"model": model_2, "roast_map": roast_map}, f)
print(f"✅ Saved {MODEL_2_PATH.name} with roast mapping")