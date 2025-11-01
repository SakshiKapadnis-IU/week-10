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
    # ---------- Load dataset ----------
    df = pd.read_csv(CSV_URL)
    df = df.dropna(subset=["100g_USD", "rating", "roast"]).copy()
    
    # Normalize roast strings (Title case) for consistency
    df["roast_norm"] = df["roast]()_]()