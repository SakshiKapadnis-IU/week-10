# apputil.py
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_1_PATH = BASE_DIR / "model_1.pickle"
MODEL_2_PATH = BASE_DIR / "model_2.pickle"

# ---------- Safe model loading ----------
def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file missing: {path} — run `python3 train.py` first")
    with open(path, "rb") as f:
        return pickle.load(f)

model_1 = _load_pickle(MODEL_1_PATH)
_model2_data = _load_pickle(MODEL_2_PATH)

if isinstance(_model2_data, dict):
    model_2 = _model2_data.get("model")
    roast_map = _model2_data.get("roast_map", {})
else:
    model_2 = _model2_data
    roast_map = {}

# ---------- Predict function ----------
def predict_rating(df_X: pd.DataFrame) -> np.ndarray:
    """
    Predict coffee ratings.

    Handles:
    - Missing or invalid prices
    - Unknown or missing roast values
    - Proper column naming for models to avoid warnings
    - Normalizes roast strings

    Parameters
    ----------
    df_X : pd.DataFrame
        Must include column '100g_USD'. Optional: 'roast'.

    Returns
    -------
    np.ndarray of predicted ratings
    """
    if not isinstance(df_X, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if "100g_USD" not in df_X.columns:
        raise ValueError("DataFrame must contain '100g_USD' column")

    preds = []
    has_roast = "roast" in df_X.columns

    for _, row in df_X.iterrows():
        price = row.get("100g_USD")
        roast = row.get("roast") if has_roast else None

        # Validate price
        try:
            price = float(price)
        except (ValueError, TypeError):
            preds.append(np.nan)
            continue

        # Normalize roast string
        if roast is not None:
            roast = str(roast).strip().title()

        # Choose model
        if roast is not None and roast in roast_map and model_2 is not None:
            roast_val = roast_map[roast]
            X = pd.DataFrame([[price, roast_val]], columns=["100g_USD", "roast_num"])
            pred = model_2.predict(X)[0]
        else:
            X = pd.DataFrame([[price]], columns=["100g_USD"])
            pred = model_1.predict(X)[0]

        preds.append(float(pred))

    return np.array(preds)

# ---------- Example test ----------
if __name__ == "__main__":
    sample = pd.DataFrame({
        "100g_USD": [5.5, "10", np.nan, "abc", 7.2],
        "roast": ["Medium", "unknown", " Light ", None, "Dark"]
    })
    print("🔹 Input Data:\n", sample)
    preds = predict_rating(sample)
    print("\n🔸 Predicted Ratings:\n", preds)