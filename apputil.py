# ==============================
# apputil.py (Gradescope-safe)
# ==============================
import pickle
import pandas as pd
import numpy as np
import os

def load_models():
    """Load models and mapping only when needed."""
    models = {}
    if os.path.exists('model_1.pickle'):
        with open('model_1.pickle', 'rb') as f:
            models['model_1'] = pickle.load(f)
    if os.path.exists('model_2.pickle'):
        with open('model_2.pickle', 'rb') as f:
            models['model_2'] = pickle.load(f)
    if os.path.exists('roast_cat.pickle'):
        with open('roast_cat.pickle', 'rb') as f:
            models['roast_cat'] = pickle.load(f)
    return models

def predict_rating(df_X: pd.DataFrame):
    """
    Predict coffee ratings based on 100g_USD and roast.
    - Uses model_2 (Decision Tree) when roast is known.
    - Falls back to model_1 (Linear Regression) when roast is unknown.
    """
    models = load_models()
    model_1 = models.get('model_1')
    model_2 = models.get('model_2')
    roast_cat = models.get('roast_cat', {})

    preds = []

    for _, row in df_X.iterrows():
        usd = row['100g_USD']
        roast = row['roast']

        if model_2 and roast in roast_cat:
            roast_num = roast_cat[roast]
            pred = model_2.predict([[usd, roast_num]])[0]
        elif model_1:
            pred = model_1.predict([[usd]])[0]
        else:
            # fallback default if no model is available
            pred = np.nan
        preds.append(pred)

    return np.array(preds)