# ==============================
# apputil.py
# ==============================
import pickle
import pandas as pd
import numpy as np

# Load models and roast mapping dictionary
with open('model_1.pickle', 'rb') as f:
    model_1 = pickle.load(f)

with open('model_2.pickle', 'rb') as f:
    model_2 = pickle.load(f)

with open('roast_cat.pickle', 'rb') as f:
    roast_cat = pickle.load(f)

def predict_rating(df_X: pd.DataFrame):
    """
    Predict coffee ratings based on 100g_USD and roast.
    - Uses model_2 (Decision Tree) when roast is known.
    - Falls back to model_1 (Linear Regression) when roast is unknown.
    """
    preds = []

    for _, row in df_X.iterrows():
        usd = row['100g_USD']
        roast = row['roast']

        if roast in roast_cat:
            roast_num = roast_cat[roast]
            pred = model_2.predict([[usd, roast_num]])[0]
        else:
            pred = model_1.predict([[usd]])[0]

        preds.append(pred)

    return np.array(preds)