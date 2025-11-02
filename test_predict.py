# ==============================
# test_predict.py
# ==============================
import pandas as pd
from apputil import predict_rating

df_X = pd.DataFrame([
    [10.00, "Dark"],
    [15.00, "Very Light"]
], columns=["100g_USD", "roast"])

y_pred = predict_rating(df_X)
print(y_pred)