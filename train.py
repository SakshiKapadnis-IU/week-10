# ==============================
# train.py
# ==============================
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle

# -------------------------------
# Step 1: Load dataset
# -------------------------------
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# -------------------------------
# Step 2: Train model_1 (Linear Regression using only 100g_USD)
# -------------------------------
X1 = df[['100g_USD']]
y = df['rating']

model_1 = LinearRegression()
model_1.fit(X1, y)

with open('model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)

print("✅ model_1.pickle saved successfully.")

# -------------------------------
# Step 3: Train model_2 (Decision Tree using 100g_USD + roast)
# -------------------------------
# Create roast mapping dictionary
roast_cat = {roast: i for i, roast in enumerate(df['roast'].unique())}

# Map categorical roast values to numeric labels
df['roast_num'] = df['roast'].map(roast_cat)

X2 = df[['100g_USD', 'roast_num']]

model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X2, y)

# -------------------------------
# Step 4: Save both model_2 and roast_cat dictionary
# -------------------------------
with open('model_2.pickle', 'wb') as f:
    pickle.dump(model_2, f)

with open('roast_cat.pickle', 'wb') as f:
    pickle.dump(roast_cat, f)

print("✅ model_2.pickle and roast_cat.pickle saved successfully.")
