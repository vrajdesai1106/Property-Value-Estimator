# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import PassiveAggressiveRegressor

df = pd.read_csv("Dataset\\USA Housing Dataset.csv")

# Keep only numeric features
X = df.select_dtypes(include=["int64", "float64"]).drop("price", axis=1)
y = df["price"]

# Train-test split (optional, mainly for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = PassiveAggressiveRegressor(random_state=42)
best_model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model trained and saved as model.pkl")
