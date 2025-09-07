import sys
import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load("artifacts/model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

# Column order (13 features)
cols = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

# Get input values from command line
# Example usage:
# python predict.py 63 1 3 145 233 1 0 150 0 2.3 0 0 1
vals = list(map(float, sys.argv[1:]))
if len(vals) != len(cols):
    raise ValueError(f"Expected {len(cols)} values, got {len(vals)}")

row = pd.DataFrame([vals], columns=cols)

# Scale + predict
row_scaled = scaler.transform(row)
pred = int(model.predict(row_scaled)[0])
proba = float(model.predict_proba(row_scaled)[0][1])

print({"prediction": pred, "probability_of_1": proba})
