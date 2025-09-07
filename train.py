import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

# 1) Load dataset
df = pd.read_csv("data/heart.csv")

# 2) Features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# 3) Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# 6) Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc:.3f}")

# 7) Save model + scaler
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(scaler, "artifacts/scaler.joblib")
print("✅ Saved model + scaler in artifacts/")
