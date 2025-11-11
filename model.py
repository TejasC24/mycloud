# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# === CONFIG ===
DATA_PATH = "House_price.csv"             
PIPELINE_PATH = "house_price_pipeline.pkl"

# === 1. Load dataset ===
df = pd.read_csv(DATA_PATH)

# Quick cleaning: drop rows with critical nulls (adjust if needed)
df = df.dropna(subset=["total_sqft", "bath", "balcony", "price", "area_type", "availability", "location", "size"])

# Convert numeric columns to numeric types (coerce just in case)
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")
df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Drop rows that became NaN
df = df.dropna(subset=["total_sqft", "bath", "balcony", "price"])

# === 2. Define features and target ===
FEATURES = ["area_type", "availability", "location", "size", "society", "total_sqft", "bath", "balcony"]
TARGET = "price"

X = df[FEATURES]
y = df[TARGET]

# === 3. Preprocessing ===
categorical_cols = ["area_type", "availability", "location", "size", "society"]
numeric_cols = ["total_sqft", "bath", "balcony"]

categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols),
    ],
    remainder="drop",
)

# === 4. Pipeline and training ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
pipeline.fit(X_train, y_train)
print("Done. Saving pipeline to", PIPELINE_PATH)

joblib.dump(pipeline, PIPELINE_PATH)
print("Saved.")
