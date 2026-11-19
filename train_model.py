
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# 1. Load dataset
data = pd.read_csv("Salary_Data.csv")

print("Dataset loaded successfully!")
print(data.head())

# 2. Remove rows with missing salary
data = data.dropna(subset=["Salary"])

# Convert salary into numeric format
data["Salary"] = pd.to_numeric(
    data["Salary"], errors="coerce"
)
data = data.dropna(subset=["Salary"])

# 3. Select input and output
features = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]

X = data[features]
y = data["Salary"]

# 4. Separate numerical and categorical columns
numeric_columns = [
    "Age",
    "Years of Experience"
]

categorical_columns = [
    "Gender",
    "Education Level",
    "Job Title"
]

# 5. Preprocessing
numeric_transformer = SimpleImputer(
    strategy="median"
)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_columns),
    ("cat", categorical_transformer, categorical_columns)
])

# 6. Create machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    ))
])

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 9. Make predictions
predictions = model.predict(X_test)

# 10. Evaluate model
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("\nModel Evaluation")
print("R2 Score:", round(r2, 3))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# 11. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/salary_model.joblib")

print("\nModel saved successfully!")
