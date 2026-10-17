
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn.csv")

# Target and features
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify feature types
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

# Split before fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["No Churn", "Churn"],
))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

# Save trained pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.joblib")

print("\nModel saved to models/churn_model.joblib")
