
import pandas as pd

input_path = "data/raw/Telco-Customer-Churn.csv"
output_path = "data/processed/cleaned_churn.csv"

# Load original dataset
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Remove leading/trailing spaces from text columns
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
# New customers with zero tenure have no accumulated charges
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isna()),
    "TotalCharges"
] = 0.0

# Check invalid or missing TotalCharges values
print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isna().sum())

print("\nRows with missing TotalCharges:")
print(df[df["TotalCharges"].isna()][
    ["customerID", "tenure", "TotalCharges", "Churn"]
])

# Remove duplicate rows if any
df = df.drop_duplicates()

# Create output directory
import os
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("\nCleaned shape:", df.shape)
print("Saved to:", output_path)
print("\nTotalCharges data type:", df["TotalCharges"].dtype)
