
print("Customer Churn inspection script started!")

import pandas as pd

file_path = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts(dropna=False))
