
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/processed/cleaned_churn.csv")

os.makedirs("reports/figures", exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Churn distribution
print("CHURN COUNTS")
print(df["Churn"].value_counts())
print("\nCHURN PERCENTAGE")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.savefig("reports/figures/churn_distribution.png", dpi=150)
plt.close()

# 2. Churn by contract type
print("\nCHURN BY CONTRACT")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index").round(3))

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_contract.png", dpi=150)
plt.close()

# 3. Tenure by churn
print("\nAVERAGE TENURE BY CHURN")
print(df.groupby("Churn")["tenure"].mean().round(2))

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure by Churn")
plt.savefig("reports/figures/tenure_by_churn.png", dpi=150)
plt.close()

# 4. Monthly charges by churn
print("\nAVERAGE MONTHLY CHARGES BY CHURN")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.savefig("reports/figures/monthly_charges_by_churn.png", dpi=150)
plt.close()

print("\nEDA completed! Charts saved in reports/figures/")
