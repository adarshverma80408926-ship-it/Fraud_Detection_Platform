import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/raw/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
print(df["Class"].value_counts())

print("\n" + "=" * 60)
print("CLASS PERCENTAGE")
print("=" * 60)
print(df["Class"].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe().T)

# Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class")
plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("reports/class_distribution.png", dpi=300)
plt.close()

# Transaction amount distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Amount", bins=100)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/amount_distribution.png", dpi=300)
plt.close()

print("\nEDA completed successfully.")
