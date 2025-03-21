import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Display first few rows
print(df.head())


# Count phishing vs. legitimate emails
print(df['label'].value_counts())
