import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Function to check an email
def check_email(text):
    for _, row in df.iterrows():
        if text in row['email_text']:
            return f"Label: {row['label']}"
    return "Email not found in dataset."

# User input from CMD
email_input = input("Enter email text to check: ")
print(check_email(email_input))
