import pandas as pd

# Load your training data
df = pd.read_csv("train.csv")   # change path if needed

# Count missing values per column
missing_values = df.isnull().sum()

# Count filled (non-missing) values per column
filled_values = df.notnull().sum()

# Combine into one table
summary = pd.DataFrame({
    "Filled Values": filled_values,
    "Missing Values": missing_values
})

print(summary)
