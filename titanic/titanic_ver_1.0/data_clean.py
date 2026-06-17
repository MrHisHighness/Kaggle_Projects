import pandas as pd

# Fetch Train Data
train_df = pd.read_csv("../data_file/train.csv")


# Create Survive (Train) + Prediction CSV
survived_train_df = train_df[["PassengerId", "Survived"]].copy()
survived_train_df["Prediction"] = ""
survived_train_df.to_csv("modified_train/survived_train.csv", index=False)


# Create train file without Survive data
source_without_survive_df = train_df.drop(columns=["Survived"])
source_without_survive_df.to_csv("modified_train/train_without_survived.csv", index=False)

print("Created survived_predictions.csv successfully")
print(survived_train_df.head())

print("\nSource file:")
print(source_without_survive_df.head())