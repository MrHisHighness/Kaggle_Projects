import pandas as pd

file_input = "train" # Change Train/Test Here
# Fetch Train/Test Data
clean_file_df = pd.read_csv(f"../data_file/{file_input}.csv")


# For opposite Test/Train Fetch
file_opp = ""
if file_input == "train":
    file_opp = "test"
elif file_input == "test":
    file_opp = "train" 
clean_opp_df = pd.read_csv(f"../data_file/{file_opp}.csv")


# Fetch Super File with Both Train and Test data for Family Linking
super_file_df = pd.concat([clean_file_df, clean_opp_df], ignore_index=True)
super_file_df.to_csv("modified_source/super_file.csv", index=False)

# Create Base Core Train Parameter CSV [ Pclass + Sex + Age + Deck + Dependents + Family Id]
Core_para_df = clean_file_df[["PassengerId", "Pclass", "Name", "Sex", "Age"]]
Core_para_df["Deck"] = ""
Core_para_df["FamilySize"] = ""
Core_para_df["FamilyId"] = ""



# Deck Prediction
## Known Deck
Core_para_df["Deck"] = clean_file_df["Cabin"].str[0] + "1"



# Family Size
Core_para_df["FamilySize"] = clean_file_df["SibSp"] + clean_file_df["Parch"]



Core_para_df.to_csv(f"modified_source/Core_{file_input}_para.csv", index=False)
print(Core_para_df.head())

