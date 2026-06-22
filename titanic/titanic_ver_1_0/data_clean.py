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

# Create Base Core Train Parameter CSV [ Pclass + Sex + Age + Deck + Dependents + Family Id + Ticket]
Core_para_df = clean_file_df[["PassengerId", "Pclass", "Name", "Sex", "Age", "Ticket"]]
Core_para_df["Deck"] = ""
Core_para_df["Deck_Source"] = ""
Core_para_df["FamilySize"] = ""
Core_para_df["FamilyId"] = ""


# Extract title for age/family/social category analysis
# Regex breakdown:
# ,        -> find comma after surname
# \s*      -> allow zero or more spaces after comma
# ([^\.]+) -> capture all characters until the period
# \.       -> stop at the title period
Core_para_df["Title"] = clean_file_df["Name"].str.extract(r",\s*([^\.]+)\.")

# Extract Surname for Family Linking
# Regex breakdown:
# ^       -> start from beginning of string
# [^,]+   -> capture one or more characters that are not comma
Core_para_df["Surname"] = clean_file_df["Name"].str.extract(r"^([^,]+)") # Core file
super_file_df["Surname"] = super_file_df["Name"].str.extract(r"^([^,]+)") # super file

# Family Size
Core_para_df["FamilySize"] = clean_file_df["SibSp"] + clean_file_df["Parch"] + 1
super_file_df["FamilySize"] = super_file_df["SibSp"] + super_file_df["Parch"] + 1

# Deck Prediction

## Known Deck Level 1
Core_para_df["Deck"] = clean_file_df["Cabin"].str[0]
### Mark source 1 only where Deck is known from actual Cabin
known_deck = Core_para_df["Deck"].notna()
Core_para_df.loc[known_deck, "Deck_Source"] = "1"

## Prediction using same ticket no Deck Level 2
### Create plain deck letter in combined train+test dataframe
super_file_df["Deck_Letter"] = super_file_df["Cabin"].str[0]

### Build Ticket -> Deck_Letter map from all known cabins in train+test
ticket_deck_map = (
    super_file_df
    .dropna(subset=["Deck_Letter"]) # Drop Deck_Letter rows without value
    .groupby("Ticket")["Deck_Letter"]
    .agg(lambda x: x.mode()[0]) # For each Ticket group, choose most common known deck letter
)

### Find train rows where Deck is still missing
missing_deck = Core_para_df["Deck"].isna()

### Infer deck from ticket map
ticket_inferred_deck = Core_para_df["Ticket"].map(ticket_deck_map)

### Fill only missing decks where ticket inference exists
level_2_deck = missing_deck & ticket_inferred_deck.notna()

Core_para_df.loc[level_2_deck, "Deck"] = ticket_inferred_deck.loc[level_2_deck]
Core_para_df.loc[level_2_deck, "Deck_Source"] = "2"



# Save Core Para CSV
Core_para_df.to_csv(f"modified_source/Core_{file_input}_para.csv", index=False)
print(Core_para_df.head())

