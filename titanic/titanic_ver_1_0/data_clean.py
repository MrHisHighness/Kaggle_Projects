import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
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
Core_para_df = clean_file_df[["PassengerId", "Pclass", "Name", "Sex", "Age", "Ticket", "Fare", "Embarked"]]
Core_para_df["Deck"] = ""
Core_para_df["Deck_Source"] = ""
Core_para_df["FamilySize"] = ""
Core_para_df["FamilyId"] = ""
Core_para_df["FamilyLink_Source"] = ""

# Extract numeric part of ticket
super_file_df["Ticket_Number"] = (
    super_file_df["Ticket"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)


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
super_file_df["FamilyLink_Source"] = pd.NA

## Source 1/2: same Ticket, same Surname
## FamilyId stays common for the whole Ticket+Surname group.
## If the group has only one FamilySize value, source = 1.
## If the group has different FamilySize values, source = 2.
ticket_surname_key = ["Ticket", "Surname"]
ticket_surname_group_size = super_file_df.groupby(ticket_surname_key)["PassengerId"].transform("count") # count no of members sharing ticket and surname
ticket_surname_family_size_count = super_file_df.groupby(ticket_surname_key)["FamilySize"].transform("nunique") # count no of unique family sizes

ticket_surname_family = ticket_surname_group_size > 1
level_1_family = ticket_surname_family & (ticket_surname_family_size_count == 1)
level_2_family = ticket_surname_family & (ticket_surname_family_size_count > 1)

super_file_df.loc[ticket_surname_family, "FamilyId"] = (
    "F_" + super_file_df.loc[ticket_surname_family, ticket_surname_key].astype(str).agg("_".join, axis=1)
)
super_file_df.loc[level_1_family, "FamilyLink_Source"] = "1"
super_file_df.loc[level_2_family, "FamilyLink_Source"] = "2"


## Source 3: same Surname, same Pclass, same Embarked, nearby Ticket number
## This is weaker than Ticket+Surname, so we only use it for still-unlinked passengers
## who have FamilySize > 1 and nearby numeric ticket numbers.

# Extract numeric part from Ticket
super_file_df["Ticket_Number"] = (
    super_file_df["Ticket"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

level_3_family_key = ["Surname", "Pclass", "Embarked"]

level_3_candidates = super_file_df[
    super_file_df["FamilyId"].isna()
    & (super_file_df["FamilySize"] > 1)
    & (super_file_df["Ticket_Number"].notna())
].copy()

level_3_candidates = level_3_candidates.sort_values(
    level_3_family_key + ["Ticket_Number"]
)

level_3_candidates["Prev_Ticket_Diff"] = (
    level_3_candidates
    .groupby(level_3_family_key)["Ticket_Number"]
    .diff()
    .abs()
)

level_3_candidates["Next_Ticket_Diff"] = (
    level_3_candidates
    .groupby(level_3_family_key)["Ticket_Number"]
    .diff(-1)
    .abs()
)

level_3_candidates["Near_Ticket"] = (
    level_3_candidates["Prev_Ticket_Diff"].le(2)
    | level_3_candidates["Next_Ticket_Diff"].le(2)
)

level_3_family = level_3_candidates["Near_Ticket"].reindex(
    super_file_df.index,
    fill_value=False
)

super_file_df.loc[level_3_family, "FamilyId"] = (
    "F3_" +
    super_file_df.loc[level_3_family, level_3_family_key]
    .astype(str)
    .agg("_".join, axis=1)
)

super_file_df.loc[level_3_family, "FamilyLink_Source"] = "3"



## Source 4: no linked family found
## FamilySize == 1 means truly alone.
## FamilySize > 1 means family exists, but exact family members are unknown.
level_4_family = super_file_df["FamilyId"].isna()
super_file_df.loc[level_4_family, "FamilyLink_Source"] = "4"

level_4_alone = level_4_family & (super_file_df["FamilySize"] == 1)
level_4_unknown_family = level_4_family & (super_file_df["FamilySize"] > 1)

super_file_df.loc[level_4_alone, "FamilyId"] = "Alone"
super_file_df.loc[level_4_unknown_family, "FamilyId"] = "Unknown_Family"


## Bring FamilyId and FamilyLink_Source back into Core_para_df
family_link_map = super_file_df.set_index("PassengerId")["FamilyId"]
family_source_map = super_file_df.set_index("PassengerId")["FamilyLink_Source"]

Core_para_df["FamilyId"] = Core_para_df["PassengerId"].map(family_link_map)
Core_para_df["FamilyLink_Source"] = Core_para_df["PassengerId"].map(family_source_map)

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
level_2_deck = missing_deck & ticket_inferred_deck.notna() #intersection of still missing and ticket inferred deck
Core_para_df.loc[level_2_deck, "Deck"] = ticket_inferred_deck.loc[level_2_deck] # where level_2_deck true-> load ticket inferred deck
Core_para_df.loc[level_2_deck, "Deck_Source"] = "2" # Where level 2 intersection true-> load "2" in Deck_Source


# Fare Cleaning + Pclass-specific FareBand

## Use existing Deck_Letter if already created, otherwise create it
if "Deck_Letter" not in super_file_df.columns:
    super_file_df["Deck_Letter"] = super_file_df["Cabin"].str[0]

## FarePerPerson prevents group-ticket fares from overpowering the signal
super_file_df["FarePerPerson"] = super_file_df["Fare"] / super_file_df["FamilySize"]


## Fill missing FarePerPerson from strongest to weakest grouping
fare_fill_levels = [
    (["Embarked", "Pclass", "Deck_Letter"], "2"),
    (["Embarked", "Pclass"], "3"),
    (["Pclass"], "4"),
]

for fare_keys, fare_source in fare_fill_levels:
    missing_fare = super_file_df["FarePerPerson"].isna()

    fare_fill_value = (
        super_file_df
        .groupby(fare_keys)["FarePerPerson"]
        .transform("median")
    )

    fillable_fare = missing_fare & fare_fill_value.notna()

    super_file_df.loc[fillable_fare, "FarePerPerson"] = fare_fill_value.loc[fillable_fare]
    super_file_df.loc[fillable_fare, "Fare_Source"] = fare_source


## Final emergency fallback
missing_fare = super_file_df["FarePerPerson"].isna()
super_file_df.loc[missing_fare, "FarePerPerson"] = super_file_df["FarePerPerson"].median()
super_file_df.loc[missing_fare, "Fare_Source"] = "5"

# Pclass-specific FareBand
# This avoids FareBand simply becoming another copy of Pclass.

def create_pclass_fare_band(fare_series):
    fare_band_code = pd.qcut(
        fare_series,
        q=3,
        labels=False,
        duplicates="drop"
    )

    return fare_band_code.map({
        0: "LowFare",
        1: "MidFare",
        2: "HighFare"
    })


super_file_df["FareBand_Level"] = (
    super_file_df
    .groupby("Pclass", group_keys=False)["FarePerPerson"]
    .apply(create_pclass_fare_band)
)

super_file_df["FareBand"] = (
    "P"
    + super_file_df["Pclass"].astype(str)
    + "_"
    + super_file_df["FareBand_Level"].astype(str)
)
fare_per_person_map = super_file_df.set_index("PassengerId")["FarePerPerson"]
fare_band_map = super_file_df.set_index("PassengerId")["FareBand"]

Core_para_df["FarePerPerson"] = Core_para_df["PassengerId"].map(fare_per_person_map)
Core_para_df["FareBand"] = Core_para_df["PassengerId"].map(fare_band_map)


# Save Core Para CSV
print(Core_para_df.columns)
print(Core_para_df.head())
Core_para_df.to_csv(f"modified_source/Core_{file_input}_para.csv", index=False)


