import pandas as pd


pd.set_option("display.max_columns", None) # Show all columns when printing a DataFrame.
pd.set_option("display.width", 200) # Use up to 200 characters of horizontal space before wrapping the display.


file_input = "test" # Change Train/Test Here
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


## Prediction using ticket and sequential surname-ticket logic Deck Level 2/3

# Direct deck from actual Cabin only
super_file_df["Deck_Letter_Known"] = super_file_df["Cabin"].str[0]

# -----------------------
# Deck Level 2: exact same ticket
# -----------------------

ticket_deck_map = (
    super_file_df
    .dropna(subset=["Deck_Letter_Known"])
    .groupby("Ticket")["Deck_Letter_Known"]
    .agg(lambda x: x.mode()[0])
)

# Broadcast Level 2 deck knowledge back into super_file_df.
# This gives Level 3 access to both direct cabin decks and exact-ticket inferred decks.
super_file_df["Deck_Letter_L2"] = super_file_df["Deck_Letter_Known"].fillna(
    super_file_df["Ticket"].map(ticket_deck_map)
)

missing_deck = Core_para_df["Deck"].isna()
ticket_inferred_deck = Core_para_df["Ticket"].map(ticket_deck_map)

level_2_deck = missing_deck & ticket_inferred_deck.notna()

Core_para_df.loc[level_2_deck, "Deck"] = ticket_inferred_deck.loc[level_2_deck]
Core_para_df.loc[level_2_deck, "Deck_Source"] = "2"






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

# Deck Prediction Level 3
# Structural deck inference using Embarked + Pclass + FareBand
# Only fills when the group has enough known deck evidence and a clear dominant deck.

deck_level_3_keys = ["Embarked", "Pclass", "FareBand"]

deck_level_3_stats = (
    super_file_df
    .dropna(subset=["Deck_Letter_L2", "Embarked", "Pclass", "FareBand"])
    .groupby(deck_level_3_keys)["Deck_Letter_L2"]
    .agg(
        Known_Deck_Count="count",
        Unique_Known_Decks="nunique",
        Proxy_Deck=lambda x: x.mode()[0],
        Dominant_Deck_Share=lambda x: x.value_counts(normalize=True).iloc[0]
    )
)

valid_level_3_groups = (
    deck_level_3_stats["Known_Deck_Count"].ge(5)
    & deck_level_3_stats["Dominant_Deck_Share"].ge(0.65)
)

deck_level_3_map = deck_level_3_stats.loc[
    valid_level_3_groups,
    "Proxy_Deck"
]

level_3_lookup_index = pd.MultiIndex.from_frame(
    Core_para_df[deck_level_3_keys]
)

level_3_inferred_deck = pd.Series(
    level_3_lookup_index.map(deck_level_3_map),
    index=Core_para_df.index
)

level_3_deck = (
    Core_para_df["Deck"].isna()
    & level_3_inferred_deck.notna()
)

Core_para_df.loc[level_3_deck, "Deck"] = level_3_inferred_deck.loc[level_3_deck]
Core_para_df.loc[level_3_deck, "Deck_Source"] = "3"

# Deck Prediction Level 4
# Broader structural fallback using controlled hierarchy.
# This is weaker than Level 3, so each fallback still needs enough evidence.

deck_fallback_levels = [
    (["Embarked", "Pclass"], "4"),
    (["Pclass", "FareBand"], "4"),
]

for deck_fallback_keys, deck_source_value in deck_fallback_levels:
    missing_deck = Core_para_df["Deck"].isna()

    if not missing_deck.any():
        break

    deck_fallback_stats = (
        super_file_df
        .dropna(subset=["Deck_Letter_L2"] + deck_fallback_keys)
        .groupby(deck_fallback_keys)["Deck_Letter_L2"]
        .agg(
            Known_Deck_Count="count",
            Proxy_Deck=lambda x: x.mode()[0],
            Dominant_Deck_Share=lambda x: x.value_counts(normalize=True).iloc[0]
        )
    )

    valid_fallback_groups = (
        deck_fallback_stats["Known_Deck_Count"].ge(5)
        & deck_fallback_stats["Dominant_Deck_Share"].ge(0.50)
    )

    deck_fallback_map = deck_fallback_stats.loc[
        valid_fallback_groups,
        "Proxy_Deck"
    ]

    fallback_lookup_index = pd.MultiIndex.from_frame(
        Core_para_df[deck_fallback_keys]
    )

    fallback_inferred_deck = pd.Series(
        fallback_lookup_index.map(deck_fallback_map),
        index=Core_para_df.index
    )

    fallback_deck = missing_deck & fallback_inferred_deck.notna()

    Core_para_df.loc[fallback_deck, "Deck"] = fallback_inferred_deck.loc[fallback_deck]
    Core_para_df.loc[fallback_deck, "Deck_Source"] = deck_source_value


# Final fallback Level 5A: Pclass + FareBand
still_missing = Core_para_df["Deck"].isna()

if still_missing.any():
    pclass_fareband_fallback_map = (
        super_file_df
        .dropna(subset=["Deck_Letter_L2", "Pclass", "FareBand"])
        .groupby(["Pclass", "FareBand"])["Deck_Letter_L2"]
        .agg(lambda x: x.mode()[0])
    )

    level_5a_lookup_index = pd.MultiIndex.from_frame(
        Core_para_df.loc[still_missing, ["Pclass", "FareBand"]]
    )

    level_5a_deck = pd.Series(
        level_5a_lookup_index.map(pclass_fareband_fallback_map),
        index=Core_para_df.loc[still_missing].index
    )

    fill_level_5a = still_missing & level_5a_deck.notna()

    Core_para_df.loc[fill_level_5a, "Deck"] = level_5a_deck.loc[fill_level_5a]
    Core_para_df.loc[fill_level_5a, "Deck_Source"] = "5"


# Emergency fallback Level 6: Pclass-only
still_missing = Core_para_df["Deck"].isna()

if still_missing.any():
    pclass_fallback_map = (
        super_file_df
        .dropna(subset=["Deck_Letter_L2"])
        .groupby("Pclass")["Deck_Letter_L2"]
        .agg(lambda x: x.mode()[0])
    )

    Core_para_df.loc[still_missing, "Deck"] = (
        Core_para_df.loc[still_missing, "Pclass"].map(pclass_fallback_map)
    )
    Core_para_df.loc[still_missing, "Deck_Source"] = "6"

    # Rare deck cleanup
# Deck T appears only once, so merge it into A instead of keeping a one-row category.
Core_para_df["Deck"] = Core_para_df["Deck"].replace("T", "A")


# Save Core Para CSV
print(Core_para_df.columns)
print(Core_para_df.head())
Core_para_df.to_csv(f"modified_source/Core_{file_input}_para.csv", index=False)


