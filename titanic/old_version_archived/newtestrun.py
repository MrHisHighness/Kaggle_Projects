import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. Load engineered TRAIN data and train survival model (XGBoost)
# ============================================================
train = pd.read_csv("gender_encoded_with_deck.csv")

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]

survival_model = XGBClassifier(
    n_estimators=700,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    gamma=0.5,
    min_child_weight=2,
    random_state=42,
    objective="binary:logistic"
)

survival_model.fit(X_train, y_train)

# We'll reuse these exact feature columns for test:
train_feature_cols = X_train.columns.tolist()

# ============================================================
# 2. Train DeckGroup model on RAW train.csv
# ============================================================
df1 = pd.read_csv("train.csv")

# Ticket & fare features for deck model
df1["TicketGroupSize"] = df1.groupby("Ticket")["Ticket"].transform("count")
df1["FarePerPerson"] = df1["Fare"] / df1["TicketGroupSize"]
df1["FareMedianByClass"] = df1.groupby("Pclass")["FarePerPerson"].transform("median")
df1["FareSurplus"] = df1["FarePerPerson"] - df1["FareMedianByClass"]

df1["Embarked"] = df1["Embarked"].fillna("S")
df1["Embarked"] = df1["Embarked"].map({"S": 0, "C": 1, "Q": 2})

df1["GenderEncoded"] = df1["Sex"].map({"male": 0, "female": 1})

# Title for deck model
df1["Title"] = df1["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df1["Title"] = df1["Title"].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Royalty", "Countess": "Royalty", "Dona": "Royalty",
    "Sir": "Royalty", "Jonkheer": "Royalty", "Don": "Royalty",
    "Capt": "Officer", "Col": "Officer", "Major": "Officer",
    "Rev": "Officer", "Dr": "Officer"
})
df1["TitleCode"] = df1["Title"].astype("category").cat.codes

# Extract Deck and DeckGroup
df1["Deck"] = df1["Cabin"].str[0]
deck_df = df1[df1["Deck"].notnull()].copy()

def deck_group(d):
    if d in ["A", "B", "C"]:
        return 0   # upper
    elif d in ["D", "E"]:
        return 1   # middle
    else:
        return 2   # lower

deck_df["DeckGroup"] = deck_df["Deck"].apply(deck_group)

# Train deck model
deck_features = [
    "Pclass",
    "FareSurplus",
    "FarePerPerson",
    "Embarked",
    "GenderEncoded",
    "TitleCode",
    "TicketGroupSize"
]

deck_X = deck_df[deck_features]
deck_y = deck_df["DeckGroup"]

deck_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42
)
deck_model.fit(deck_X, deck_y)

# ============================================================
# 3. Load TEST and apply SAME feature logic
# ============================================================
test = pd.read_csv("test.csv")

# Ticket & fare
test["TicketGroupSize"] = test.groupby("Ticket")["Ticket"].transform("count")
test["FarePerPerson"] = test["Fare"] / test["TicketGroupSize"]
test["FarePerPerson"] = test["FarePerPerson"].fillna(test["FarePerPerson"].median())

# Embarked
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Gender
test["GenderEncoded"] = test["Sex"].map({"male": 0, "female": 1})

# Title (for Age and Deck)
test["Title"] = test["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
test["Title"] = test["Title"].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Royalty", "Countess": "Royalty", "Dona": "Royalty",
    "Sir": "Royalty", "Jonkheer": "Royalty", "Don": "Royalty",
    "Capt": "Officer", "Col": "Officer", "Major": "Officer",
    "Rev": "Officer", "Dr": "Officer"
})
test["TitleCode"] = test["Title"].astype("category").cat.codes

# ---- AgeBand using Title-based age medians from RAW train ----
title_age_medians = df1.groupby("Title")["Age"].median()

def impute_age(row):
    if pd.notnull(row["Age"]):
        return row["Age"]
    return title_age_medians.get(row["Title"], title_age_medians.median())

test["Age"] = test.apply(impute_age, axis=1)

# AgeBand same bins as train script
test["AgeBand"] = pd.cut(
    test["Age"],
    bins=[0, 5, 12, 18, 35, 55, 80],
    labels=[0, 1, 2, 3, 4, 5]
).astype(float).fillna(3)

# Family features
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
test["IsChild"] = (test["AgeBand"] <= 1).astype(int)

# Fare class features
test["FareMedianByClass"] = test.groupby("Pclass")["FarePerPerson"].transform("median")
test["FareSurplus"] = test["FarePerPerson"] - test["FareMedianByClass"]
test["FareSurplusRatio"] = test["FareSurplus"] / test["FareMedianByClass"]

# CabinExists & FareCabinInteraction
test["CabinExists"] = test["Cabin"].notna().astype(int)
test["FarePerPerson_log"] = np.log1p(test["FarePerPerson"])
test["FareCabinInteraction"] = test["FarePerPerson_log"] * test["CabinExists"]

# ---- Predict DeckGroup for test ----
test["DeckGroup"] = deck_model.predict(test[deck_features])

# ============================================================
# 4. Build final feature matrix for survival model
# ============================================================
X_test_final = test[train_feature_cols]

# ============================================================
# 5. Predict Survived with threshold and build submission
# ============================================================

probs_test = survival_model.predict_proba(X_test_final)[:, 1]
THRESH = 0.46
test["Survived"] = (probs_test >= THRESH).astype(int)



submission = test[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)

print("submission.csv created!")
