import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load Core Train & Test Para files
train_df = pd.read_csv("modified_source/Core_train_para.csv")
test_df = pd.read_csv("modified_source/Core_test_para.csv")

# Add Survived in Core Para train file as it does not already contain it
if "Survived" not in train_df.columns:
    original_train = pd.read_csv("../data_file/train.csv")[["PassengerId", "Survived"]]
    train_df = train_df.merge(original_train, on="PassengerId", how="left")

# Select Core Features
features_core = [
    "Pclass",
    "FareBand",
    "Sex_Age_Group",
    "Deck",
    "Deck_Source",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    #"FamilyLink_Source",
    "FamilySize",
    "Age_Role",
]


categorical_features = [
    "FareBand",
    "Sex_Age_Group",
    "Deck",
    "Deck_Source",
    #"FamilyLink_Source",
    "Age_Role",
]

numeric_features = [
    "Pclass",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    "FamilySize",
]


def add_final_family_signal(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    linked_train = train_df[
        ~train_df["FamilyId"].isin(["Alone", "Unknown_Family"])
    ]

    family_survive_map = linked_train.groupby("FamilyId")["Survived"].sum()
    family_known_map = linked_train.groupby("FamilyId")["Survived"].count()
    family_dead_map = family_known_map - family_survive_map

    # Train rows: remove passenger's own survival result
    train_df["Family_survive"] = train_df["FamilyId"].map(family_survive_map).fillna(0)
    train_df["Family_dead"] = train_df["FamilyId"].map(family_dead_map).fillna(0)

    train_df["Family_survive"] = train_df["Family_survive"] - train_df["Survived"]
    train_df["Family_dead"] = train_df["Family_dead"] - (1 - train_df["Survived"])

    train_df["Family_survive"] = train_df["Family_survive"].clip(lower=0)
    train_df["Family_dead"] = train_df["Family_dead"].clip(lower=0)

    # Test rows: use full train family evidence
    test_df["Family_survive"] = test_df["FamilyId"].map(family_survive_map).fillna(0)
    test_df["Family_dead"] = test_df["FamilyId"].map(family_dead_map).fillna(0)

    for part in [train_df, test_df]:
        no_family_signal = part["FamilyId"].isin(["Alone", "Unknown_Family"])
        part.loc[no_family_signal, ["Family_survive", "Family_dead"]] = 0

        part["Family_known_count"] = (
            part["Family_survive"] + part["Family_dead"]
        )

    return train_df, test_df


train_df, test_df = add_final_family_signal(train_df, test_df)

X_train = train_df[features_core]
y_train = train_df["Survived"]

X_test = test_df[features_core]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ],
    sparse_threshold=0
)


model = SVC(
    C=1.0,
    kernel="rbf",
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


pipeline.fit(X_train, y_train)

test_predictions = pipeline.predict(X_test)


submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions.astype(int)
})

submission.to_csv("submission/submission.csv", index=False)

print("Created submission.csv successfully")
print(submission.head())