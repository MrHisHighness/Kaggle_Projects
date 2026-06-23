import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


df = pd.read_csv("modified_source/Core_train_para.csv")

if "Survived" not in df.columns:
    original_train = pd.read_csv("../data_file/train.csv")[["PassengerId", "Survived"]]
    df = df.merge(original_train, on="PassengerId", how="left")


features_core = [
    "Pclass",
    "FareBand",
    "Sex_Age_Group",
    "Deck",
    "Deck_Source",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    "FamilyLink_Source",
    "FamilySize",
    "Age_Role",
]


categorical_features = [
    "FareBand",
    "Sex_Age_Group",
    "Deck",
    "Deck_Source",
    "FamilyLink_Source",
    "Age_Role",
]

numeric_features = [
    "Pclass",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    "FamilySize",
]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ],
    sparse_threshold=0
)


models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=15,
        random_state=42
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=1000
    ),

    "SVC": SVC(
        C=1.0,
        kernel="rbf",
        random_state=42
    ),
}


def add_family_signal(train_part, valid_part):
    train_part = train_part.copy()
    valid_part = valid_part.copy()

    linked_train = train_part[
        ~train_part["FamilyId"].isin(["Alone", "Unknown_Family"])
    ]

    family_survive_map = linked_train.groupby("FamilyId")["Survived"].sum()
    family_known_map = linked_train.groupby("FamilyId")["Survived"].count()
    family_dead_map = family_known_map - family_survive_map

    # Train rows: use family evidence, then remove passenger's own result.
    train_part["Family_survive"] = train_part["FamilyId"].map(family_survive_map).fillna(0)
    train_part["Family_dead"] = train_part["FamilyId"].map(family_dead_map).fillna(0)

    train_part["Family_survive"] = train_part["Family_survive"] - train_part["Survived"]
    train_part["Family_dead"] = train_part["Family_dead"] - (1 - train_part["Survived"])

    train_part["Family_survive"] = train_part["Family_survive"].clip(lower=0)
    train_part["Family_dead"] = train_part["Family_dead"].clip(lower=0)

    # Validation rows: only use family evidence from train_part.
    valid_part["Family_survive"] = valid_part["FamilyId"].map(family_survive_map).fillna(0)
    valid_part["Family_dead"] = valid_part["FamilyId"].map(family_dead_map).fillna(0)

    for part in [train_part, valid_part]:
        no_family_signal = part["FamilyId"].isin(["Alone", "Unknown_Family"])
        part.loc[no_family_signal, ["Family_survive", "Family_dead"]] = 0

        part["Family_known_count"] = (
            part["Family_survive"] + part["Family_dead"]
        )

    return train_part, valid_part


for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scores = []

    print("=" * 60)
    print(model_name)

    for seed in [0, 1, 2, 3, 4, 5, 10, 20, 42, 100]:
        train_part, valid_part = train_test_split(
            df,
            test_size=0.20,
            random_state=seed,
            stratify=df["Survived"]
        )

        train_part, valid_part = add_family_signal(train_part, valid_part)

        X_train = train_part[features_core]
        y_train = train_part["Survived"]

        X_valid = valid_part[features_core]
        y_valid = valid_part["Survived"]

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)

        acc = accuracy_score(y_valid, predictions)
        scores.append(acc)

        print("-" * 60)
        print(f"Seed {seed}: {acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_valid, predictions))
        print("Classification Report:")
        print(classification_report(y_valid, predictions))

    print("-" * 60)
    print("Average accuracy:", sum(scores) / len(scores))
    print("Best accuracy:", max(scores))
    print("Worst accuracy:", min(scores))