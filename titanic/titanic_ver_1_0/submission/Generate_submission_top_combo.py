import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOP_AVG_FEATURES = [
    "Pclass",
    "Sex",
    "Sex_Age_Group",
    "Age_Role",
    "FareBand",
    "FarePerPerson",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    "Title",
]

STABLE_FEATURES = [
    "Pclass",
    "Sex",
    "Sex_Age_Group",
    "Age_Role",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    "Embarked",
    "Title",
]

STABLE_DECK_FEATURES = [
    "Pclass",
    "Sex",
    "Sex_Age_Group",
    "Age_Role",
    "Deck",
    "Deck_Source",
    "Family_survive",
    "Family_dead",
    "Family_known_count",
    "Embarked",
    "Title",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Kaggle Titanic submission from the best validation feature combo."
    )
    parser.add_argument(
        "--core-train",
        default="modified_source/Core_train_para.csv",
        help="Path to engineered train CSV.",
    )
    parser.add_argument(
        "--core-test",
        default="modified_source/Core_test_para.csv",
        help="Path to engineered test CSV.",
    )
    parser.add_argument(
        "--original-train",
        default="../data_file/train.csv",
        help="Path to original Kaggle train.csv containing Survived.",
    )
    parser.add_argument(
        "--output",
        default="modified_source/submission_top_combo.csv",
        help="Output path for Kaggle submission CSV.",
    )
    parser.add_argument(
        "--method",
        choices=["top_avg", "stable", "stable_deck"],
        default="top_avg",
        help=(
            "top_avg uses GradientBoosting best average score. "
            "stable uses HistGradientBoosting best worst-case score. "
            "stable_deck adds Deck and Deck_Source to the stable method."
        ),
    )
    return parser.parse_args()


def load_data(core_train_path, core_test_path, original_train_path):
    train_df = pd.read_csv(core_train_path)
    test_df = pd.read_csv(core_test_path)

    if "Survived" not in train_df.columns:
        original_train = pd.read_csv(original_train_path)[["PassengerId", "Survived"]]
        train_df = train_df.merge(original_train, on="PassengerId", how="left")

    if train_df["Survived"].isna().any():
        raise ValueError("Some train rows have no Survived value after merging original train.csv.")

    return train_df, test_df


def add_full_train_family_signal(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    linked_train = train_df[
        ~train_df["FamilyId"].isin(["Alone", "Unknown_Family"])
    ]

    family_survive_map = linked_train.groupby("FamilyId")["Survived"].sum()
    family_known_map = linked_train.groupby("FamilyId")["Survived"].count()
    family_dead_map = family_known_map - family_survive_map

    # Train rows can use family evidence, but must remove the passenger's own target.
    train_df["Family_survive"] = train_df["FamilyId"].map(family_survive_map).fillna(0)
    train_df["Family_dead"] = train_df["FamilyId"].map(family_dead_map).fillna(0)

    train_df["Family_survive"] = train_df["Family_survive"] - train_df["Survived"]
    train_df["Family_dead"] = train_df["Family_dead"] - (1 - train_df["Survived"])

    train_df["Family_survive"] = train_df["Family_survive"].clip(lower=0)
    train_df["Family_dead"] = train_df["Family_dead"].clip(lower=0)

    # Test rows can use all known train family evidence.
    test_df["Family_survive"] = test_df["FamilyId"].map(family_survive_map).fillna(0)
    test_df["Family_dead"] = test_df["FamilyId"].map(family_dead_map).fillna(0)

    for part in [train_df, test_df]:
        no_family_signal = part["FamilyId"].isin(["Alone", "Unknown_Family"])
        part.loc[no_family_signal, ["Family_survive", "Family_dead"]] = 0
        part["Family_known_count"] = part["Family_survive"] + part["Family_dead"]

    return train_df, test_df


def split_feature_types(df, features):
    numeric_features = [
        feature
        for feature in features
        if is_numeric_dtype(df[feature])
    ]
    categorical_features = [
        feature
        for feature in features
        if feature not in numeric_features
    ]

    return categorical_features, numeric_features


def build_pipeline(train_df, features, method):
    categorical_features, numeric_features = split_feature_types(train_df, features)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0)

    if method in ["stable", "stable_deck"]:
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            random_state=42,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(
        args.core_train,
        args.core_test,
        args.original_train,
    )
    train_df, test_df = add_full_train_family_signal(train_df, test_df)

    if args.method == "stable":
        features = STABLE_FEATURES
    elif args.method == "stable_deck":
        features = STABLE_DECK_FEATURES
    else:
        features = TOP_AVG_FEATURES
    missing_train_features = [feature for feature in features if feature not in train_df.columns]
    missing_test_features = [feature for feature in features if feature not in test_df.columns]

    if missing_train_features or missing_test_features:
        raise ValueError(
            f"Missing train features: {missing_train_features}; "
            f"missing test features: {missing_test_features}"
        )

    pipeline = build_pipeline(train_df, features, args.method)

    X_train = train_df[features]
    y_train = train_df["Survived"]
    X_test = test_df[features]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test).astype(int)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions,
        }
    )
    submission.to_csv(output_path, index=False)

    print(f"Method: {args.method}")
    print(f"Features: {features}")
    print(f"Saved submission: {output_path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
