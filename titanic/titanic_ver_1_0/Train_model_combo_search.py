import argparse
import itertools
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


SEEDS = [0, 1, 2, 3, 4, 5, 10, 20, 42, 100]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search Titanic feature-group combinations using repeated validation splits."
    )
    parser.add_argument(
        "--core-train",
        default="modified_source/Core_train_para.csv",
        help="Path to your engineered Core_train_para.csv file.",
    )
    parser.add_argument(
        "--original-train",
        default="../data_file/train.csv",
        help="Path to the original Kaggle train.csv file containing Survived.",
    )
    parser.add_argument(
        "--results",
        default="modified_source/feature_combo_results.csv",
        help="Where to save ranked feature/model results.",
    )
    parser.add_argument(
        "--seed-results",
        default="modified_source/feature_combo_seed_scores.csv",
        help="Where to save per-seed validation scores.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use only the faster, usually strongest models for a first pass.",
    )
    return parser.parse_args()


def load_data(core_train_path, original_train_path):
    df = pd.read_csv(core_train_path)

    if "Survived" not in df.columns:
        original_train = pd.read_csv(original_train_path)[["PassengerId", "Survived"]]
        df = df.merge(original_train, on="PassengerId", how="left")

    if df["Survived"].isna().any():
        raise ValueError("Some rows have no Survived value after merging train.csv.")

    return df


def add_family_signal(train_part, valid_part):
    train_part = train_part.copy()
    valid_part = valid_part.copy()

    linked_train = train_part[
        ~train_part["FamilyId"].isin(["Alone", "Unknown_Family"])
    ]

    family_survive_map = linked_train.groupby("FamilyId")["Survived"].sum()
    family_known_map = linked_train.groupby("FamilyId")["Survived"].count()
    family_dead_map = family_known_map - family_survive_map

    train_part["Family_survive"] = train_part["FamilyId"].map(family_survive_map).fillna(0)
    train_part["Family_dead"] = train_part["FamilyId"].map(family_dead_map).fillna(0)

    train_part["Family_survive"] = train_part["Family_survive"] - train_part["Survived"]
    train_part["Family_dead"] = train_part["Family_dead"] - (1 - train_part["Survived"])

    train_part["Family_survive"] = train_part["Family_survive"].clip(lower=0)
    train_part["Family_dead"] = train_part["Family_dead"].clip(lower=0)

    valid_part["Family_survive"] = valid_part["FamilyId"].map(family_survive_map).fillna(0)
    valid_part["Family_dead"] = valid_part["FamilyId"].map(family_dead_map).fillna(0)

    for part in [train_part, valid_part]:
        no_family_signal = part["FamilyId"].isin(["Alone", "Unknown_Family"])
        part.loc[no_family_signal, ["Family_survive", "Family_dead"]] = 0
        part["Family_known_count"] = part["Family_survive"] + part["Family_dead"]

    return train_part, valid_part


def available_feature_groups(df):
    wanted_groups = {
        "base": ["Pclass", "Sex"],
        "age_raw": ["Age", "Age_Source"],
        "age_engineered": ["Sex_Age_Group", "Age_Role"],
        "fare": ["FareBand", "FarePerPerson"],
        "deck": ["Deck", "Deck_Source"],
        "family_basic": ["FamilySize", "FamilyLink_Source"],
        "family_signal": ["Family_survive", "Family_dead", "Family_known_count"],
        "embarked": ["Embarked"],
        "title": ["Title"],
    }

    groups = {}
    for group_name, columns in wanted_groups.items():
        existing_columns = [column for column in columns if column in df.columns]
        if existing_columns:
            groups[group_name] = existing_columns

    return groups


def build_feature_sets(feature_groups):
    fixed_groups = ["base"]
    optional_groups = [group for group in feature_groups if group not in fixed_groups]
    feature_sets = []

    for group_count in range(len(optional_groups) + 1):
        for combo in itertools.combinations(optional_groups, group_count):
            selected_groups = fixed_groups + list(combo)
            selected_features = []
            for group in selected_groups:
                selected_features.extend(feature_groups[group])

            feature_sets.append(
                {
                    "feature_set": " + ".join(selected_groups),
                    "groups": selected_groups,
                    "features": selected_features,
                }
            )

    return feature_sets


from pandas.api.types import is_numeric_dtype

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


def build_pipeline(df, features, model):
    categorical_features, numeric_features = split_feature_types(df, features)

    transformers = []

    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_models(fast=False):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVC": SVC(C=1.0, kernel="rbf", random_state=42),
    }

    if fast:
        return {
            "RandomForest": models["RandomForest"],
            "ExtraTrees": models["ExtraTrees"],
            "GradientBoosting": models["GradientBoosting"],
            "LogisticRegression": models["LogisticRegression"],
        }

    return models


def evaluate_feature_set(df, feature_set, model_name, model):
    seed_rows = []
    scores = []
    last_confusion_matrix = None

    for seed in SEEDS:
        train_part, valid_part = train_test_split(
            df,
            test_size=0.20,
            random_state=seed,
            stratify=df["Survived"],
        )

        if "family_signal" in feature_set["groups"]:
            train_part, valid_part = add_family_signal(train_part, valid_part)

        features = feature_set["features"]
        pipeline = build_pipeline(train_part, features, model)

        X_train = train_part[features]
        y_train = train_part["Survived"]
        X_valid = valid_part[features]
        y_valid = valid_part["Survived"]

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)
        score = accuracy_score(y_valid, predictions)
        scores.append(score)

        last_confusion_matrix = confusion_matrix(y_valid, predictions)
        seed_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set["feature_set"],
                "seed": seed,
                "accuracy": score,
            }
        )

    return {
        "model": model_name,
        "feature_set": feature_set["feature_set"],
        "features": ", ".join(feature_set["features"]),
        "avg_accuracy": sum(scores) / len(scores),
        "best_accuracy": max(scores),
        "worst_accuracy": min(scores),
        "score_range": max(scores) - min(scores),
        "last_confusion_matrix": last_confusion_matrix.tolist(),
    }, seed_rows


def main():
    args = parse_args()
    results_path = Path(args.results)
    seed_results_path = Path(args.seed_results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    seed_results_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.core_train, args.original_train)
    feature_groups = available_feature_groups(df)
    feature_sets = build_feature_sets(feature_groups)
    models = get_models(fast=args.fast)

    all_results = []
    all_seed_rows = []
    total_runs = len(feature_sets) * len(models)
    completed_runs = 0

    print(f"Testing {len(feature_sets)} feature sets x {len(models)} models = {total_runs} runs")
    print(f"Each run uses {len(SEEDS)} validation splits")

    for feature_set in feature_sets:
        for model_name, model in models.items():
            result, seed_rows = evaluate_feature_set(df, feature_set, model_name, model)
            all_results.append(result)
            all_seed_rows.extend(seed_rows)

            completed_runs += 1
            print(
                f"{completed_runs:>4}/{total_runs} | "
                f"{result['avg_accuracy']:.4f} avg | "
                f"{model_name} | {feature_set['feature_set']}"
            )

    results_df = pd.DataFrame(all_results).sort_values(
        ["avg_accuracy", "worst_accuracy", "score_range"],
        ascending=[False, False, True],
    )
    seed_results_df = pd.DataFrame(all_seed_rows)

    results_df.to_csv(results_path, index=False)
    seed_results_df.to_csv(seed_results_path, index=False)

    print("\nTop 20 feature/model combinations:")
    print(
        results_df[
            [
                "model",
                "feature_set",
                "avg_accuracy",
                "best_accuracy",
                "worst_accuracy",
                "score_range",
                "features",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved ranked results to: {results_path}")
    print(f"Saved per-seed scores to: {seed_results_path}")


if __name__ == "__main__":
    main()
