import pandas as pd
import matplotlib.pyplot as plt


# Fetch Train Data
df = pd.read_csv("../data_file/train.csv")

# 1. Create child-only dataframe
children = df[df["Age"] < 16].copy()
adult = df[df["Age"] >= 16].copy() # Similarly for adults

# 2. Create custom age groups
# Child Group
bins_child = [0, 2, 3, 5, 7, 10, 13, 16]
labels_child = ["0-2", "2-3", "3-5", "5-7", "7-10", "10-13", "13-16"]

children["Child_AgeGroup"] = pd.cut(
    children["Age"],
    bins=bins_child,
    labels=labels_child,
    right=False
)
# Adult Group
bins_adult = [16, 19, 21, 25, 30, 35, 40, 45, 50, 60,90]
labels_adult = ["16-19", "19-21", "21-25", "25-30","30-35", "35-40", "40-45", "45-50","50-60", "60+"]

adult["Adult_AgeGroup"] = pd.cut(
    adult["Age"],
    bins=bins_adult,
    labels=labels_adult,
    right=False
)

# Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

#groupby - groups dataframe as "SibSp:0 SibSp:1 .. etc"
#.agg() generates summary statistics
def plot_count_survived_dead(data, column, compare_by = None):
    # Optional comparision with Sex parameter feature
    if compare_by:
        group_columns = [column, compare_by]
    else:
        group_columns = column

    summary = data.groupby(group_columns, observed=False)["Survived"].agg(
        Count="count",
        Survived="sum" #since [ dead = 0 + live = 1 ] = Survived
    )

    summary["Dead"] = summary["Count"] - summary["Survived"]

    summary[["Count", "Survived", "Dead"]].plot(kind="bar")

    plt.title(f"Count, Survived, and Dead by {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Passengers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_count_survived_dead(adult, "Adult_AgeGroup")
plot_count_survived_dead(children, "Child_AgeGroup", compare_by="Sex") # With optional Sex comparision
plot_count_survived_dead(df, "FamilySize")

import numpy as np

from sklearn.tree import DecisionTreeClassifier


# Load data and remove rows where Age is missing for bin calculation

df_clean = df.dropna(subset=["Age"]).copy()


# --- SEX-SPECIFIC CHILD + ADULT TREE BINNING ---

def extract_tree_cuts(tree):
    cuts = tree.tree_.threshold[tree.tree_.threshold != -2]
    cuts = sorted(np.round(cuts, 1))
    return cuts


def create_age_bins_for_sex(df_clean, sex, child_age_limit=16):
    sex_df = df_clean[df_clean["Sex"] == sex].copy()

    child_df = sex_df[sex_df["Age"] <= child_age_limit].copy()
    adult_df = sex_df[sex_df["Age"] > child_age_limit].copy()

    # --- CHILDREN SECTION ---
    # Smaller min_samples_leaf allows finer child bins.
    child_tree = DecisionTreeClassifier(
        max_leaf_nodes=12,
        min_samples_leaf=3,
        random_state=42
    )

    child_tree.fit(child_df[["Age"]], child_df["Survived"])
    child_cuts = extract_tree_cuts(child_tree)

    # --- ADULT SECTION ---
    # Larger min_samples_leaf encourages broader adult bins.
    adult_tree = DecisionTreeClassifier(
        max_leaf_nodes=10,
        min_samples_leaf=20,
        random_state=42
    )

    adult_tree.fit(adult_df[["Age"]], adult_df["Survived"])
    adult_cuts = extract_tree_cuts(adult_tree)

    max_age = float(np.ceil(sex_df["Age"].max()))

    final_bins = sorted(
        list(set(
            [0.0] +
            child_cuts +
            [float(child_age_limit)] +
            adult_cuts +
            [max_age + 1]
        ))
    )

    return final_bins, child_tree, adult_tree


female_bins, female_child_tree, female_adult_tree = create_age_bins_for_sex(
    df_clean,
    sex="female"
)

male_bins, male_child_tree, male_adult_tree = create_age_bins_for_sex(
    df_clean,
    sex="male"
)

sex_age_bins = {
    "female": female_bins,
    "male": male_bins
}

print("Female Statistical Age Bin Edges:")
print(female_bins)

print("\nMale Statistical Age Bin Edges:")
print(male_bins)






