import pandas as pd
import matplotlib.pyplot as plt


# Fetch Train Data
df = pd.read_csv("../data_file/train.csv")

# 1. Create child-only dataframe
children = df[df["Age"] < 16].copy()

# 2. Create custom age groups
bins_child = [0, 2, 3, 5, 7, 10, 13, 16]
labels_child = ["0-2", "2-3", "3-5", "5-7", "7-10", "10-13", "13-16"]

children["Child_AgeGroup"] = pd.cut(
    children["Age"],
    bins=bins_child,
    labels=labels_child,
    right=False
)

#groupby - groups dataframe as "SibSp:0 SibSp:1 .. etc"
#.agg() generates summary statistics
def plot_count_survived_dead(data, column):
    summary = data.groupby(column)["Survived"].agg(
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

plot_count_survived_dead(children, "Child_AgeGroup")
plot_count_survived_dead(df, "SibSp")
plot_count_survived_dead(df, "Parch")
plot_count_survived_dead(df, "Sex")
