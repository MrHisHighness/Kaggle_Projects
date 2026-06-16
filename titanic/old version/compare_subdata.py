import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("train.csv")

sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Sex", y="Survived", estimator="mean")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Pclass", y="Survived", estimator="mean")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", estimator="mean")
plt.title("Survival Rate by Class and Sex")
plt.ylabel("Survival Rate")
plt.show()
plt.figure(figsize=(8,4))
sns.kdeplot(data=df[df["Survived"] == 0], x="Age", label="Died", shade=True)
sns.kdeplot(data=df[df["Survived"] == 1], x="Age", label="Survived", shade=True)
plt.title("Age Distribution by Survival")
plt.xlim(0, 80)
plt.legend()
plt.show()
df["Child"] = df["Age"].apply(lambda x: 1 if x < 16 else 0)

plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Child", y="Survived", estimator="mean")
plt.title("Survival Rate: Children (1) vs Adults (0)")
plt.ylabel("Survival Rate")
plt.show()
# Family size = self + siblings/spouse + parents/children
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="IsAlone", y="Survived", estimator="mean")
plt.title("Survival Rate: Alone (1) vs With Family (0)")
plt.ylabel("Survival Rate")
plt.show()
family_survival = df.groupby("FamilySize")["Survived"].mean().reset_index()

plt.figure(figsize=(7,4))
sns.lineplot(data=family_survival, x="FamilySize", y="Survived", marker="o")
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.xticks(family_survival["FamilySize"])
plt.show()

