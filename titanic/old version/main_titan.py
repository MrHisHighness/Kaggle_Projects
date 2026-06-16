import numpy as np # linear algebra
import pandas as pd
# Load the dataset
train = pd.read_csv("train.csv")
import matplotlib.pyplot as plt
import seaborn as sns




plt.figure(figsize=(8,6))
sns.catplot(x="Pclass", y="Survived", hue="Sex", kind="bar", data=train, height=5, aspect=1.2)
plt.title("Average Survival Rate by Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()
