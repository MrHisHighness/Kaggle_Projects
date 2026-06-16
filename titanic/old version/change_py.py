import pandas as pd
import numpy as np


# Load your dataset
df = pd.read_csv("train.csv")
df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
df['FarePerPerson_log'] = np.log1p(df['FarePerPerson'])
def classwise_band(x):
    # qcut can fail if too few unique values; duplicates='drop' avoids errors
    return pd.qcut(x, 4, labels=False, duplicates='drop')

df['FareClassBand'] = df.groupby('Pclass')['FarePerPerson'] \
                        .transform(classwise_band)
df['FareMedianByClass'] = df.groupby('Pclass')['FarePerPerson'].transform('median')
df['FareSurplus'] = df['FarePerPerson'] - df['FareMedianByClass']
df['FareSurplusRatio'] = df['FareSurplus'] / df['FareMedianByClass']

# 9. Cabin interaction: does a cabin exist, and combine with fare
df['CabinExists'] = df['Cabin'].notna().astype(int)
df['FareCabinInteraction'] = df['FarePerPerson_log'] * df['CabinExists']
fare_features = df[[
    'PassengerId',
    'Survived',          
    'Pclass',
    'Sex',
    'Fare',
    'TicketGroupSize',
    'FarePerPerson',
    'FarePerPerson_log',
    'FareClassBand',
    'FareMedianByClass',
    'FareSurplus',
    'CabinExists',
    'FareCabinInteraction'
]]
# Extract Title
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Normalize titles
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Lady': 'Royalty',
    'Countess': 'Royalty',
    'Dona': 'Royalty',
    'Sir': 'Royalty',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Capt': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Rev': 'Officer',
    'Dr': 'Officer'
})

# Median age per title
title_median = df.groupby('Title')['Age'].median()

# Fill missing ages
df['Age'] = df.apply(
    lambda row: title_median[row['Title']] if pd.isnull(row['Age']) else row['Age'],
    axis=1
)
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['AgeBand'] = pd.cut(
    df['Age'],
    bins=[0, 5, 12, 18, 35, 55, 80],
    labels=[0, 1, 2, 3, 4, 5]
)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsChild'] = (df['AgeBand'] <= 1).astype(int)
df["FareSurplusRatio"]= df['FareSurplus'] / df['FareMedianByClass']

print("Age imputed successfully!")
# Create new dataframe
new_df = pd.DataFrame()
new_df["Survived"] = df["Survived"]
new_df["PassengerId"] = df["PassengerId"]
new_df["AgeBand"] = df["AgeBand"]
new_df["IsChild"] = df["IsChild"]
new_df["FamilySize"] = df["FamilySize"]
new_df["FareCabinInteraction"] = df["FareCabinInteraction"]
new_df["GenderEncoded"] = df["Sex"].apply(lambda x: 1 if x == "female" else 0)
new_df["Pclass"] = df["Pclass"]
new_df['FareSurplus'] =  df['FareSurplus'] 
new_df["FareSurplusRatio"] = df["FareSurplusRatio"]

new_df["Embarked"] = df["Embarked"]
# Save to new CSV
new_df.to_csv("gender_encoded.csv", index=False)
fare_features.to_csv("titanic_fare_features.csv", index=False)
print("titanic_fare_features.csv created with mixed Fare features.")
print("gender_encoded.csv created successfully!")
