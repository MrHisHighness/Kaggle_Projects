import pandas as pd
import numpy as np
df1 = pd.read_csv("train.csv")
df = pd.read_csv("gender_encoded.csv")

# Build extra features in df1 (just like you did in your main pipeline)
df1['TicketGroupSize'] = df1.groupby('Ticket')['Ticket'].transform('count')
df1['FarePerPerson'] = df1['Fare'] / df1['TicketGroupSize']
df1['FareMedianByClass'] = df1.groupby('Pclass')['FarePerPerson'].transform('median')
df1['FareSurplus'] = df1['FarePerPerson'] - df1['FareMedianByClass']

df1['Embarked'] = df1['Embarked'].fillna('S')
df1['Embarked'] = df1['Embarked'].map({'S':0,'C':1,'Q':2})

# Sex encoded
df1['GenderEncoded'] = df1['Sex'].map({'male':0,'female':1})

# Title
df1['Title'] = df1['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df1['Title'] = df1['Title'].replace({
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Royalty', 'Countess': 'Royalty', 'Dona': 'Royalty',
    'Sir': 'Royalty', 'Jonkheer': 'Royalty', 'Don': 'Royalty',
    'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
    'Rev': 'Officer', 'Dr': 'Officer'
})
df1['TitleCode'] = df1['Title'].astype('category').cat.codes

# Extract deck for known cabins
df1['Deck'] = df1['Cabin'].str[0]
deck_df = df1[df1['Deck'].notnull()].copy()
deck_df['DeckCode'] = deck_df['Deck'].astype('category').cat.codes
def deck_group(d):
    if d in ['A','B','C']:
        return 0   # upper
    elif d in ['D','E']:
        return 1   # middle
    else:
        return 2   # lower

deck_df['DeckGroup'] = deck_df['Deck'].apply(deck_group)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

feature_cols = [
    'Pclass',
    'FareSurplus',
    'FarePerPerson',
    'Embarked',
    'GenderEncoded',
    'TitleCode',
    'TicketGroupSize',
]

X = deck_df[feature_cols]
y = deck_df['DeckCode']
y_group = deck_df['DeckGroup']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_group, test_size=0.2, random_state=42
)

model_group = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42
)
model_group.fit(X_train, y_train)
print("DeckGroup accuracy:", model_group.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
def deck_group(d):
    if d in ['A','B','C']:
        return 0   # upper
    elif d in ['D','E']:
        return 1   # middle
    else:
        return 2   # lower

deck_df['DeckGroup'] = deck_df['Deck'].apply(deck_group)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))


deck_df['DeckGroup'] = deck_df['Deck'].apply(deck_group)
# We already have:
# df1 = train.csv with extra features
# deck_df = df1[df1['Deck'].notnull()]
# model_group = trained RandomForest for DeckGroup
# feature_cols = [...]

# 1. Find rows with missing Deck
missing = df1[df1['Deck'].isnull()].copy()

# 2. Predict DeckGroup for missing rows
missing['DeckGroup'] = model_group.predict(missing[feature_cols])

# 3. Create full DeckGroup column in df1 (known + predicted)
df1['DeckGroup'] = np.nan
df1.loc[deck_df.index, 'DeckGroup'] = deck_df['DeckGroup']      # real groups
df1.loc[missing.index, 'DeckGroup'] = missing['DeckGroup']      # predicted groups
# Keep only the columns we need from df1
deck_info = df1[['PassengerId', 'DeckGroup']]

# Merge into your engineered dataset
df_merged = df.merge(deck_info, on='PassengerId', how='left')

# Save updated file
df_merged.to_csv("gender_encoded_with_deck.csv", index=False)

print(df_merged.head())
