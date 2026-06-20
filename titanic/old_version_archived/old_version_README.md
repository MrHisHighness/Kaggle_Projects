# Titanic Old Version Archive

This folder is a dump of older Titanic project experiments. The files are not a clean pipeline yet. They are better treated as an archaeology folder: useful for remembering the ideas tried, but not as the current source of truth.

The main theme of these files is:

- explore survival patterns with basic EDA
- engineer intuitive features from Titanic data
- try to recover cabin/deck information
- train simple ML models on engineered features
- generate a Kaggle submission

## File Map

| File | What it seems to do | Status |
|---|---|---|
| `fetch_data.py` | Prints filled vs missing values for each column in `train.csv`. | Useful first inspection script. |
| `main_titan(1).py` | Plots survival rate by passenger class and gender. | Small EDA script. |
| `compare_subdata.py` | Creates several EDA plots: sex, class, class+sex, age distribution, child/adult, alone/family, family size. | Useful EDA, but should be cleaned into reusable plotting functions. |
| `change_py.py` | Builds engineered training features: ticket group size, fare per person, fare surplus, cabin existence, title-based age filling, age bands, family size, child flag, gender encoding, embarked encoding. Saves `gender_encoded.csv` and `titanic_fare_features.csv`. | Important feature-engineering experiment. |
| `deck_typoe.py` | Trains a Random Forest to predict cabin `DeckGroup` from known cabin rows, then merges predicted deck groups into `gender_encoded.csv`. Saves `gender_encoded_with_deck.csv`. | Important but experimental. Needs validation and cleaner naming. |
| `tarining_pyh(1).py` | Trains Logistic Regression, Random Forest, and Gradient Boosting on `gender_encoded_with_deck.csv`; prints validation accuracy/report. | Model comparison script. Typo in filename. |
| `testrun(1).py` | Trains Gradient Boosting on engineered train data, reproduces features for `test.csv`, predicts deck group, and writes `submission.csv`. | Submission pipeline attempt. |
| `newtestrun.py` | Similar to `testrun(1).py`, but uses XGBoost and a custom probability threshold of `0.46`. | Later/more aggressive submission attempt. |
| `train_mod1.py` | Sketch of a more advanced feature pipeline using surname, title, ticket root, shared ticket, cabin deck, LightGBM, target encoding, and GroupKFold. | Concept sketch, not directly runnable as-is. |

## Rough Timeline

1. Basic inspection started with `fetch_data.py`.
2. EDA scripts explored obvious Titanic survival patterns:
   - gender
   - passenger class
   - age
   - child/adult grouping
   - family size
3. Feature engineering expanded in `change_py.py`:
   - title-based age imputation
   - age bands
   - family size
   - fare per person
   - fare surplus within passenger class
   - cabin-existence interaction
4. Cabin/deck inference was attempted in `deck_typoe.py`.
5. Models were compared in `tarining_pyh(1).py`.
6. Submission scripts were built in `testrun(1).py` and `newtestrun.py`.
7. `train_mod1.py` looks like a later idea for a cleaner, more advanced ML pipeline.



### Cabin/deck idea

The old code tries to infer missing cabin/deck information using:

- `Pclass`
- `FareSurplus`
- `FarePerPerson`
- `Embarked`
- `GenderEncoded`
- `TitleCode`
- `TicketGroupSize`

This is an interesting idea, but it must be handled carefully because the known-cabin passengers are not a random sample. Cabin data is heavily missing, so predicted deck may partly encode class/fare privilege rather than true physical location.

### Fare features

`FarePerPerson` and `FareSurplus` are better than raw `Fare` because Titanic fares are affected by shared tickets and class. The old code was trying to ask:

> Did this passenger pay more or less than typical for their class?

That is a useful feature, especially if fare is being used as a proxy for cabin quality or social status.







