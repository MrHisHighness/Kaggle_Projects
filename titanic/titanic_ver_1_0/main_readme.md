## Project Process Summary

- I started this Titanic project as a beginner Kaggle problem, but it gradually became a feature-engineering and validation exercise.

- My main goal was not only to submit predictions, but to understand how raw passenger information could be converted into meaningful survival signals.

- I began with exploratory analysis of the original columns:
  - `Sex`
  - `Age`
  - `Pclass`
  - `Fare`
  - `Cabin`
  - `Ticket`
  - `SibSp`
  - `Parch`
  - `Name`

- Instead of directly feeding all raw columns into the model, I studied what each column could imply about survival.

- One of the first major observations was that age did not behave in a simple linear way.

- Child survival changed across smaller age intervals, and the pattern also differed between male and female passengers.

- However, some child and age-specific groups had small sample sizes, so I had to balance two competing goals:
  - capturing distinct survival patterns that looked important
  - avoiding overly fragile bins based on too few passengers

- This led to the creation of `Sex_Age_Group`.

- `Sex_Age_Group` was designed to preserve detailed age-survival patterns separately for male and female passengers, especially where survival behavior changed sharply.

- To reduce purely manual bias, I used decision-tree-based survival splits to generate separate male and female age bins.

- I also created `Age_Role` as a companion feature to `Sex_Age_Group`.

- While `Sex_Age_Group` captured statistical age-survival patterns, `Age_Role` tried to capture the passenger's likely family or social role.

- This was based on the idea that passengers with similar ages could still have very different survival contexts depending on whether they were likely travelling as:
  - a child with parents
  - a wife
  - a husband or father
  - a bachelor
  - an adult travelling alone
  - an older family member

- This mattered because Titanic survival was not only affected by raw age. It was also affected by social priority, family responsibility, gender expectations, and whether a passenger was part of a family unit.

- Some of these role groups had small sample sizes, but they carried distinct survival information. I used `Age_Role` to preserve this contextual signal without relying only on raw age or narrow age bins.

- In simple terms:
  - `Sex_Age_Group` captured sex-specific statistical age patterns.
  - `Age_Role` captured likely family/social role.
  - Together, they represented both age-based survival behavior and social survival context.

- The second major direction was family linking.

- The original `SibSp` and `Parch` columns only gave partial family information, so I built a `FamilyId` system using:
  - ticket number
  - surname
  - family size
  - passenger class
  - embarkation
  - nearby ticket numbers

- I then used `FamilyId` as a hidden linking key to create:
  - `Family_survive`
  - `Family_dead`
  - `Family_known_count`

- For train rows, I removed each passenger's own survival result from the family survival/death counts to avoid target leakage.

- I also worked on cabin and deck information.

- Since many `Cabin` values were missing, I extracted known decks where available and inferred missing decks using:
  - same ticket groups
  - family links
  - broader passenger patterns

- I created `Deck_Source` to track whether a deck value was directly known or inferred.

- For fare, I avoided relying only on raw fare because ticket fare could represent either an individual passenger or a shared group ticket.

- I converted fare into `FareBand` to capture fare-level survival patterns more cleanly.

- After feature engineering, I tested different model and feature combinations, including:
  - SVC
  - XGBoost
  - HistGradientBoosting
  - ensemble-style approaches

- The best result came from an RBF-kernel SVC using a focused final feature set rather than every feature I created.

- The final model used:
  - `Sex_Age_Group`
  - `FareBand`
  - `Deck`
  - `Deck_Source`
  - `Age_Role`
  - `Pclass`
  - `FamilySize`
  - `Family_survive`
  - `Family_dead`
  - `Family_known_count`

- Some features were useful for tracing and debugging, but were removed from the final model when they did not improve validation or Kaggle performance.

- My best Kaggle score for this version was `0.79186`.

- I decided to stop at this version because the project had reached its main goal: completing a full ML workflow from EDA, feature creation, leakage checks, model testing, and final submission.

- The biggest lesson was that meaningful feature engineering and careful validation mattered more than simply using a more complex model.