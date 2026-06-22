Source |	Logic |	Meaning
1 |	`Same Ticket + same Surname + consistent FamilySize` |	Strong family link
2 |	`Same Ticket + same Surname, but inconsistent FamilySize` |	Strong travel/family link, but lower confidence
3 |	`Same Surname + same Pclass + same Embarked + nearby numeric ticket number` |	Inferred family link
4 |	`No confident link` | found	Alone or unresolved family

## Family Linking

The Titanic dataset provides `SibSp` and `Parch`, which describe how many siblings/spouses and parents/children a passenger had onboard. I used these columns to create a basic family-size feature:
`FamilySize = SibSp + Parch + 1`

`FamilySize` is useful, but it only tells us the size of a passenger's family group. It does not identify which passengers belong to the same family. To capture that relationship, I created a `FamilyId` feature using a staged linking system.
The family linking is performed on the combined train + test dataset so that family connections split across the two files can still be detected. The model is still trained only on the training data; the combined file is used only as a reference table for feature engineering.

## Level 1 and Level 2: Ticket + Surname
The strongest family signal is a shared ticket number combined with the same surname. Passengers with the same `Ticket` and `Surname` are treated as belonging to the same family/travel group.

`FamilySize` is then used as a confidence check.
If everyone in the group has the same `FamilySize`, the link is marked as `source 1`.
If the group has different `FamilySize` values, the passengers are still kept under the same FamilyId, but the link is marked as `source 2`. This prevents one real family from being split into multiple IDs just because of a possible inconsistency in the raw family-size data.

## Level 3: Nearby Ticket Inference
Some related passengers may not share the exact same ticket number. For passengers who are still unlinked but have FamilySize > 1, I use a weaker inference rule: `Surname + Pclass + Embarked`

Then the numeric part of their ticket is extracted. If passengers in the same group have very close ticket numbers, they are treated as a possible family link.
Example:
`347082` and `347083`
This level is marked as `source 3` because it is useful, but less reliable than exact `Ticket + Surname` matching.

## Level 4: Alone or Unknown
If no reliable link is found, the passenger is marked as `source 4`.
| Condition         | FamilyId         |
| `FamilySize == 1` | `Alone`          | treated as travelling alone
| `FamilySize > 1`  | `Unknown_Family` | exact family group could not be identified safely from the available data.
