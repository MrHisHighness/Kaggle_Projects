Deck | Source |	Meaning
A | 1 |	Deck directly known from Cabin
A | 2 |	Deck inferred from same `Ticket` group
A | 3 |	Deck inferred from [`Embarked`,`Pclass`,FareBand`]
A | 4 |	Deck inferred from either [`Embarked` + `Pclass`] or [`Pclass` + `Fareband`]
A | 5 | Deck inferred from  `Pclass` + `Embarked` 
A | 6 | Emergency Deck inferred from common `Pclass` 

## Deck Feature
Many passengers have missing cabin data, so after extracting the known decks, I used the combined train + test file as a broader reference table.

To track how reliable the deck value is, I also created a `Deck_Source` column.
- `Deck_Source = 1` means the deck was directly extracted from the passenger's own cabin value.
- `Deck_Source = 2` means the deck was inferred from another passenger with the same ticket number.

### Deck level 1
The `Deck` feature is extracted from the `Cabin` column. Since cabin values like `C85` or `E46` contain the deck letter as the first character, I first extract the known deck directly from each passenger's cabin.

### Deck level 2
If a passenger had no cabin value but shared a ticket number with another passenger who had a known deck, I filled the missing deck using the most common deck for that ticket.

This gives layered confidence levels: direct cabin-based deck values are stronger, while ticket-based deck values are weaker but still useful because passengers sharing a ticket were often travelling together or booked under the same group.

Before modeling, the rare `T` deck value was merged into `A` because it appeared only once and would otherwise act as a one-row category.

### Deck Level 3: Strong Structural Deck Inference

After direct cabin extraction and exact-ticket deck propagation, I used a stronger structural grouping to infer some remaining missing decks.

Deck Level 3 groups passengers by:

- `Embarked`
- `Pclass`
- `FareBand`

This combines boarding port, passenger class, and relative fare position within that class. The idea is that passengers with similar class/fare/boarding patterns may have been placed in similar physical regions of the ship.

However, I only used this inference when the group had enough evidence:

- At least 5 known or ticket-inferred deck examples
- The most common deck had at least 65% dominance inside the group

If both conditions were met, the missing deck was filled using the dominant deck of that group.

### Deck Level 4: Broader Structural Fallback

For passengers still missing a deck after Level 3, I used broader structural fallbacks.

These fallbacks use wider groupings such as:

- `Embarked + Pclass`
- `Pclass + FareBand`

This is weaker than Level 3 because it uses fewer passenger details, but it still preserves meaningful structure from class, fare, and boarding context.

I again applied evidence checks before filling:

- The group needed enough known deck examples
- The group needed a reasonably dominant most-common deck

### Deck Level 5: Pclass + FareBand Final Fallback

If a passenger still had no deck after the stronger structural methods, I used a final class/fare-based fallback.

This groups passengers by:

- `Pclass`
- `FareBand`

The most common known or ticket-inferred deck for that class/fare band is used.

This is weaker than Levels 1-4, but it is still better than using only passenger class because it separates low, middle, and high fare passengers within each class.

### Deck Level 6: Emergency Pclass-Only Fallback

As a final safety step, if any deck values were still missing, I used the most common known deck within the passenger's `Pclass`.

This is the weakest deck inference level because it only uses passenger class. It exists only to make sure the final modeling file has no missing deck values.

