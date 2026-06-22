Deck | Source |	Meaning
A | 1 |	Deck directly known from Cabin
A | 2 |	Deck inferred from same Ticket group
A | 3 |	Deck inferred from same Surname/FamilyKey
A | 4 |	Deck inferred from Pclass + FarePerPerson + Embarked
A | unknown	| Not enough signal

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