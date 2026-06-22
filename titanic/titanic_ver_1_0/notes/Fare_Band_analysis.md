# Missing fare handling
For missing fare values, I used the combined train + test super_file_df as a reference source. The missing FarePerPerson values are filled from strongest to weakest grouping:
1 | `Embarked + Pclass + Deck`
2 | `Embarked + Pclass`
3 | `Pclass`
4 | `Overall median fare per person as final fallback`
This keeps the missing fare estimate connected to passenger class, boarding location, and cabin/deck position where available.

# Pclass-specific fare bands
Instead of creating global fare bands, I created fare bands separately inside each Pclass.
This is important because global fare bands may mostly duplicate Pclass. For example, most high fares naturally belong to 1st class, while most low fares belong to 3rd class. That would make FareBand less meaningful as a separate feature.
So the fare bands are created within each class:
Pclass 1 -> P1_LowFare, P1_MidFare, P1_HighFare
Pclass 2 -> P2_LowFare, P2_MidFare, P2_HighFare
Pclass 3 -> P3_LowFare, P3_MidFare, P3_HighFare
This allows the model to compare passengers within the same class, such as a low-fare 3rd class passenger versus a higher-fare 3rd class passenger.