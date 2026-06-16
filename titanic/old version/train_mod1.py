# features.py
import re
import numpy as np
import pandas as pd

def ticket_root(s):
    if pd.isna(s): return "NA"
    t = re.sub(r"[./\s]", "", str(s))
    m = re.match(r"([A-Za-z]+)", t)
    return m.group(1) if m else "NA"

def engineer(df):
    out = df.copy()
    out["Surname"]     = out["Name"].str.extract(r"^([^,]+),")
    out["Title"]       = out["Name"].str.extract(r",\s*([^\.]+)\.")
    rare = out["Title"].value_counts()
    out["Title"]       = out["Title"].where(~out["Title"].isin(rare[rare<10].index), "Rare")
    out["FamilySize"]  = out["SibSp"].fillna(0)+out["Parch"].fillna(0)+1
    out["IsAlone"]     = (out["FamilySize"]==1).astype(int)
    out["TicketRoot"]  = out["Ticket"].apply(ticket_root)
    out["TicketCount"] = out.groupby("Ticket")["Ticket"].transform("count")
    out["IsSharedTicket"] = (out["TicketCount"]>1).astype(int)
    out["FarePerPerson"]  = (out["Fare"]/(out["FamilySize"].clip(lower=1))).astype(float)
    out["HasCabin"]    = out["Cabin"].notna().astype(int)
    out["Deck"]        = out["Cabin"].str[0].fillna("U")
    out["IsWomanChild"]= ((out["Sex"]=="female") | (out["Age"]<14)).astype(int)
    out["Pclass"]      = out["Pclass"].astype(int)
    return out
# modeling.py (sketch)
from sklearn.model_selection import GroupKFold
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np, pandas as pd

def cv_train(train):
    X = train.drop(columns=["Survived"])
    y = train["Survived"].values
    groups = train["Surname"].fillna("NA") + "_" + train["TicketRoot"].fillna("NA")

    num = ["Age","SibSp","Parch","Fare","FarePerPerson","FamilySize","TicketCount"]
    cat = ["Sex","Embarked","Title","Deck","TicketRoot","IsAlone","IsWomanChild","HasCabin","Pclass"]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            # target encoding ONLY for higher-card cats
            ("te", TargetEncoder(cols=["Surname","TicketRoot","Deck"], smoothing=0.3))
        ]), cat)
    ], remainder="drop")

    # monotone constraints example: Pclass ↑ => death ↑ (+1), IsWomanChild ↑ => death ↓ (-1)
    clf = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
        # if using LightGBM monotone: monotone_constraints=[+1,...,-1,...] aligned to features
    )

    pipe = Pipeline([("eng", None), ("pre", pre), ("clf", clf)])

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train))
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:,1]
    # calibrate
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    return pipe, iso, oof
