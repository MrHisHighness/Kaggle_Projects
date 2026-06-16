import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset with DECK
df = pd.read_csv("gender_encoded_with_deck.csv")

# Split features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- MODEL 1: Logistic Regression -----
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_log))

# ----- MODEL 2: Random Forest -----
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=7,
    random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

# ----- MODEL 3: Gradient Boosting -----
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, pred_gb))

# Detailed report
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, pred_gb))
