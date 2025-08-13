# ==================================================
# Forest Cover Type Classification - Complete Script
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier, plot_importance

# -----------------------------
# 1) Load dataset
# -----------------------------
CSV_PATH = "covtype.csv"  # CSV file path
df = pd.read_csv(CSV_PATH)

print("\n>>> Dataset shape:", df.shape)
print("\n>>> First 5 rows:")
print(df.head())

# Target column name
target_col = "Cover_Type"
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# 2) Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3) Random Forest Classifier
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# -----------------------------
# 4) XGBoost Classifier
#    (Shift labels from 1–7 to 0–6)
# -----------------------------
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    objective='multi:softmax',
    num_class=len(np.unique(y))
)
xgb.fit(X_train, y_train_xgb)
xgb_preds = xgb.predict(X_test) + 1  # Shift back to 1–7

# -----------------------------
# 5) Evaluation function
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)

# -----------------------------
# 6) Feature Importance
# -----------------------------
# Random Forest Feature Importance
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
rf_importances.head(20).plot(kind='bar')
plt.title("Random Forest - Top 20 Feature Importances")
plt.show()

# XGBoost Feature Importance
plt.figure(figsize=(10, 6))
plot_importance(xgb, max_num_features=20)
plt.title("XGBoost - Top 20 Feature Importances")
plt.show()

# -----------------------------
# 7) Model Comparison
# -----------------------------
comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, rf_preds),
        accuracy_score(y_test, xgb_preds)
    ]
})
print("\nModel Comparison:")
print(comparison)

# -----------------------------
# 8) Bonus - Hyperparameter Tuning Example
# -----------------------------
# Uncomment to run (can take long)
"""
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
best_preds = best_rf.predict(X_test)
print("Tuned RF Accuracy:", accuracy_score(y_test, best_preds))
"""
