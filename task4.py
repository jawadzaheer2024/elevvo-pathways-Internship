# ==========================================
# Task 4 — Loan Approval Prediction (Complete)
# ==========================================
# What you get:
# - Robust column & value cleaning (handles trailing spaces in 'loan_status')
# - Missing value imputation (numeric: median, categorical: mode)
# - Categorical encoding via one-hot (train/test-safe using align)
# - Class imbalance handling with SMOTE (train only)
# - Models: Logistic Regression vs Decision Tree
# - Metrics: precision, recall, F1, confusion matrix, ROC–AUC
# - Clear printouts + plots
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "loan_approval_dataset.csv"  # <-- CSV file name

# -----------------------------
# 1) Load & clean column names
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# Normalize column names for robust target detection (lowercase, underscores)
def normalize_col(s: str) -> str:
    return (
        s.strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

norm_map = {c: normalize_col(c) for c in df.columns}
rev_map = {v: k for k, v in norm_map.items()}  # normalized -> original
df.rename(columns=norm_map, inplace=True)

# Try to find the target column 'loan_status'
possible_targets = ["loan_status", "status", "approval_status"]
target_col = next((c for c in possible_targets if c in df.columns), None)
if target_col is None:
    raise ValueError(
        f"Could not find target column. Looked for: {possible_targets}. "
        f"Columns present: {list(df.columns)}"
    )

# -----------------------------
# 2) Clean target values
# -----------------------------
# Ensure text, strip spaces, lower
df[target_col] = df[target_col].astype(str).str.strip().str.lower()

# Map approved/rejected to 1/0 (adjust here if your dataset uses different labels)
label_map = {"approved": 1, "reject": 0, "rejected": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
y = df[target_col].map(label_map)

if y.isna().any():
    # Show unseen labels to help user correct quickly
    unseen = sorted(df[target_col].unique().tolist())
    raise ValueError(
        "Target has values I couldn't map to {Approved/Rejected}. "
        f"Found values: {unseen}. Update 'label_map' accordingly."
    )

# -----------------------------
# 3) Drop non-predictive ID cols
# -----------------------------
X = df.drop(columns=[target_col])
for id_like in ["loan_id", "id", "app_id", "application_id"]:
    if id_like in X.columns:
        X = X.drop(columns=[id_like])

# -----------------------------
# 4) Missing values
# -----------------------------
# Numeric -> median; Categorical -> mode
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

for c in num_cols:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())

for c in cat_cols:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].mode().iloc[0])

# -----------------------------
# 5) Train / Test split
# -----------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 6) One-hot encode categoricals safely
# -----------------------------
X_train = pd.get_dummies(X_train_raw, drop_first=True)
X_test  = pd.get_dummies(X_test_raw,  drop_first=True)

# Align test to train columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# -----------------------------
# 7) Address class imbalance with SMOTE (train only)
# -----------------------------
print("\nClass counts before SMOTE:\n", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nClass counts after SMOTE:\n", y_train_res.value_counts())

# -----------------------------
# 8) Models
# -----------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
log_reg.fit(X_train_res, y_train_res)
log_pred = log_reg.predict(X_test)
log_prob = log_reg.predict_proba(X_test)[:, 1]

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train_res, y_train_res)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]

# -----------------------------
# 9) Evaluation helpers
# -----------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n{name} — Classification Report (positive=Approved=1)")
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc(models):
    plt.figure(figsize=(6.5, 5.5))
    for name, probs in models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 10) Report & plots
# -----------------------------
evaluate("Logistic Regression", y_test, log_pred)
evaluate("Decision Tree", y_test, dt_pred)

plot_roc([
    ("Logistic Regression", log_prob),
    ("Decision Tree", dt_prob),
])

# Quick comparison table
def f1_prec_rec(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )

lp, lr_, lf1 = f1_prec_rec(y_test, log_pred)
dp, dr_, df1 = f1_prec_rec(y_test, dt_pred)

summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Precision": [lp, dp],
    "Recall": [lr_, dr_],
    "F1-Score": [lf1, df1],
    "ROC-AUC": [roc_auc_score(y_test, log_prob), roc_auc_score(y_test, dt_prob)]
}).sort_values("F1-Score", ascending=False)

print("\n=== Model Comparison (focus on Precision/Recall/F1) ===\n")
print(summary.to_string(index=False))

# -----------------------------
# 11) (Optional) Threshold tuning demo for Logistic Regression
#      Uncomment to check F1 across thresholds (useful on imbalanced data)
# -----------------------------
"""
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.9, 17)
scores = []
for t in thresholds:
    pred_t = (log_prob >= t).astype(int)
    scores.append((t, f1_score(y_test, pred_t)))
print("\\nLogistic Regression — F1 by threshold:")
for t, s in scores:
    print(f"threshold={t:.2f}  F1={s:.4f}")
"""
