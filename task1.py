# ============================================
# Student Score Prediction — Complete Pipeline
# ============================================
# What this script does:
# 1) Load & clean data
# 2) Basic EDA (head, info)
# 3) Correlation heatmap (not overcrowded): only top k features vs target
# 4) Train/test split
# 5) Models:
#    A) Linear Regression (StudyHours only)
#    B) Linear Regression (Multi-feature)
#    C) Polynomial Regression (degree=2) on top features
# 6) Evaluate with MAE, RMSE, R² + visualizations
#
# Notes:
# - Automatically detects likely target ("ExamScore", "final_score", etc.)
# - Automatically detects "study hours" column
# - One-hot encodes categoricals, median-imputes numerics
# - Safe if some columns or names vary; will guide you if missing.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 0) Config: dataset path here
# -----------------------------
CSV_PATH = "StudentPerformanceFactors.csv"  # <--- CSV file path

# ---------------------------------------
# 1) Load data + normalize column names
# ---------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lower-case, strip, replace spaces & special chars with underscores
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

df = pd.read_csv(CSV_PATH)
df = normalize_columns(df)

print("\n>>> HEAD")
print(df.head())
print("\n>>> INFO")
print(df.info())

# -------------------------------------------------
# 2) Identify target and study-hours feature names
# -------------------------------------------------
# Common target column candidates (adjust if needed)
TARGET_CANDIDATES = [
    "exam_score", "final_score", "final_grade", "score", "performance_index",
    "math_score", "reading_score", "writing_score", "overall_grade"
]

# Common study-hours candidates
STUDY_HOURS_CANDIDATES = [
    "studyhours", "study_hours", "hours_studied", "hours_of_study",
    "studytime", "study_time", "hours", "time_study"
]

def find_first_present(candidates: List[str], cols: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

target_col = find_first_present(TARGET_CANDIDATES, df.columns.tolist())
study_col  = find_first_present(STUDY_HOURS_CANDIDATES, df.columns.tolist())

if target_col is None:
    raise ValueError(
        "Could not find the target column. "
        f"Tried: {TARGET_CANDIDATES}. Please rename your target column to one of these."
    )

if study_col is None:
    raise ValueError(
        "Could not find the study-hours column. "
        f"Tried: {STUDY_HOURS_CANDIDATES}. Please rename your study-hours column to one of these."
    )

print(f"\nDetected target column: {target_col}")
print(f"Detected study-hours column: {study_col}")

# -------------------------------------
# 3) Basic cleaning & type inference
# -------------------------------------
# Drop exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Separate numeric and non-numeric for imputation/encoding
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

# Fill numeric NaNs with median
for c in numeric_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# For non-numeric: fill NaNs with most frequent then one-hot encode
for c in non_numeric_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

# Ensure target exists & is numeric after encoding
if target_col not in df.columns:
    raise ValueError(
        f"Target column '{target_col}' disappeared after encoding. "
        "Ensure target is numeric and not categorical."
    )

# -----------------------------------------------
# 4) Correlation heatmap (NOT overcrowded)
#    - Use only the top-k correlated features with
#      the target (absolute correlation)
# -----------------------------------------------
corr = df.corr(numeric_only=True)
# If target is not numeric, this will error earlier. Safe here.
target_corr = corr[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False)

TOP_K_FOR_HEATMAP = 10  # keep small to avoid clutter
top_features = target_corr.head(TOP_K_FOR_HEATMAP).index.tolist()
heatmap_cols = [target_col] + top_features

plt.figure(figsize=(8, 6))
plt.imshow(corr.loc[heatmap_cols, heatmap_cols], aspect='auto')
plt.xticks(range(len(heatmap_cols)), heatmap_cols, rotation=45, ha='right')
plt.yticks(range(len(heatmap_cols)), heatmap_cols)
plt.title(f'Correlation Heatmap (Top {TOP_K_FOR_HEATMAP} vs Target)')
plt.colorbar()
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 5) Feature sets: minimal (study-hours) & extended
# ---------------------------------------------------
# Minimal: only study-hours
X_min = df[[study_col]].copy()
y = df[target_col].copy()

# Extended: drop target; keep everything else
X_full = df.drop(columns=[target_col]).copy()

# -----------------------------------------
# 6) Train/test split (consistent splits)
# -----------------------------------------
Xmin_train, Xmin_test, ymin_train, ymin_test = train_test_split(
    X_min, y, test_size=0.2, random_state=42
)

Xfull_train, Xfull_test, yfull_train, yfull_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 7A) Linear Regression — StudyHours only
# -----------------------------------------
lr_min = LinearRegression()
lr_min.fit(Xmin_train, ymin_train)
ymin_pred = lr_min.predict(Xmin_test)

# -----------------------------------------
# 7B) Linear Regression — Multiple features
# -----------------------------------------
lr_full = LinearRegression()
lr_full.fit(Xfull_train, yfull_train)
yfull_pred = lr_full.predict(Xfull_test)

# ------------------------------------------------------
# 7C) Polynomial Regression (degree=2) on key features
#     We'll use the top 3 features by correlation with
#     target (plus study-hours, ensuring it's included).
# ------------------------------------------------------
top3 = target_corr.index[:3].tolist()
if study_col not in top3:
    top3 = [study_col] + [f for f in top3 if f != study_col]
poly_feats = list(dict.fromkeys(top3))  # unique, keep order

X_poly = df[poly_feats].copy()
Xpoly_train, Xpoly_test, ypoly_train, ypoly_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin", LinearRegression())
])
poly_model.fit(Xpoly_train, ypoly_train)
ypoly_pred = poly_model.predict(Xpoly_test)

# -----------------------------------------
# 8) Evaluation helper
# -----------------------------------------
def evaluate(y_true, y_pred, label: str) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{label}]")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    return mae, rmse, r2

res_min  = evaluate(ymin_test,  ymin_pred,  "Linear (StudyHours only)")
res_full = evaluate(yfull_test, yfull_pred, "Linear (Multi-feature)")
res_poly = evaluate(ypoly_test, ypoly_pred, "Polynomial (deg=2, key features)")

# -----------------------------------------
# 9) Visualizations
#    - Actual vs Predicted (for the 3 models)
#    - Residual plots (optional)
# -----------------------------------------

def plot_actual_vs_pred(y_true, y_pred, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    # y=x reference line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_actual_vs_pred(ymin_test,  ymin_pred,  "Actual vs Predicted — Linear (StudyHours only)")
plot_actual_vs_pred(yfull_test, yfull_pred, "Actual vs Predicted — Linear (Multi-feature)")
plot_actual_vs_pred(ypoly_test, ypoly_pred, "Actual vs Predicted — Polynomial (deg=2)")

def plot_residuals(y_true, y_pred, title: str):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20)
    plt.title(f"Residuals — {title}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

plot_residuals(ymin_test,  ymin_pred,  "Linear (StudyHours only)")
plot_residuals(yfull_test, yfull_pred, "Linear (Multi-feature)")
plot_residuals(ypoly_test, ypoly_pred, "Polynomial (deg=2)")

# -----------------------------------------
# 10) Quick model comparison table
# -----------------------------------------
summary = pd.DataFrame(
    [res_min, res_full, res_poly],
    columns=["MAE", "RMSE", "R2"],
    index=[
        "Linear (StudyHours only)",
        "Linear (Multi-feature)",
        "Polynomial (deg=2, key feats)"
    ]
)
print("\n>>> Model Performance Summary")
print(summary)

# -----------------------------------------
# 11) Optional: Feature coefficients (full LR)
#     (Helps interpret which features matter)
# -----------------------------------------
coef_series = pd.Series(lr_full.coef_, index=X_full.columns).sort_values(key=lambda s: s.abs(), ascending=False)
print("\n>>> Top 10 features by |coefficient| (Linear Multi-feature)")
print(coef_series.head(10))

# -----------------------------------------
# 12) Bonus: Try removing/adding features
#     Example: drop sleep or participation if present
# -----------------------------------------
def try_feature_subset(drop_cols: List[str]):
    drop_cols = [c for c in drop_cols if c in X_full.columns]
    if not drop_cols:
        print("\n(No matching columns to drop for this trial.)")
        return
    X_sub = X_full.drop(columns=drop_cols)
    Xtr, Xte, ytr, yte = train_test_split(X_sub, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)
    yhat = model.predict(Xte)
    _ = evaluate(yte, yhat, f"Linear (Dropped: {', '.join(drop_cols)})")

# Example trials (uncomment to experiment if your dataset has these):
# try_feature_subset(["sleep_hours", "sleep", "participation"])
# try_feature_subset(["attendance", "parental_education"])
