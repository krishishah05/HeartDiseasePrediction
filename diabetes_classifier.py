import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ──────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────
df = pd.read_csv("heart_disease_dataset.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

# ──────────────────────────────────────────
# 2. CLEANING
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

print(f"\nMissing values per column:\n{df.isnull().sum()}")

dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {dupes}")
df = df.drop_duplicates()
df = df.dropna(subset=["Diabetes"])

print(f"\nShape after cleaning: {df.shape}")

# ──────────────────────────────────────────
# 3. EDA — VISUALIZATIONS
# ──────────────────────────────────────────
numeric_cols = ["Age", "Cholesterol", "Blood Pressure", "Heart Rate",
                "Exercise Hours", "Stress Level", "Blood Sugar"]
categorical_cols = ["Gender", "Smoking", "Alcohol Intake", "Family History",
                    "Obesity", "Exercise Induced Angina", "Chest Pain Type"]

print("\n" + "=" * 50)
print("EDA")
print("=" * 50)

# Target distribution
plt.figure(figsize=(5, 4))
df["Diabetes"].value_counts().plot(kind="bar", color=["steelblue", "salmon"], edgecolor="black")
plt.title("Diabetes Class Distribution")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_target_distribution.png")
plt.show()
print("Saved: plot_target_distribution.png")

# Numeric features vs Diabetes
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    df.boxplot(column=col, by="Diabetes", ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel("Diabetes")
axes[-1].set_visible(False)
plt.suptitle("Numeric Features by Diabetes Status")
plt.tight_layout()
plt.savefig("plot_numeric_boxplots.png")
plt.show()
print("Saved: plot_numeric_boxplots.png")

# Categorical features vs Diabetes
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    ct = pd.crosstab(df[col], df["Diabetes"], normalize="index")
    ct.plot(kind="bar", ax=axes[i], colormap="coolwarm", edgecolor="black")
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30)
    axes[i].legend(title="Diabetes", labels=["No", "Yes"])
axes[-1].set_visible(False)
plt.suptitle("Diabetes Rate by Categorical Feature")
plt.tight_layout()
plt.savefig("plot_categorical_bars.png")
plt.show()
print("Saved: plot_categorical_bars.png")

# Correlation heatmap (numeric only)
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png")
plt.show()
print("Saved: plot_correlation_heatmap.png")

# ──────────────────────────────────────────
# 4. FEATURE SELECTION
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE SELECTION")
print("=" * 50)

# Encode target for tests
target = df["Diabetes"].map({"Yes": 1, "No": 0})

# --- Point-Biserial Correlation (numeric features) ---
print("\nPoint-Biserial Correlation with Diabetes (p < 0.05 = significant):")
print(f"{'Feature':<30} {'Correlation':>12} {'p-value':>12} {'Selected':>10}")
print("-" * 68)

selected_numeric = []
for col in numeric_cols:
    r, p = pointbiserialr(df[col], target)
    selected = p < 0.05
    if selected:
        selected_numeric.append(col)
    print(f"{col:<30} {r:>12.4f} {p:>12.4f} {'YES' if selected else 'NO':>10}")

# --- Chi-Squared Test (categorical features) ---
print("\nChi-Squared Test with Diabetes (p < 0.05 = significant):")
print(f"{'Feature':<30} {'Chi2 Stat':>12} {'p-value':>12} {'Selected':>10}")
print("-" * 68)

# Label encode categoricals for chi2 (requires non-negative integers)
cat_encoded = df[categorical_cols].apply(LabelEncoder().fit_transform)
chi2_stats, chi2_pvalues = chi2(cat_encoded, target)

selected_categorical = []
for col, stat, p in zip(categorical_cols, chi2_stats, chi2_pvalues):
    selected = p < 0.05
    if selected:
        selected_categorical.append(col)
    print(f"{col:<30} {stat:>12.4f} {p:>12.4f} {'YES' if selected else 'NO':>10}")

print(f"\nSelected numeric features   : {selected_numeric}")
print(f"Selected categorical features: {selected_categorical}")

# ──────────────────────────────────────────
# 5. PREPROCESSING (selected features only)
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("PREPROCESSING")
print("=" * 50)

df["Diabetes"] = target

# Keep only selected features + target
df_model = df[selected_numeric + selected_categorical + ["Diabetes"]].copy()

# Encode binary Yes/No categoricals
binary_yes_no = ["Family History", "Obesity", "Exercise Induced Angina"]
for col in selected_categorical:
    if col in binary_yes_no:
        df_model[col] = df_model[col].map({"Yes": 1, "No": 0})
    elif col == "Gender":
        df_model[col] = df_model[col].map({"Male": 0, "Female": 1})

# One-hot encode remaining multi-class categoricals
multi_class = [c for c in selected_categorical
               if c not in binary_yes_no and c != "Gender"]
if multi_class:
    df_model = pd.get_dummies(df_model, columns=multi_class, drop_first=True)

print(f"Shape after encoding: {df_model.shape}")
print(f"Features used: {[c for c in df_model.columns if c != 'Diabetes']}")

# ──────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ──────────────────────────────────────────
feature_cols = [c for c in df_model.columns if c != "Diabetes"]
X = df_model[feature_cols]
y = df_model["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────
# 7. TRAIN LOGISTIC REGRESSION
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained.")

# ──────────────────────────────────────────
# 8. EVALUATE
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"F1-Score : {f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plot_confusion_matrix.png")
plt.show()
print("Saved: plot_confusion_matrix.png")

# Coefficients
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

print(f"\nFeature coefficients:\n{coef_df.to_string(index=False)}")

plt.figure(figsize=(10, 6))
colors = ["salmon" if c > 0 else "steelblue" for c in coef_df["Coefficient"]]
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="black")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression Coefficients (Selected Features)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("plot_coefficients.png")
plt.show()
print("Saved: plot_coefficients.png")
