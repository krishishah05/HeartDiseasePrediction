import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# load and clean data
df = pd.read_csv("heart_disease_dataset.csv")
df = df.drop_duplicates()
df = df.dropna(subset=["Diabetes"])

# encode target
df["Diabetes"] = df["Diabetes"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Diabetes"])

print("Shape:", df.shape)
print("Class balance:", df["Diabetes"].value_counts().to_dict())

# split before encoding to avoid data leakage
X = df.drop(columns=["Diabetes", "Heart Disease"])
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# one-hot encode on train only, align test to same columns
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test,  drop_first=True)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))
print("Features:", X_train.shape[1])

# logistic regression baseline (pipeline keeps scaling inside each CV fold)
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(max_iter=1000, random_state=42)),
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_f1  = f1_score(y_test, lr_pred)
print("\n--- Logistic Regression ---")
print("Accuracy:", round(lr_acc, 4))
print("F1-Score:", round(lr_f1, 4))

# decision tree with grid search hyperparameter tuning
param_grid = {
    "criterion"        : ["gini", "entropy"],
    "max_depth"        : [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=1,
)
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_
print("\n--- Decision Tree ---")
print("Best params:", grid_search.best_params_)

dt_pred = best_dt.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
dt_f1   = f1_score(y_test, dt_pred)
print("Accuracy:", round(dt_acc, 4))
print("F1-Score:", round(dt_f1, 4))
print("\n", classification_report(y_test, dt_pred, target_names=["No Diabetes", "Diabetes"]))

# confusion matrix
cm = confusion_matrix(y_test, dt_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plot_dt_confusion_matrix.png")
plt.show()

# model comparison table
results = {
    "Model"    : ["Logistic Regression", "Decision Tree (Tuned)"],
    "Accuracy" : [lr_acc, dt_acc],
    "F1-Score" : [lr_f1,  dt_f1],
}
comparison_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# comparison bar chart
x     = range(len(results["Model"]))
width = 0.3
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar([i - width/2 for i in x], results["Accuracy"], width, label="Accuracy",  color="steelblue", edgecolor="black")
ax.bar([i + width/2 for i in x], results["F1-Score"], width, label="F1-Score",  color="salmon",    edgecolor="black")

ax.set_xticks(list(x))
ax.set_xticklabels(results["Model"], fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("plot_stage2_comparison.png")
plt.show()

# feature importances
fi_df = pd.DataFrame({
    "Feature"   : X_train.columns,
    "Importance": best_dt.feature_importances_,
}).sort_values("Importance", ascending=False).head(10)

print("\nTop 10 Feature Importances:")
print(fi_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="steelblue", edgecolor="black")
plt.title("Decision Tree - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plot_dt_feature_importance.png")
plt.show()
