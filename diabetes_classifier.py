import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
df = pd.read_csv("heart_disease_dataset.csv")
print("Shape of data:", df.shape)


# getting rid of duplicates
df = df.drop_duplicates()
df = df.dropna(subset=["Diabetes"])
print("Shape after cleaning:", df.shape)

# column names by type
numeric_cols = ["Age", "Cholesterol", "Blood Pressure", "Heart Rate",
                "Exercise Hours", "Stress Level", "Blood Sugar"]
categorical_cols = ["Gender", "Smoking", "Alcohol Intake", "Family History",
                    "Obesity", "Exercise Induced Angina", "Chest Pain Type"]

target = df["Diabetes"].map({"Yes": 1, "No": 0})


# how many Yes vs No in the target
plt.figure(figsize=(5, 4))
df["Diabetes"].value_counts().plot(kind="bar", color=["steelblue", "salmon"], edgecolor="black")
plt.title("Diabetes Class Distribution")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_target_distribution.png")
plt.show()


# boxplots of each numeric feature split by target
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i in range(len(numeric_cols)):
    col = numeric_cols[i]
    df.boxplot(column=col, by="Diabetes", ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel("Diabetes")
axes[-1].set_visible(False)
plt.suptitle("Numeric Features by Diabetes Status")
plt.tight_layout()
plt.savefig("plot_numeric_boxplots.png")
plt.show()

# bar charts of each categorical feature vs target
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i in range(len(categorical_cols)):
    col = categorical_cols[i]
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

# correlation heatmap of numeric features
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png")
plt.show()


# building the data
df_model = df[numeric_cols + categorical_cols].copy()
df_model["Diabetes"] = target

# encode yes and no
yes_no_cols = ["Family History", "Obesity", "Exercise Induced Angina"]
for col in yes_no_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0})

# encode gender
df_model["Gender"] = df_model["Gender"].map({"Male": 0, "Female": 1})

# One-hot encoding
df_model = pd.get_dummies(df_model,
                          columns=["Smoking", "Alcohol Intake", "Chest Pain Type"],
                          drop_first=True)


# split x and y axis
y = df_model["Diabetes"]
X = df_model.drop(columns=["Diabetes"])

# split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)


# evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 4))
print("F1-Score:", round(f1, 4))
