import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease_dataset.csv")
plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)


def save_plot(filename):
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap(matrix, labels, title, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(len(labels) * 1.2 + 1, len(labels) * 1.2))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()


def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()


def main():
    df = pd.read_csv(data_path)

    print("\nDataset check")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    print("\nData types:")
    print(df.dtypes)

    numeric_cols = [
        "Age",
        "Cholesterol",
        "Blood Pressure",
        "Heart Rate",
        "Exercise Hours",
        "Stress Level",
        "Blood Sugar",
    ]
    categorical_cols = [
        "Gender",
        "Smoking",
        "Family History",
        "Obesity",
        "Exercise Induced Angina",
        "Chest Pain Type",
    ]

    print("\nCleaning")
    print("-" * 40)
    print("Missing values per column (before cleaning):")
    print(df.isna().sum())

    if "Alcohol Intake" in df.columns:
        df = df.drop(columns=["Alcohol Intake"])
        print("\nDropped column: Alcohol Intake")

    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")

    df = df.drop_duplicates()
    df = df.dropna(subset=["Heart Disease"])
    df["Heart Disease"] = df["Heart Disease"].astype(int)

    print("\nMissing values per column (after cleaning):")
    print(df.isna().sum())
    print(f"\nShape after cleaning: {df.shape}")
    print("Target distribution after cleaning:")
    print(df["Heart Disease"].value_counts())

    print("\nEDA")
    print("-" * 40)

    plt.figure(figsize=(5, 4))
    df["Heart Disease"].value_counts().sort_index().plot(
        kind="bar", color=["steelblue", "salmon"], edgecolor="black"
    )
    plt.title("Heart Disease Class Distribution")
    plt.xlabel("Heart Disease (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    save_plot("plot_target_distribution.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        df.boxplot(column=col, by="Heart Disease", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("Heart Disease")
    axes[-1].set_visible(False)
    plt.suptitle("Numeric Features by Heart Disease Status")
    save_plot("plot_numeric_boxplots.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        ct = pd.crosstab(df[col], df["Heart Disease"], normalize="index")
        ct.plot(kind="bar", ax=axes[i], colormap="coolwarm", edgecolor="black")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    axes[-1].set_visible(False)
    plt.suptitle("Heart Disease Rate by Categorical Feature")
    save_plot("plot_categorical_bars.png")

    corr = df[numeric_cols].corr().values
    plot_heatmap(corr, numeric_cols, "Correlation Heatmap (Numeric Features)")
    save_plot("plot_correlation_heatmap.png")

    print("\nModel training")
    print("-" * 40)

    selected_numeric = numeric_cols.copy()
    selected_categorical = categorical_cols.copy()
    print(f"Selected numeric features: {selected_numeric}")
    print(f"Selected categorical features: {selected_categorical}")

    df_model = df[selected_numeric + selected_categorical + ["Heart Disease"]].copy()
    X = df_model.drop(columns=["Heart Disease"])
    y = df_model["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), selected_numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), selected_categorical),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Heart Disease", "Heart Disease"]))

    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Heart Disease", "Heart Disease"]
    plot_confusion_matrix(cm, labels, "Confusion Matrix - Logistic Regression")
    save_plot("plot_confusion_matrix.png")


if __name__ == "__main__":
    main()
