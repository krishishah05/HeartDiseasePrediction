from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data" / "heart_disease_dataset.csv"
plots_dir = base_dir / "stage_1" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str) -> None:
    output_path = plots_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
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
    df = df.dropna(subset=["Diabetes"])
    df["Diabetes"] = df["Diabetes"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Diabetes"])
    df["Diabetes"] = df["Diabetes"].astype(int)

    print("\nMissing values per column (after cleaning):")
    print(df.isna().sum())
    print(f"\nShape after cleaning: {df.shape}")
    print("Target distribution after cleaning:")
    print(df["Diabetes"].value_counts())

    print("\nEDA")
    print("EDA")
    print("-" * 40)

    plt.figure(figsize=(5, 4))
    df["Diabetes"].value_counts().sort_index().plot(
        kind="bar", color=["steelblue", "salmon"], edgecolor="black"
    )
    plt.title("Diabetes Class Distribution")
    plt.xlabel("Diabetes (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    save_plot("plot_target_distribution.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        df.boxplot(column=col, by="Diabetes", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("Diabetes")
    axes[-1].set_visible(False)
    plt.suptitle("Numeric Features by Diabetes Status")
    save_plot("plot_numeric_boxplots.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        ct = pd.crosstab(df[col], df["Diabetes"], normalize="index")
        ct.plot(kind="bar", ax=axes[i], colormap="coolwarm", edgecolor="black")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    axes[-1].set_visible(False)
    plt.suptitle("Diabetes Rate by Categorical Feature")
    save_plot("plot_categorical_bars.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features)")
    save_plot("plot_correlation_heatmap.png")

    print("\nModel training")
    print("-" * 40)

    selected_numeric = numeric_cols.copy()
    selected_categorical = categorical_cols.copy()
    print(f"Selected numeric features: {selected_numeric}")
    print(f"Selected categorical features: {selected_categorical}")

    df_model = df[selected_numeric + selected_categorical + ["Diabetes"]].copy()
    X = df_model.drop(columns=["Diabetes"])
    y = df_model["Diabetes"]

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
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("plot_confusion_matrix.png")


if __name__ == "__main__":
    main()
