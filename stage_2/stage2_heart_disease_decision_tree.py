import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease_dataset.csv")
plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)


def save_plot(filename):
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    df = pd.read_csv(data_path)
    if "Alcohol Intake" in df.columns:
        df = df.drop(columns=["Alcohol Intake"])

    df = df.drop_duplicates()
    df = df.dropna(subset=["Heart Disease"])
    df["Heart Disease"] = df["Heart Disease"].astype(int)

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
        "Diabetes",
        "Family History",
        "Obesity",
        "Exercise Induced Angina",
        "Chest Pain Type",
    ]

    X = df[numeric_cols + categorical_cols]
    y = df["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Shape:", df.shape)
    print("Class balance:", y.value_counts().to_dict())
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    print("\n--- Logistic Regression (Baseline) ---")
    print("Accuracy:", round(lr_acc, 4))
    print("F1-Score:", round(lr_f1, 4))

    dt_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )
    param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [3, 5, 7, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    grid_search = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring="f1", n_jobs=1)
    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_
    dt_pred = best_dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    dt_cv_f1 = cross_val_score(best_dt, X_train, y_train, cv=5, scoring="f1")

    print("\n--- Decision Tree (Tuned) ---")
    print("Best params:", grid_search.best_params_)
    print("Accuracy:", round(dt_acc, 4))
    print("F1-Score:", round(dt_f1, 4))
    print("5-Fold CV F1:", [round(v, 4) for v in dt_cv_f1], "Mean:", round(dt_cv_f1.mean(), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, dt_pred, target_names=["No Heart Disease", "Heart Disease"]))

    cm = confusion_matrix(y_test, dt_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Heart Disease", "Heart Disease"],
        yticklabels=["No Heart Disease", "Heart Disease"],
    )
    plt.title("Confusion Matrix - Decision Tree (Tuned)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("plot_dt_confusion_matrix.png")

    comparison_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree (Tuned)"],
            "Accuracy": [lr_acc, dt_acc],
            "F1-Score": [lr_f1, dt_f1],
        }
    )
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    x = range(len(comparison_df))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], comparison_df["Accuracy"], width, label="Accuracy", color="steelblue", edgecolor="black")
    ax.bar([i + width / 2 for i in x], comparison_df["F1-Score"], width, label="F1-Score", color="salmon", edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison_df["Model"], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    plt.tight_layout()
    save_plot("plot_stage2_dt_comparison.png")

    model = best_dt.named_steps["model"]
    feature_names = best_dt.named_steps["preprocess"].get_feature_names_out()
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    print("\nTop 10 Feature Importances:")
    print(fi_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="steelblue", edgecolor="black")
    plt.title("Decision Tree - Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    save_plot("plot_dt_feature_importance.png")


if __name__ == "__main__":
    main()
