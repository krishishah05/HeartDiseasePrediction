import io
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Stage 3 - ML App", layout="wide")
st.title("CS301 Stage 3: Application Development & Deployment")
st.caption(
    "Upload CSV, choose target/features, train using a scikit-learn pipeline, and run manual predictions."
)


def normalize_yes_no(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    s = series.astype(str).str.strip()
    replacements = {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "1": "Yes",
        "no": "No",
        "n": "No",
        "false": "No",
        "0": "No",
    }
    lowered = s.str.lower()
    mapped = lowered.map(replacements)
    return mapped.where(mapped.notna(), s)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
            out[col] = normalize_yes_no(out[col])
    out = out.drop_duplicates()
    return out


def infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        unique_count = y.nunique(dropna=True)
        if unique_count <= 10:
            return "classification"
        return "regression"
    return "classification"


def split_feature_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    categorical_features = [f for f in features if f not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def choose_model(task_type: str, model_name: str):
    if task_type == "classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        return RandomForestClassifier(n_estimators=300, random_state=42)

    if model_name == "Linear Regression":
        return LinearRegression()
    return RandomForestRegressor(n_estimators=300, random_state=42)


def target_summary(df: pd.DataFrame, target: str, task_type: str) -> None:
    st.subheader("Target Summary")
    y = df[target]

    if task_type == "classification":
        counts = y.value_counts(dropna=False)
        pct = (counts / len(y) * 100).round(2)
        summary = pd.DataFrame({"count": counts, "percent": pct})
        st.dataframe(summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Class Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        stats = pd.DataFrame(
            {
                "mean": [y.mean()],
                "median": [y.median()],
                "std": [y.std()],
                "min": [y.min()],
                "max": [y.max()],
            }
        )
        st.dataframe(stats, use_container_width=True)


def correlation_bar_chart(df: pd.DataFrame, features: List[str], target: str, task_type: str) -> None:
    st.subheader("Feature Correlation Bar Chart")
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    if not numeric_features:
        st.warning("No numeric features selected. Please include at least one numeric feature.")
        return

    corr_df = df[numeric_features + [target]].copy()

    if task_type == "classification" and not pd.api.types.is_numeric_dtype(corr_df[target]):
        encoded = pd.factorize(corr_df[target])[0]
        corr_df[target] = encoded

    corr = corr_df.corr(numeric_only=True)[target].drop(target, errors="ignore")
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    corr.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
    ax.set_title(f"Correlation of Numeric Features with {target}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Correlation")
    ax.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(corr.rename("correlation").to_frame(), use_container_width=True)


def manual_prediction_form(df: pd.DataFrame, features: List[str], pipeline: Pipeline, task_type: str) -> None:
    st.subheader("Manual Input Prediction")
    st.write("Enter values below for a real-time prediction.")

    user_input = {}
    cols = st.columns(2)

    for i, feature in enumerate(features):
        col = cols[i % 2]
        series = df[feature]

        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if not series.dropna().empty else 0.0
            user_input[feature] = col.number_input(feature, value=default)
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            user_input[feature] = col.selectbox(feature, options=options)

    if st.button("Predict", type="primary"):
        sample = pd.DataFrame([user_input])
        pred = pipeline.predict(sample)[0]

        st.markdown("### Prediction Result")
        st.write(f"Predicted value: **{pred}**")

        if task_type == "classification" and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(sample)[0]
            classes = pipeline.named_steps["model"].classes_
            proba_df = pd.DataFrame({"class": classes, "probability": proba}).sort_values(
                "probability", ascending=False
            )
            st.dataframe(proba_df, use_container_width=True)


st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin. You can use `data/heart_disease_dataset.csv` from this project.")
    st.stop()

raw_bytes = uploaded_file.read()
if not raw_bytes:
    st.error("Uploaded file is empty.")
    st.stop()

try:
    df_raw = pd.read_csv(io.BytesIO(raw_bytes))
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

df = clean_dataframe(df_raw)

st.subheader("Uploaded Data")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]}")
col2.metric("Columns", f"{df.shape[1]}")
col3.metric("Duplicates Removed", f"{len(df_raw) - len(df)}")
st.dataframe(df.head(20), use_container_width=True)

missing_summary = df.isna().sum().rename("missing_values").to_frame()
with st.expander("Missing Values by Column"):
    st.dataframe(missing_summary, use_container_width=True)

if df.shape[1] < 12:
    st.warning("Dataset has fewer than 12 features. Stage 1/3 requirement asks for at least 12 features.")

st.sidebar.header("Model Setup")
all_columns = df.columns.tolist()
target_col = st.sidebar.selectbox("Select target column", options=all_columns)

inferred_task = infer_task_type(df[target_col])
task_override = st.sidebar.selectbox(
    "Task type",
    options=["classification", "regression"],
    index=0 if inferred_task == "classification" else 1,
    help="Auto-inferred from target data type; you can override if needed.",
)

task_type = task_override

candidate_features = [c for c in all_columns if c != target_col]
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = candidate_features.copy()
if "select_all_features" not in st.session_state:
    st.session_state["select_all_features"] = True
if "feature_target_col" not in st.session_state:
    st.session_state["feature_target_col"] = target_col

if st.session_state["feature_target_col"] != target_col:
    st.session_state["feature_target_col"] = target_col
    st.session_state["selected_features"] = candidate_features.copy()
    st.session_state["select_all_features"] = True

st.session_state["selected_features"] = [
    f for f in st.session_state["selected_features"] if f in candidate_features
]
st.session_state["select_all_features"] = len(st.session_state["selected_features"]) == len(candidate_features)


def on_select_all_features_change() -> None:
    if st.session_state["select_all_features"]:
        st.session_state["selected_features"] = candidate_features.copy()


def on_selected_features_change() -> None:
    st.session_state["select_all_features"] = (
        len(st.session_state["selected_features"]) == len(candidate_features)
    )


st.sidebar.checkbox(
    "Select all input features",
    key="select_all_features",
    on_change=on_select_all_features_change,
)
selected_features = st.sidebar.multiselect(
    "Select input features",
    options=candidate_features,
    key="selected_features",
    on_change=on_selected_features_change,
)

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

st.sidebar.markdown("---")
if task_type == "classification":
    model_name = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])
else:
    model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])

test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42)

target_summary(df, target_col, task_type)
correlation_bar_chart(df, selected_features, target_col, task_type)

train_clicked = st.button("Train Model", type="primary")
if not train_clicked:
    st.stop()

model_df = df[selected_features + [target_col]].copy()
model_df = model_df.dropna(subset=[target_col])
X = model_df[selected_features]
y = model_df[target_col]

numeric_features, categorical_features = split_feature_types(X, selected_features)
preprocessor = build_preprocessor(numeric_features, categorical_features)
model = choose_model(task_type, model_name)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

stratify = y if task_type == "classification" else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.subheader("Model Evaluation")
if task_type == "classification":
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("F1-Score (weighted)", f"{f1:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
else:
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    m1, m2 = st.columns(2)
    m1.metric("R^2", f"{r2:.4f}")
    m2.metric("RMSE", f"{rmse:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

manual_prediction_form(df, selected_features, pipeline, task_type)
