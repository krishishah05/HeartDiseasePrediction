import base64
import io
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Dash(__name__)
server = app.server

model_store = {"pipeline": None, "features": [], "df": None, "target": None, "task": None}

app.layout = html.Div(
    style={"fontFamily": "Arial", "maxWidth": "1100px", "margin": "auto", "padding": "20px"},
    children=[
        dcc.Store(id="stored-data"),

        dcc.Upload(
            id="upload-data",
            children=html.Div(
                "Upload File",
                style={
                    "textAlign": "center", "padding": "12px",
                    "backgroundColor": "#d3d3d3", "cursor": "pointer",
                    "fontWeight": "bold", "fontSize": "14px",
                },
            ),
            style={"width": "100%", "marginBottom": "4px"},
            multiple=False,
        ),
        html.Div(id="upload-info", style={"fontSize": "12px", "color": "gray", "marginBottom": "6px"}),

        html.Div(
            style={
                "backgroundColor": "#e0e0e0", "padding": "8px 14px",
                "display": "flex", "alignItems": "center", "marginBottom": "20px",
            },
            children=[
                html.Label("Select Target:", style={"marginRight": "14px", "fontWeight": "bold", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="target-dropdown",
                    placeholder="Upload a CSV first",
                    style={"width": "220px", "fontSize": "13px"},
                    clearable=True,
                ),
            ],
        ),

        html.Div(
            style={"display": "flex", "gap": "30px", "marginBottom": "24px"},
            children=[
                html.Div(
                    style={"flex": 1},
                    children=[
                        dcc.RadioItems(
                            id="cat-radio",
                            inline=True,
                            style={"fontSize": "13px", "marginBottom": "6px"},
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px"},
                        ),
                        dcc.Graph(id="cat-bar-chart", style={"height": "340px"}),
                    ],
                ),
                html.Div(
                    style={"flex": 1},
                    children=[
                        dcc.Graph(id="corr-bar-chart", style={"height": "360px"}),
                    ],
                ),
            ],
        ),

        dcc.Checklist(
            id="feature-checklist",
            options=[],
            value=[],
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "16px", "fontSize": "13px"},
        ),
        html.Button(
            "Select/Deselect All",
            id="toggle-features-btn",
            n_clicks=0,
            style={"marginTop": "8px", "padding": "4px 12px", "fontSize": "12px"},
        ),

        html.Button(
            "Train",
            id="train-btn",
            n_clicks=0,
            style={"marginTop": "10px", "padding": "5px 22px", "fontSize": "13px"},
        ),
        html.Div(id="train-result", style={"marginTop": "8px", "fontSize": "13px"}),

        html.Hr(),

        html.Div(id="predict-label", style={"fontSize": "12px", "color": "gray", "marginBottom": "4px"}),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "10px"},
            children=[
                dcc.Input(
                    id="predict-input",
                    type="text",
                    debounce=False,
                    style={"width": "480px", "padding": "4px 8px", "fontSize": "13px"},
                ),
                html.Button(
                    "Predict",
                    id="predict-btn",
                    n_clicks=0,
                    style={"padding": "4px 16px", "fontSize": "13px"},
                ),
                html.Span(id="predict-result", style={"fontSize": "13px"}),
            ],
        ),
    ],
)


def parse_csv(contents):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df.drop_duplicates()


def coerce_numeric(series):
    as_string = series.astype("string").str.strip()
    return pd.to_numeric(as_string, errors="coerce")


def is_numeric_column(series, min_valid_ratio=0.95):
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = coerce_numeric(series)
    non_null_mask = series.notna()
    if non_null_mask.sum() == 0:
        return False
    valid_ratio = coerced[non_null_mask].notna().mean()
    return valid_ratio >= min_valid_ratio


def is_categorical_like_column(series):
    if not is_numeric_column(series):
        return True
    numeric = coerce_numeric(series).dropna()
    if numeric.empty:
        return True
    return (numeric % 1 == 0).all() and numeric.nunique() <= 10


def is_classification(series):
    non_null = series.dropna()
    if non_null.empty:
        return False

    # Explicit binary-label detection (supports numeric and string-like 0/1).
    numeric_cast = coerce_numeric(non_null)
    if not numeric_cast.isna().any():
        unique_numeric = set(numeric_cast.unique())
        if unique_numeric.issubset({0, 1}):
            return True

    # Non-numeric targets are usually labels.
    if pd.api.types.is_bool_dtype(series):
        return True
    if not is_numeric_column(series):
        n_unique = non_null.nunique()
        return n_unique <= 50 and (n_unique / len(non_null) <= 0.5)

    # Numeric targets are treated as classification only when they clearly
    # behave like label IDs; otherwise default to regression.
    non_null = coerce_numeric(non_null).dropna()
    if not (non_null % 1 == 0).all():
        return False

    n_unique = non_null.nunique()
    if n_unique <= 2:
        return True

    # Too many unique integer values likely indicates regression/ordinal score.
    if n_unique > 20:
        return False

    value_counts = non_null.value_counts()
    return (n_unique / len(non_null) <= 0.1) and value_counts.min() >= 3


@app.callback(
    Output("stored-data", "data"),
    Output("upload-info", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def store_data(contents, filename):
    df = parse_csv(contents)
    return df.to_json(date_format="iso", orient="split"), f"Loaded: {filename} — {df.shape[0]} rows, {df.shape[1]} columns"


@app.callback(
    Output("target-dropdown", "options"),
    Output("target-dropdown", "value"),
    Input("stored-data", "data"),
    prevent_initial_call=True,
)
def update_target_options(data):
    df = pd.read_json(io.StringIO(data), orient="split")
    cols = sorted(df.columns.tolist())
    return [{"label": c, "value": c} for c in cols], None


@app.callback(
    Output("cat-radio", "options"),
    Output("cat-radio", "value"),
    Input("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_radio(target, data):
    if not target or not data:
        return [], None
    df = pd.read_json(io.StringIO(data), orient="split")
    cat_cols = [c for c in df.columns if c != target and is_categorical_like_column(df[c])]
    options = [{"label": c, "value": c} for c in cat_cols]
    return options, cat_cols[0] if cat_cols else None


@app.callback(
    Output("cat-bar-chart", "figure"),
    Input("cat-radio", "value"),
    State("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_cat_chart(cat_col, target, data):
    if not cat_col or not target or not data:
        return {}
    df = pd.read_json(io.StringIO(data), orient="split")
    plot_df = df[[cat_col, target]].dropna().copy()
    if plot_df.empty:
        return {}

    target_is_classification = is_classification(plot_df[target])

    # Regression-style target: show average target by category.
    if not target_is_classification:
        plot_df[target] = coerce_numeric(plot_df[target])
        grouped = plot_df.groupby(cat_col)[target].mean().dropna().reset_index()
        fig = go.Figure(go.Bar(
            x=grouped[cat_col].astype(str),
            y=grouped[target].round(6),
            text=grouped[target].round(6),
            textposition="inside",
            marker_color="lightsteelblue",
        ))
        fig.update_layout(
            title=f"Average {target} by {cat_col}",
            xaxis_title=cat_col,
            yaxis_title=f"{target} (average)",
            margin={"t": 45, "b": 40, "l": 50, "r": 20},
        )
        return fig

    # Classification target: show target class distribution by category.
    dist = pd.crosstab(plot_df[cat_col], plot_df[target], normalize="index")
    fig = go.Figure()
    for cls in dist.columns:
        fig.add_trace(go.Bar(
            name=str(cls),
            x=dist.index.astype(str),
            y=dist[cls].values,
        ))
    fig.update_layout(
        title=f"{target} Distribution by {cat_col}",
        xaxis_title=cat_col,
        yaxis_title="Proportion",
        barmode="stack",
        margin={"t": 45, "b": 40, "l": 50, "r": 20},
    )
    return fig


@app.callback(
    Output("corr-bar-chart", "figure"),
    Input("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_corr_chart(target, data):
    if not target or not data:
        return {}
    df = pd.read_json(io.StringIO(data), orient="split")
    if is_classification(df[target]):
        return {}
    num_cols = [c for c in df.columns if c != target and is_numeric_column(df[c])]
    if not num_cols:
        return {}
    numeric_df = pd.DataFrame({c: coerce_numeric(df[c]) for c in num_cols})
    target_numeric = coerce_numeric(df[target])
    corr = numeric_df.corrwith(target_numeric).abs().dropna().sort_values(ascending=False)
    if corr.empty:
        return {}
    fig = go.Figure(go.Bar(
        x=corr.index.tolist(),
        y=corr.values.round(2),
        text=corr.values.round(2),
        textposition="outside",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title=f"Numerical Correlation Strength with {target}",
        xaxis_title="Numerical Variables",
        yaxis_title="Correlation Strength (Absolute Value)",
        yaxis_range=[0, corr.values.max() * 1.25 if len(corr) else 1],
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
    )
    return fig


@app.callback(
    Output("feature-checklist", "options"),
    Output("feature-checklist", "value"),
    Input("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_checkboxes(target, data):
    if not target or not data:
        return [], []
    df = pd.read_json(io.StringIO(data), orient="split")
    features = [c for c in df.columns if c != target]
    return [{"label": f, "value": f} for f in features], []


@app.callback(
    Output("feature-checklist", "value", allow_duplicate=True),
    Input("toggle-features-btn", "n_clicks"),
    State("feature-checklist", "options"),
    State("feature-checklist", "value"),
    prevent_initial_call=True,
)
def toggle_all_features(n_clicks, options, selected):
    if not options:
        return []

    all_values = [opt["value"] for opt in options]
    selected = selected or []

    if len(selected) == len(all_values):
        return []
    return all_values


@app.callback(
    Output("train-result", "children"),
    Output("predict-input", "placeholder"),
    Output("predict-label", "children"),
    Input("train-btn", "n_clicks"),
    State("stored-data", "data"),
    State("target-dropdown", "value"),
    State("feature-checklist", "value"),
    prevent_initial_call=True,
)
def train_model(n_clicks, data, target, features):
    if not data or not target:
        return "Please upload data and select a target.", "", ""
    if not features:
        return "Please select at least one feature.", "", ""

    df = pd.read_json(io.StringIO(data), orient="split")
    df = df.dropna(subset=[target])
    X, y = df[features].copy(), df[target].copy()

    num_cols = [f for f in features if is_numeric_column(df[f])]
    cat_cols = [f for f in features if f not in num_cols]
    for col in num_cols:
        X[col] = coerce_numeric(X[col])

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))

    task = "classification" if is_classification(y) else "regression"
    estimator = RandomForestClassifier(n_estimators=200, random_state=42) if task == "classification" else LinearRegression()
    if task == "regression":
        y = coerce_numeric(y)
        valid_rows = y.notna()
        X, y = X.loc[valid_rows], y.loc[valid_rows]

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("model", estimator),
    ])

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    model_store["pipeline"] = pipeline
    model_store["features"] = features
    model_store["df"] = df
    model_store["target"] = target
    model_store["task"] = task

    placeholder = ", ".join(features)
    label = f"Enter values in order (with commas in between): {placeholder}"
    if task == "classification":
        score = round(accuracy_score(y_test, y_pred), 4)
        metric_display = html.Span(f"Accuracy is: {score}")
    else:
        score = round(r2_score(y_test, y_pred), 4)
        metric_display = html.Span(["R", html.Sup("2"), f" score is: {score}"])
    return metric_display, placeholder, label


@app.callback(
    Output("predict-result", "children"),
    Input("predict-btn", "n_clicks"),
    State("predict-input", "value"),
    prevent_initial_call=True,
)
def predict(n_clicks, input_value):
    if not model_store["pipeline"]:
        return "Please train a model first."
    if not input_value:
        return "Please enter feature values."
    features = model_store["features"]
    df_ref = model_store["df"]
    target = model_store["target"]
    parts = [v.strip() for v in input_value.split(",")]
    if len(parts) != len(features):
        return f"Expected {len(features)} values in order: {', '.join(features)}"
    row = {}
    for feat, val in zip(features, parts):
        if is_numeric_column(df_ref[feat]):
            try:
                row[feat] = float(val)
            except ValueError:
                return f"'{val}' is not a valid number for {feat}"
        else:
            row[feat] = val
    pred = model_store["pipeline"].predict(pd.DataFrame([row]))[0]
    if model_store["task"] == "regression":
        return f"Predicted {target} is: {round(float(pred), 2)}"
    label = " (Yes)" if str(pred) == "1" else " (No)" if str(pred) == "0" else ""
    return f"Predicted {target} is: {pred}{label}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)

    # app.run(debug=True)
