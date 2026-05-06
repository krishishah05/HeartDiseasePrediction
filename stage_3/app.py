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


def is_classification(series):
    non_null = series.dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return (non_null % 1 == 0).all() and non_null.nunique() <= 10


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
    num_cols = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    return [{"label": c, "value": c} for c in num_cols], None


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
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != target]
    if not cat_cols:
        cat_cols = [c for c in df.columns if c != target and df[c].nunique() <= 10]
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
    grouped = df.groupby(cat_col)[target].mean().reset_index()
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
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    if not num_cols:
        return {}
    corr = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
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
    X, y = df[features], df[target]

    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    cat_cols = [f for f in features if f not in num_cols]

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
        if pd.api.types.is_numeric_dtype(df_ref[feat]):
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
