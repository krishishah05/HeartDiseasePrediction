import base64
import io

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ALL
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

app = Dash(__name__)

# store trained pipeline between callbacks
model_store = {"pipeline": None, "features": [], "df": None, "target": None, "task": None}

app.layout = html.Div(style={"fontFamily": "Arial", "maxWidth": "950px", "margin": "auto", "padding": "20px"}, children=[

    html.H2("CS301 - Heart Disease Prediction App"),

    # step 1: upload
    html.H4("Step 1: Upload Dataset"),
    dcc.Upload(
        id="upload-data",
        children=html.Button("Upload CSV File"),
        multiple=False,
    ),
    html.Div(id="upload-info"),
    dcc.Store(id="stored-data"),

    html.Hr(),

    # step 2: target selection
    html.H4("Step 2: Select Target Column"),
    dcc.Dropdown(id="target-dropdown", placeholder="Upload a CSV first"),
    html.Div(id="target-info"),
    dcc.Graph(id="class-dist-chart"),

    html.Hr(),

    # step 3: feature selection + correlation
    html.H4("Step 3: Select Features"),
    dcc.Dropdown(id="feature-dropdown", multi=True, placeholder="Select features"),
    dcc.Graph(id="correlation-chart"),

    html.Hr(),

    # step 4: train
    html.H4("Step 4: Train Model"),
    dcc.Dropdown(id="model-dropdown", placeholder="Select target first to choose model"),
    html.Div(id="model-info"),
    html.Button("Train Model", id="train-btn", n_clicks=0),
    html.Div(id="train-output"),

    html.Hr(),

    # step 5: manual prediction
    html.H4("Step 5: Make a Prediction"),
    html.Div(id="prediction-form"),
    html.Div([
        html.Button("Predict", id="predict-btn", n_clicks=0),
    ]),
    html.Div(id="prediction-result"),
])


def parse_csv(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    df = df.drop_duplicates()
    return df


# parse and store uploaded csv
@app.callback(
    Output("stored-data", "data"),
    Output("upload-info", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def store_data(contents, filename):
    df = parse_csv(contents)
    info = html.P(f"Loaded {filename} — {df.shape[0]} rows, {df.shape[1]} columns")
    return df.to_json(date_format="iso", orient="split"), info


# populate target dropdown from uploaded data
@app.callback(
    Output("target-dropdown", "options"),
    Output("target-dropdown", "value"),
    Input("stored-data", "data"),
    prevent_initial_call=True,
)
def update_target_options(data):
    df = pd.read_json(io.StringIO(data), orient="split")
    options = [{"label": c, "value": c} for c in sorted(df.columns)]
    return options, None


# show class distribution or histogram depending on target type
@app.callback(
    Output("class-dist-chart", "figure"),
    Output("target-info", "children"),
    Input("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def show_class_distribution(target, data):
    if not target or not data:
        return {}, ""
    df = pd.read_json(io.StringIO(data), orient="split")
    series = df[target]

    if is_categorical_target(series):
        # classification: bar chart of class counts
        counts = series.value_counts().reset_index()
        counts.columns = [target, "Count"]
        counts[target] = counts[target].astype(str)
        fig = go.Figure(go.Bar(x=counts[target], y=counts["Count"], marker_color="steelblue"))
        fig.update_layout(title=f"Class Distribution: {target}", xaxis_title=target, yaxis_title="Count")
        info = html.P(f"Class counts: {series.value_counts().to_dict()}")
    else:
        # regression: histogram of value distribution
        fig = go.Figure(go.Histogram(x=series, marker_color="steelblue", nbinsx=30))
        fig.update_layout(title=f"Distribution of {target}", xaxis_title=target, yaxis_title="Count")
        info = html.P(f"Min: {round(series.min(), 2)}  |  Max: {round(series.max(), 2)}  |  Mean: {round(series.mean(), 2)}  |  Std: {round(series.std(), 2)}")

    return fig, info


# populate feature dropdown when target is selected
@app.callback(
    Output("feature-dropdown", "options"),
    Output("feature-dropdown", "value"),
    Input("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_feature_options(target, data):
    if not target or not data:
        return [], []
    df = pd.read_json(io.StringIO(data), orient="split")
    cols = sorted([c for c in df.columns if c != target])
    options = [{"label": c, "value": c} for c in cols]
    return options, cols


# show correlation bar chart when features are selected
@app.callback(
    Output("correlation-chart", "figure"),
    Input("feature-dropdown", "value"),
    State("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def show_correlation(features, target, data):
    if not features or not target or not data:
        return {}
    df = pd.read_json(io.StringIO(data), orient="split")
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    if not numeric_features:
        return {}
    target_series = df[target]
    if not pd.api.types.is_numeric_dtype(target_series):
        target_series = pd.factorize(target_series)[0]
    corr = df[numeric_features].corrwith(pd.Series(target_series, name=target))
    corr = corr.sort_values(key=abs, ascending=False)
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=corr_df["Feature"],
                y=corr_df["Correlation"],
                marker_color="teal",
            )
        ]
    )
    fig.update_layout(
        title=f"Feature Correlation with {target}",
        xaxis_title="Feature",
        yaxis_title="Correlation",
    )
    return fig


def is_categorical_target(series):
    if not pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return True
    unique_count = non_null.nunique()
    # treat low-cardinality integer-like numeric targets as categorical labels
    is_integer_like = (non_null % 1 == 0).all()
    return is_integer_like and unique_count <= 10


@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Output("model-info", "children"),
    Input("target-dropdown", "value"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_model_options(target, data):
    if not target or not data:
        return [], None, ""
    df = pd.read_json(io.StringIO(data), orient="split")
    target_series = df[target]
    if is_categorical_target(target_series):
        options = [
            {"label": "Decision Tree", "value": "decision_tree"},
            {"label": "Random Forest", "value": "random_forest"},
            {"label": "Logistic Regression", "value": "logistic_regression"},
        ]
        return options, "decision_tree", html.P("Detected categorical target: classification models enabled.")
    options = [{"label": "Linear Regression", "value": "linear_regression"}]
    return options, "linear_regression", html.P("Detected numerical target: regression model enabled.")


# train sklearn pipeline when button is clicked
@app.callback(
    Output("train-output", "children"),
    Input("train-btn", "n_clicks"),
    State("stored-data", "data"),
    State("target-dropdown", "value"),
    State("feature-dropdown", "value"),
    State("model-dropdown", "value"),
    prevent_initial_call=True,
)
def train_model(n_clicks, data, target, features, selected_model):
    if not data or not target or not features or not selected_model:
        return html.P("Please upload data and select target/features first.")

    df = pd.read_json(io.StringIO(data), orient="split")
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]
    is_classification = is_categorical_target(y)

    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    cat_cols = [f for f in features if f not in num_cols]

    # build preprocessing pipeline (imputer + encoding)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    if is_classification:
        if selected_model == "random_forest":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif selected_model == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = DecisionTreeClassifier(max_depth=7, criterion="entropy", random_state=42)
    else:
        model = LinearRegression()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    split_kwargs = {"test_size": 0.2, "random_state": 42}
    if is_classification:
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    model_store["pipeline"] = pipeline
    model_store["features"] = features
    model_store["df"] = df
    model_store["target"] = target
    model_store["task"] = "classification" if is_classification else "regression"

    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        return html.Div([
            html.P(f"Model: {selected_model.replace('_', ' ').title()}"),
            html.P(f"Accuracy : {round(acc, 4)}"),
            html.P(f"F1-Score : {round(f1, 4)}"),
            html.P("Model trained. Fill in Step 5 to make a prediction."),
        ])

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return html.Div([
        html.P("Model: Linear Regression"),
        html.P(f"MAE : {round(mae, 4)}"),
        html.P(f"R2  : {round(r2, 4)}"),
        html.P("Model trained. Fill in Step 5 to make a prediction."),
    ])


# generate manual input form after training
@app.callback(
    Output("prediction-form", "children"),
    Input("train-output", "children"),
    prevent_initial_call=True,
)
def generate_prediction_form(train_output):
    if not model_store["features"] or model_store["df"] is None:
        return ""
    df = model_store["df"]
    inputs = []
    for f in model_store["features"]:
        if pd.api.types.is_numeric_dtype(df[f]):
            default = float(df[f].median())
            inputs.append(html.Div([
                html.Label(f, style={"fontWeight": "bold"}),
                dcc.Input(id={"type": "pred-input", "index": f},
                          type="number", value=default,
                          style={"margin": "5px", "width": "200px"}),
            ], style={"marginBottom": "8px"}))
        else:
            options = sorted(df[f].dropna().astype(str).unique().tolist())
            inputs.append(html.Div([
                html.Label(f, style={"fontWeight": "bold"}),
                dcc.Dropdown(id={"type": "pred-input", "index": f},
                             options=[{"label": o, "value": o} for o in options],
                             value=options[0],
                             style={"margin": "5px", "width": "250px"}),
            ], style={"marginBottom": "8px"}))
    return inputs


# run prediction from manual inputs
@app.callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    State({"type": "pred-input", "index": ALL}, "value"),
    State({"type": "pred-input", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def make_prediction(n_clicks, values, ids):
    if not model_store["pipeline"] or not values:
        return ""
    feature_values = {id_["index"]: val for id_, val in zip(ids, values)}
    sample = pd.DataFrame([feature_values])
    prediction = model_store["pipeline"].predict(sample)[0]
    target = model_store.get("target", "Target")
    if model_store.get("task") == "classification":
        return html.Div([
            html.H5(f"Prediction: {target} = {prediction}"),
        ])
    return html.Div([
        html.H5(f"Prediction: {target} = {round(float(prediction), 4)}"),
    ])


if __name__ == "__main__":
    app.run(debug=True)
