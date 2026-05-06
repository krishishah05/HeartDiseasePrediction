import base64
import io

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ALL
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

app = Dash(__name__)

# store trained pipeline between callbacks
model_store = {"pipeline": None, "features": [], "df": None}

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
    options = [{"label": c, "value": c} for c in df.columns]
    return options, None


# show class distribution when target is selected
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
    counts = df[target].value_counts().reset_index()
    counts.columns = [target, "Count"]
    fig = px.bar(counts, x=target, y="Count", title=f"Class Distribution: {target}",
                 color_discrete_sequence=["steelblue"])
    info = html.P(f"Class counts: {df[target].value_counts().to_dict()}")
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
    cols = [c for c in df.columns if c != target]
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
    fig = px.bar(corr_df, x="Feature", y="Correlation",
                 title=f"Feature Correlation with {target}",
                 color_discrete_sequence=["teal"])
    return fig


# train sklearn pipeline when button is clicked
@app.callback(
    Output("train-output", "children"),
    Input("train-btn", "n_clicks"),
    State("stored-data", "data"),
    State("target-dropdown", "value"),
    State("feature-dropdown", "value"),
    prevent_initial_call=True,
)
def train_model(n_clicks, data, target, features):
    if not data or not target or not features:
        return html.P("Please upload data and select target and features first.")

    df = pd.read_json(io.StringIO(data), orient="split")
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]

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

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=7, criterion="entropy", random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    model_store["pipeline"] = pipeline
    model_store["features"] = features
    model_store["df"] = df

    return html.Div([
        html.P(f"Accuracy : {round(acc, 4)}"),
        html.P(f"F1-Score : {round(f1, 4)}"),
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
    label = "Heart Disease" if prediction == 1 else "No Heart Disease"
    return html.Div([
        html.H5(f"Prediction: {label}"),
        html.P(f"Raw value: {prediction}"),
    ])


if __name__ == "__main__":
    app.run(debug=True)
