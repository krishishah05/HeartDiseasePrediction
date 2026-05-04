# CS301 Stage 3 App

This app implements Stage 3 requirements from the project description:
- CSV upload with automated preprocessing
- Dynamic target selection
- Feature selection and correlation bar chart
- Training with a scikit-learn Pipeline (imputation + one-hot encoding)
- Manual input interface for real-time prediction
- Deployment-ready Streamlit app

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r stage_3/requirements.txt
```

3. Run the app:

```bash
streamlit run stage_3/app.py
```

## Suggested test dataset

Use:
- `data/heart_disease_dataset.csv`

## Deploy

### Streamlit Community Cloud

1. Push the project to GitHub.
2. In Streamlit Cloud, create a new app.
3. Set main file path to `stage_3/app.py`.
4. Make sure dependency file includes `stage_3/requirements.txt` packages.
5. Deploy and copy the public app URL for your report.

### Heroku / Render (optional)

You can also deploy Streamlit using a process file and runtime config if your class expects Heroku-like platforms.
