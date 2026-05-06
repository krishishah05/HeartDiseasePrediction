# CS301 Stage 3 App (Dash)

This app implements Stage 3 requirements from the project description:
- CSV upload with automated preprocessing
- Dynamic target selection
- Feature selection and correlation bar chart
- Training with a scikit-learn Pipeline (imputation + one-hot encoding)
- Manual input interface for real-time prediction
- Deployment-ready Dash app

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r stage_3/requirements.txt
```

3. Run the app:

```bash
python stage_3/app.py
```

## Suggested test dataset

Use:
- `data/heart_disease_dataset.csv`

## Deploy

### Google Cloud Run (free tier eligible)

1. Install and initialize Google Cloud CLI (`gcloud`).
2. Create/select a GCP project and enable billing (required even when usage stays within free-tier limits).
3. From `stage_3`, deploy:

```bash
gcloud run deploy heart-disease-app \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --max-instances 1 \
  --memory 512Mi \
  --cpu 1
```

4. Open the generated Cloud Run URL.

### Heroku / Render (optional)

You can also deploy this Dash app using a process file and runtime config if your class expects Heroku-like platforms.
