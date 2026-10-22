# Project explained, folder by folder

## The idea in one paragraph
A person uploads a bank / Google Pay / PhonePe statement (PDF, Excel, CSV, JSON or Word). The backend reads it, stores the payments in a database, trains **that person's own** anomaly model from **their own** history, finds subscriptions, and the React dashboard shows: spending by month and shop, payments that look unusual ("Worth a look (Anomaly)"), and subscriptions. Uploading more later adds to the history and the model re-trains **only when the spending pattern really changed** (never because time passed).

## Folders
| Folder / file | What it is |
|---|---|
| `backend/` | FastAPI server. `main.py` = API routes (register, login, upload, dashboard). `services/analysis_service.py` reads the file and scores anomalies; `statement_extractor.py` reads PDFs; `persist_service.py` saves to / reads from the DB and builds the dashboard data; `retrain_service.py` decides when a user's model is retrained. `database/` = SQLAlchemy tables (`models.py`) and the DB connection (`db.py`). |
| `frontend/` | React + Vite + Recharts dashboard. `src/components/Dashboard.jsx` is the main screen. `npm run dev` for development; its `Dockerfile` + `nginx.conf` serve the built site in Docker. |
| `ml/` | The machine-learning code. `anomaly_model.py` = Isolation Forest + plain-language explanations; `drift_detection.py` = PSI + KS tests that tell whether spending changed; `subscription_model.py` = recurring-payment detector (any period); `categorizer.py` = guesses Food / Fuel / Bills... from the shop name; `insights_engine.py` = the sentences like "highest-spending month"; `generate_synthetic_data.py`, `preprocessing.py`, `features.py` = building the starter data set. `artifacts/` = saved models: the global starter model, and `user_models/` = one model file per user. |
| `mlops/` | `pipelines/train_with_mlflow.py` trains the global starter model and logs it to **MLflow**. `mlflow/mlflow.db` = MLflow's own record book (SQLite). `UPI-ML-Experiment/` = notebooks from the research phase. |
| `Dockerfile`, `docker-compose.yml` | Optional. Docker packages the backend, frontend, PostgreSQL and the MLflow screen so they start with one command: `docker compose up --build`. `docker/` is an empty placeholder. |
| `data/` | Starter and calibration data (not needed to run the app). |
| `tests/`, `notebooks/` | Placeholders / research notebooks. |

## Where each piece lives when you run it
* **Database** (users, payments, anomalies, subscriptions, model versions, drift reports): SQLite file `upi_app.db` by default, PostgreSQL if `DATABASE_URL` is set.
* **Per-user models**: files in `ml/artifacts/user_models/`, one set per user. The DB only records their versions.
* **Global starter model**: `ml/artifacts/isolation_forest.joblib`, used only while a user has fewer than 20 payments.

## What MLflow is, and what it is NOT doing here (honest)
MLflow is a notebook that records every training run: settings, scores, saved model. In this project it is used by `mlops/pipelines/train_with_mlflow.py` for the **global starter model** (compares a global-only model with a global+personal one and registers the better one). View it with:
`mlflow ui --backend-store-uri sqlite:///mlops/mlflow/mlflow.db` -> http://localhost:5000.
**The per-user retraining that happens when someone uploads is NOT logged to MLflow.** It is tracked in the database tables `model_versions` and `drift_reports` instead. Say this in your report; do not claim MLflow tracks every user retrain.

## What Docker is doing
Docker is only a packaging convenience. Running with `uvicorn` + `npm run dev` does not use Docker at all. Docker compose starts: PostgreSQL (database), backend (port 8000), frontend (port 5173 via nginx), MLflow UI (port 5000). It was not built or tested in the build environment.

## Evidently
`evidently` is listed in `requirements.txt` but no code imports it. Drift detection is our own PSI + KS code in `ml/drift_detection.py`. You can remove it from requirements, or leave it and not mention it.

## Flow of one upload
1. Upload -> `backend/main.py` `/upload/history`
2. File read by format -> `analysis_service.parse_and_validate` (PDF -> `statement_extractor`)
3. Category guessed from shop name if missing -> `ml/categorizer.py`
4. Payments saved -> DB `transactions`
5. Drift check against what the user's model was trained on -> `ml/drift_detection.py`
6. Retrain only if (no personal model yet) or (pattern changed and stayed changed) or (enough new payments) -> `retrain_service.py`; model file saved, version recorded in DB
7. Anomalies and subscriptions recomputed -> DB `anomaly_alerts`, `subscriptions`
8. Dashboard asks `/me/dashboard` -> built from the DB by `persist_service.get_dashboard`
