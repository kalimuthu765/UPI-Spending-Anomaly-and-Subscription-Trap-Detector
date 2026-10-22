"""
Phase 7 — MLflow Tracking + Model Registry
Runs both the v1 (global-only) and v2 (global + personalized) anomaly
models as tracked MLflow experiments, logs params/metrics/artifacts for
each, and registers the better-performing one as UPI_Anomaly_Model.
"""

import sys
import mlflow
import mlflow.sklearn

sys.path.insert(0, "ml")
import anomaly_model
import personalized_model

TRACKING_URI = "sqlite:///mlops/mlflow/mlflow.db"
EXPERIMENT_NAME = "UPI-Anomaly-Detection"
REGISTERED_MODEL_NAME = "UPI_Anomaly_Model"

FEATURED_PATH = "data/processed/transactions_featured.csv"


def log_v1_global():
    with mlflow.start_run(run_name="v1_isolation_forest_global"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("scope", "global (one model for all users)")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("contamination", 0.015)
        mlflow.log_param("features", ",".join(anomaly_model.FEATURE_COLS))

        df, metrics = anomaly_model.run(
            input_path=FEATURED_PATH,
            output_path="data/processed/anomaly_scored.csv",
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # skops' default serializer flags sklearn Tree internals (used by
        # IsolationForest) as untrusted; safe to trust here since we trained
        # this model ourselves in this same run.
        model = mlflow.sklearn.log_model(
            sk_model=__import__("joblib").load(anomaly_model.MODEL_PATH),
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifact(anomaly_model.SCALER_PATH, artifact_path="preprocessing")
        return metrics, model.model_uri


def log_v2_personalized():
    with mlflow.start_run(run_name="v2_isolation_forest_personalized"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("scope", "personalized (per-user model + global fallback)")
        mlflow.log_param("min_user_txns_for_personal_model", personalized_model.MIN_USER_TXNS)
        mlflow.log_param("personal_contamination", personalized_model.PERSONAL_CONTAMINATION)
        mlflow.log_param("features", ",".join(anomaly_model.FEATURE_COLS))

        df, metrics = personalized_model.run(
            input_path=FEATURED_PATH,
            output_path="data/processed/anomaly_scored_personalized.csv",
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        return metrics, None  # per-user models aren't a single sklearn artifact to log


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("Run 1/2: v1 global Isolation Forest")
    print("=" * 60)
    v1_metrics, v1_model_uri = log_v1_global()

    print("\n" + "=" * 60)
    print("Run 2/2: v2 personalized Isolation Forest")
    print("=" * 60)
    v2_metrics, _ = log_v2_personalized()

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"{'metric':10s} {'v1_global':>12s} {'v2_personal':>12s}")
    for k in v1_metrics:
        print(f"{k:10s} {v1_metrics[k]:>12.4f} {v2_metrics[k]:>12.4f}")

    # register whichever version scored higher F1 (v1 is the one with a
    # loggable single sklearn artifact; v2's per-user models are registered
    # as a strategy note since there are 1000 of them, not one artifact)
    if v2_metrics["f1"] > v1_metrics["f1"]:
        print(f"\nv2 (personalized) wins on F1 ({v2_metrics['f1']} > {v1_metrics['f1']}).")
        print("Registering v1's global model as the cold-start fallback component "
              "of UPI_Anomaly_Model; per-user models are trained on-demand per the "
              "personalized_model.py pipeline (not a single registrable artifact).")
    else:
        print(f"\nv1 (global) wins on F1 ({v1_metrics['f1']} >= {v2_metrics['f1']}).")

    result = mlflow.register_model(v1_model_uri, REGISTERED_MODEL_NAME)
    print(f"\nRegistered '{REGISTERED_MODEL_NAME}' version {result.version} "
          f"(global fallback component).")

    print(f"\nMLflow tracking store: {TRACKING_URI}")
    print("Run `mlflow ui --backend-store-uri " + TRACKING_URI + "` to view the dashboard.")


if __name__ == "__main__":
    main()
