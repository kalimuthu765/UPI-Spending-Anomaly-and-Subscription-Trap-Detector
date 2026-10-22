import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")


# ============================================================
# LOAD DATASET 3 MODELS
# ============================================================

def load_dataset3_models():

    dataset3_dir = os.path.join(MODEL_DIR, "dataset3")

    models = {}

    models["isolation_forest"] = joblib.load(
        os.path.join(dataset3_dir, "isolation_forest.joblib")
    )

    models["lof"] = joblib.load(
        os.path.join(dataset3_dir, "lof.joblib")
    )

    models["one_class_svm"] = joblib.load(
        os.path.join(dataset3_dir, "one_class_svm.joblib")
    )

    models["hbos"] = joblib.load(
        os.path.join(dataset3_dir, "hbos.joblib")
    )

    models["preprocessor"] = joblib.load(
        os.path.join(dataset3_dir, "preprocessor.joblib")
    )

    return models


# ============================================================
# DATASET 3 FEATURE ENGINEERING
# ============================================================

def prepare_dataset3(data):

    df = data.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Convert time
    df["time"] = pd.to_datetime(
        df["time"],
        format="%H:%M",
        errors="coerce"
    )

    # Date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    # Time features
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute

    # Weekend
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Successful transaction
    df["is_successful_encoded"] = (
        df["is_successful"]
        .map({"Yes": 1, "No": 0})
        .fillna(0)
        .astype(int)
    )

    return df


# ============================================================
# DATASET 3 PREDICTION
# ============================================================

def predict_dataset3(data):

    models = load_dataset3_models()

    df = prepare_dataset3(data)

    numerical_features = [
        "amount",
        "year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "minute",
        "is_weekend",
        "is_successful_encoded"
    ]

    categorical_features = [
        "merchant_category",
        "payment_mode",
        "location_type"
    ]

    feature_columns = (
        numerical_features +
        categorical_features
    )

    X = df[feature_columns]

    # Apply SAME preprocessing used during training
    X_processed = models["preprocessor"].transform(X)

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    iso_prediction = models["isolation_forest"].predict(
        X_processed
    )

    iso_anomaly = (
        iso_prediction == -1
    ).astype(int)

    # --------------------------------------------------------
    # LOF
    # --------------------------------------------------------

    try:

        lof_prediction = models["lof"].predict(
            X_processed
        )

        lof_anomaly = (
            lof_prediction == -1
        ).astype(int)

    except Exception as e:

        print("LOF prediction unavailable:", e)

        lof_anomaly = np.zeros(
            len(df),
            dtype=int
        )

    # --------------------------------------------------------
    # One-Class SVM
    # --------------------------------------------------------

    svm_prediction = models["one_class_svm"].predict(
        X_processed
    )

    svm_anomaly = (
        svm_prediction == -1
    ).astype(int)

    # --------------------------------------------------------
    # HBOS
    # --------------------------------------------------------

    hbos_prediction = models["hbos"].predict(
        X_processed
    )

    hbos_anomaly = (
        hbos_prediction == 1
    ).astype(int)

    # --------------------------------------------------------
    # ENSEMBLE
    # --------------------------------------------------------

    votes = (
        iso_anomaly
        + lof_anomaly
        + svm_anomaly
        + hbos_anomaly
    )

    # At least 2 models must agree
    ensemble_anomaly = (
        votes >= 2
    ).astype(int)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = df.copy()

    result["isolation_forest_anomaly"] = iso_anomaly
    result["lof_anomaly"] = lof_anomaly
    result["one_class_svm_anomaly"] = svm_anomaly
    result["hbos_anomaly"] = hbos_anomaly

    result["anomaly_votes"] = votes
    result["ensemble_anomaly"] = ensemble_anomaly

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UPI ANOMALY DETECTION - DATASET 3")
    print("=" * 60)

    dataset_path = os.path.join(
        DATASET_DIR,
        "dataset3",
        "upi_transaction_insights_dataset.csv"
    )

    print("\nLoading dataset...")
    print(dataset_path)

    df = pd.read_csv(dataset_path)

    print("Dataset shape:", df.shape)

    result = predict_dataset3(df)

    print("\nPrediction completed!")

    print("\nAnomaly counts:")
    print(
        result["ensemble_anomaly"]
        .value_counts()
    )

    total_anomalies = (
        result["ensemble_anomaly"]
        .sum()
    )

    percentage = (
        total_anomalies /
        len(result)
    ) * 100

    print("\nTotal anomalies:", total_anomalies)
    print(
        "Anomaly percentage:",
        round(percentage, 2),
        "%"
    )

    print("\nTop anomalies:")

    print(
        result[
            result["ensemble_anomaly"] == 1
        ].head(20)
    )