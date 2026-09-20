import json
import os
import sys

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
THRESHOLD_PATH = "models/optimal_threshold.txt"

BASE_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

MODEL_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Hour",
    "LogAmount"
]


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Threshold not found: {THRESHOLD_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())

    return model, scaler, threshold


def prepare_transaction(transaction, scaler):
    missing = [
        feature
        for feature in BASE_FEATURES
        if feature not in transaction
    ]

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing)
        )

    df = pd.DataFrame(
        [[transaction[feature] for feature in BASE_FEATURES]],
        columns=BASE_FEATURES
    )

    # Same feature engineering used during training
    df["Hour"] = (
        (df["Time"] % 86400) // 3600
    ).astype(int)

    df["LogAmount"] = np.log1p(
        df["Amount"]
    )

    # Same scaling used during training
    scale_columns = [
        "Time",
        "Amount",
        "LogAmount"
    ]

    df[scale_columns] = scaler.transform(
        df[scale_columns]
    )

    return df[MODEL_FEATURES]


def predict_transaction(transaction):
    model, scaler, threshold = load_artifacts()

    features = prepare_transaction(transaction, scaler)

    fraud_probability = float(
        model.predict_proba(features)[0][1]
    )

    is_fraud = (
        fraud_probability >= threshold
    )

    decision = (
        "FRAUD"
        if is_fraud
        else "LEGITIMATE"
    )

    return {
        "fraud_probability": round(
            fraud_probability,
            6
        ),
        "threshold": round(
            threshold,
            6
        ),
        "decision": decision
    }


def load_transaction_from_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Transaction file not found: {path}"
        )

    with open(path, "r") as file:
        return json.load(file)


def smoke_test():
    print("=" * 60)
    print("FRAUD DETECTION MODEL - SMOKE TEST")
    print("=" * 60)

    raw_path = "data/raw/creditcard.csv"

    sample = pd.read_csv(
        raw_path,
        nrows=1
    ).iloc[0]

    transaction = {
        feature: float(sample[feature])
        for feature in BASE_FEATURES
    }

    result = predict_transaction(
        transaction
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )


def main():
    if len(sys.argv) == 1:
        smoke_test()
        return

    json_path = sys.argv[1]

    transaction = load_transaction_from_json(
        json_path
    )

    result = predict_transaction(
        transaction
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )


if __name__ == "__main__":
    main()
