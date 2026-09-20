import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    return df


def feature_engineering(df):
    df = df.copy()

    # Convert transaction time into hour of day
    df["Hour"] = ((df["Time"] % 86400) // 3600).astype(int)

    # Log transform transaction amount
    df["LogAmount"] = __import__("numpy").log1p(df["Amount"])

    return df


def split_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = RobustScaler()

    scale_columns = ["Time", "Amount", "LogAmount"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])

    return X_train, X_test, scaler


def save_data(X_train, X_test, y_train, y_test, scaler):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train.to_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"),
        index=False
    )

    X_test.to_csv(
        os.path.join(PROCESSED_DIR, "X_test.csv"),
        index=False
    )

    y_train.to_csv(
        os.path.join(PROCESSED_DIR, "y_train.csv"),
        index=False
    )

    y_test.to_csv(
        os.path.join(PROCESSED_DIR, "y_test.csv"),
        index=False
    )

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "feature_scaler.pkl")
    )


def main():
    df = load_data()

    print(f"Original shape: {df.shape}")

    df = feature_engineering(df)

    print(f"Shape after feature engineering: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Testing fraud cases: {y_test.sum()}")

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
