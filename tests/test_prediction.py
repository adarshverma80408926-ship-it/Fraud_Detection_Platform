import pandas as pd

from src.predict import predict_transaction, BASE_FEATURES


DATA_PATH = "data/raw/creditcard.csv"


def make_transaction(row):
    return {
        feature: float(row[feature])
        for feature in BASE_FEATURES
    }


def main():
    df = pd.read_csv(DATA_PATH)

    legitimate = df[df["Class"] == 0].iloc[0]
    fraud = df[df["Class"] == 1].iloc[0]

    legitimate_result = predict_transaction(
        make_transaction(legitimate)
    )

    fraud_result = predict_transaction(
        make_transaction(fraud)
    )

    print("=" * 60)
    print("LEGITIMATE TRANSACTION TEST")
    print("=" * 60)
    print(legitimate_result)

    print("\n" + "=" * 60)
    print("FRAUD TRANSACTION TEST")
    print("=" * 60)
    print(fraud_result)


if __name__ == "__main__":
    main()
