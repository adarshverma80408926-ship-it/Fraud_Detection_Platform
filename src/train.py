import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    X_train = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv")
    )
    X_test = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_test.csv")
    )
    y_train = pd.read_csv(
        os.path.join(PROCESSED_DIR, "y_train.csv")
    ).squeeze()

    y_test = pd.read_csv(
        os.path.join(PROCESSED_DIR, "y_test.csv")
    ).squeeze()

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            y_prob
        ),
        "PR_AUC": average_precision_score(
            y_true,
            y_prob
        ),
        "False_Positive_Rate": fpr,
        "False_Negative_Rate": fnr
    }


def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])
    plt.tight_layout()

    path = os.path.join(
        REPORT_DIR,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()


def main():

    print("Loading processed data...")

    X_train, X_test, y_train, y_test = load_data()

    fraud_count = int(y_train.sum())
    legitimate_count = int(len(y_train) - fraud_count)

    scale_pos_weight = legitimate_count / fraud_count

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    models = {

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(f"TRAINING: {name}")
        print("=" * 70)

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_prob
        )

        metrics["Model"] = name
        results.append(metrics)

        model_filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".pkl"
        )

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                model_filename
            )
        )

        save_confusion_matrix(
            y_test,
            y_pred,
            name
        )

        print(f"Accuracy : {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall   : {metrics['Recall']:.4f}")
        print(f"F1 Score : {metrics['F1']:.4f}")
        print(f"ROC-AUC  : {metrics['ROC_AUC']:.4f}")
        print(f"PR-AUC   : {metrics['PR_AUC']:.4f}")
        print(
            f"False Positive Rate: "
            f"{metrics['False_Positive_Rate']:.4f}"
        )
        print(
            f"False Negative Rate: "
            f"{metrics['False_Negative_Rate']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "False_Positive_Rate",
            "False_Negative_Rate"
        ]
    ]

    results_df = results_df.sort_values(
        by=["PR_AUC", "F1"],
        ascending=False
    )

    results_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    best_model_name = results_df.iloc[0]["Model"]

    best_filename = (
        best_model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".pkl"
    )

    best_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            best_filename
        )
    )

    joblib.dump(
        best_model,
        os.path.join(
            MODEL_DIR,
            "best_model.pkl"
        )
    )

    with open(
        os.path.join(
            MODEL_DIR,
            "best_model_name.txt"
        ),
        "w"
    ) as file:
        file.write(best_model_name)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nBest model according to PR-AUC:")
    print(best_model_name)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
