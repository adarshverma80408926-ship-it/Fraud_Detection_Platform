import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
VALIDATION_SIZE = 0.20


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


def create_model(y):
    fraud_count = int(y.sum())
    legitimate_count = int(len(y) - fraud_count)

    scale_pos_weight = legitimate_count / fraud_count

    return XGBClassifier(
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


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "False_Positive_Rate": fpr,
        "False_Negative_Rate": fnr
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading processed data...")

    X_train, X_test, y_train, y_test = load_data()

    print(f"Full training data: {X_train.shape}")
    print(f"Final test data: {X_test.shape}")

    # ---------------------------------------------------------
    # 1. Create validation split from training data
    # ---------------------------------------------------------

    X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    print(f"Sub-training data: {X_subtrain.shape}")
    print(f"Validation data: {X_validation.shape}")

    # ---------------------------------------------------------
    # 2. Train temporary model for threshold optimization
    # ---------------------------------------------------------

    print("\nTraining validation model...")

    validation_model = create_model(y_subtrain)

    validation_model.fit(
        X_subtrain,
        y_subtrain
    )

    validation_probabilities = validation_model.predict_proba(
        X_validation
    )[:, 1]

    # ---------------------------------------------------------
    # 3. Test multiple thresholds on validation data
    # ---------------------------------------------------------

    thresholds = np.arange(
        0.05,
        1.00,
        0.01
    )

    validation_results = []

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            float(threshold)
        )

        validation_results.append(metrics)

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "threshold_validation_results.csv"
        ),
        index=False
    )

    # ---------------------------------------------------------
    # 4. Select threshold by validation F1
    # ---------------------------------------------------------

    best_row = validation_df.loc[
        validation_df["F1"].idxmax()
    ]

    best_threshold = float(
        best_row["Threshold"]
    )

    print("\n" + "=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold : {best_threshold:.2f}"
    )
    print(
        f"Accuracy  : {best_row['Accuracy']:.4f}"
    )
    print(
        f"Precision : {best_row['Precision']:.4f}"
    )
    print(
        f"Recall    : {best_row['Recall']:.4f}"
    )
    print(
        f"F1        : {best_row['F1']:.4f}"
    )
    print(
        f"FPR       : {best_row['False_Positive_Rate']:.4f}"
    )
    print(
        f"FNR       : {best_row['False_Negative_Rate']:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Retrain final XGBoost on all training data
    # ---------------------------------------------------------

    print("\nRetraining final XGBoost on all training data...")

    final_model = create_model(y_train)

    final_model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # 6. Evaluate only once on untouched test data
    # ---------------------------------------------------------

    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions
    ).ravel()

    test_fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0
    )

    test_fnr = (
        fn / (fn + tp)
        if (fn + tp)
        else 0
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Threshold            : {best_threshold:.2f}"
    )
    print(
        f"Accuracy             : {test_accuracy:.4f}"
    )
    print(
        f"Precision            : {test_precision:.4f}"
    )
    print(
        f"Recall               : {test_recall:.4f}"
    )
    print(
        f"F1 Score             : {test_f1:.4f}"
    )
    print(
        f"ROC-AUC              : {test_roc_auc:.4f}"
    )
    print(
        f"PR-AUC               : {test_pr_auc:.4f}"
    )
    print(
        f"False Positive Rate  : {test_fpr:.4f}"
    )
    print(
        f"False Negative Rate  : {test_fnr:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    # ---------------------------------------------------------
    # 7. Save final model and threshold
    # ---------------------------------------------------------

    joblib.dump(
        final_model,
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
        file.write("XGBoost")

    with open(
        os.path.join(
            MODEL_DIR,
            "optimal_threshold.txt"
        ),
        "w"
    ) as file:
        file.write(
            str(best_threshold)
        )

    final_results = pd.DataFrame(
        [{
            "Model": "XGBoost",
            "Threshold": best_threshold,
            "Accuracy": test_accuracy,
            "Precision": test_precision,
            "Recall": test_recall,
            "F1": test_f1,
            "ROC_AUC": test_roc_auc,
            "PR_AUC": test_pr_auc,
            "False_Positive_Rate": test_fpr,
            "False_Negative_Rate": test_fnr
        }]
    )

    final_results.to_csv(
        os.path.join(
            REPORT_DIR,
            "final_model_results.csv"
        ),
        index=False
    )

    print("\nFinal model and threshold saved successfully.")


if __name__ == "__main__":
    main()
