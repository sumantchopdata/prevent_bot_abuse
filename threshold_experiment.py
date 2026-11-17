# ML Flow experiments for Preventing Bot Abuse - Threshold analysis
#%%
from pathlib import Path
import argparse
import hashlib
import re

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("claude_datasets")
ARTIFACT_DIR = Path("artifacts/threshold_experiment")
CACHE_DIR = Path("artifacts/embeddings")

TRAIN_FILE = DATA_DIR / "combined_train_v4_balanced.csv"
EVAL_FILE = DATA_DIR / "combined_eval_v4_balanced.csv"
TEST_FILE = DATA_DIR / "combined_test_v4_balanced.csv"

TEXT_CANDIDATES = ["text"]
LABEL_COL = "labels"
CATEGORY_COL = "category"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Winning configuration from the regularisation experiment
MODEL_C = 10
CLASS_WEIGHT = None
MAX_ITER = 2000
RANDOM_STATE = 42

THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
]

EXPERIMENT_NAME = "Prevent Bot Abuse - Threshold Analysis"
TRACKING_URI = "sqlite:///mlflow.db"

# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    df = pd.read_csv(path)

    text_col = next(
        (col for col in TEXT_CANDIDATES if col in df.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            f"{path} must contain one of these text columns: "
            f"{TEXT_CANDIDATES}"
        )

    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain '{LABEL_COL}'")

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if not set(df[LABEL_COL].unique()).issubset({0, 1}):
        raise ValueError(
            f"{path}: labels must be 0 (abuse) or 1 (genuine)"
        )

    return df, text_col

# ============================================================
# EMBEDDING CACHE
# ============================================================

def get_embeddings(texts):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique cache key based on the text and model
    content = "\n".join(texts)
    cache_hash = hashlib.sha256(
        (EMBEDDING_MODEL_NAME + content).encode("utf-8")
    ).hexdigest()[:16]

    cache_path = CACHE_DIR / f"{cache_hash}.npz"

    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)["embeddings"]

    print("Generating sentence embeddings...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.savez_compressed(
        cache_path,
        embeddings=embeddings
    )

    return embeddings

#%%
# ============================================================
# MODEL
# ============================================================

def create_model():
    return LogisticRegression(
        C=MODEL_C,
        class_weight=CLASS_WEIGHT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE
    )


def get_abuse_probabilities(model, X):
    """
    Return probability of class 0 (abuse).

    Do not assume the position of class 0 in predict_proba().
    """

    probabilities = model.predict_proba(X)
    abuse_index = list(model.classes_).index(0)

    return probabilities[:, abuse_index]


def predict_using_threshold(abuse_probabilities, threshold):
    """
    Classify as abuse (0) when P(abuse) >= threshold.
    Otherwise classify as genuine (1).
    """

    return np.where(
        abuse_probabilities >= threshold,
        0,
        1
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, abuse_probabilities):
    # Class 0 = abuse
    # Class 1 = genuine

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),

        "abuse_precision": precision[0],
        "abuse_recall": recall[0],
        "abuse_f1": f1[0],

        "genuine_precision": precision[1],
        "genuine_recall": recall[1],
        "genuine_f1": f1[1],

        "abuse_roc_auc": roc_auc_score(
            (y_true == 0).astype(int),
            abuse_probabilities
        ),
        "abuse_pr_auc": average_precision_score(
            (y_true == 0).astype(int),
            abuse_probabilities
        )
    }

    return metrics


def log_category_metrics(y_true, y_pred, categories):
    if categories is None:
        return

    results = {}

    category_series = pd.Series(categories).reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for category in category_series.dropna().unique():

        mask = category_series == category

        category_true = y_true[mask.to_numpy()]
        category_pred = y_pred[mask.to_numpy()]

        safe_category = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            str(category)
        )

        results[f"category_{safe_category}_n"] = len(category_true)

        results[f"category_{safe_category}_accuracy"] = (
            accuracy_score(category_true, category_pred)
        )

        # Recall for each class, where that class exists
        if 0 in category_true:
            results[f"category_{safe_category}_abuse_recall"] = (
                recall_score_for_class(
                    category_true, category_pred, 0
                )
            )

        if 1 in category_true:
            results[f"category_{safe_category}_genuine_recall"] = (
                recall_score_for_class(
                    category_true, category_pred, 1
                )
            )

    mlflow.log_metrics(results)


def recall_score_for_class(y_true, y_pred, class_label):
    _, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[class_label],
        zero_division=0
    )
    return recall[0]


# ============================================================
# PREDICTION EXPORT
# ============================================================

def create_predictions_dataframe(
    df,
    text_col,
    y_true,
    abuse_probabilities,
    y_pred
):
    predictions = pd.DataFrame({
        "text": df[text_col].values,
        "actual_label": np.asarray(y_true),
        "predicted_label": np.asarray(y_pred),
        "abuse_probability": abuse_probabilities,
        "genuine_probability": 1 - abuse_probabilities
    })

    if CATEGORY_COL in df.columns:
        predictions["category"] = df[CATEGORY_COL].values

    predictions["correct"] = (
        predictions["actual_label"]
        == predictions["predicted_label"]
    )

    return predictions


# ============================================================
# PLOTS
# ============================================================

def create_threshold_plots(summary_df, y_true, abuse_probabilities):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(
        (np.asarray(y_true) == 0).astype(int),
        abuse_probabilities
    )

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Abuse Recall")
    plt.ylabel("Abuse Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()

    pr_path = ARTIFACT_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # Metrics against threshold
    plt.figure(figsize=(9, 6))

    for metric in [
        "abuse_precision",
        "abuse_recall",
        "macro_f1",
        "accuracy"
    ]:
        plt.plot(
            summary_df["threshold"],
            summary_df[metric],
            marker="o",
            label=metric
        )

    plt.xlabel("Abuse Probability Threshold")
    plt.ylabel("Score")
    plt.title("Model Performance at Different Thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    threshold_path = ARTIFACT_DIR / "threshold_metrics.png"
    plt.savefig(threshold_path, dpi=150)
    plt.close()

    return pr_path, threshold_path

#%%
# ============================================================
# STAGE 1: THRESHOLD SWEEP ON EVALUATION SET
# ============================================================

def run_threshold_sweep():
    print("\nLoading datasets...")

    train_df, train_text_col = load_dataset(TRAIN_FILE)
    eval_df, eval_text_col = load_dataset(EVAL_FILE)

    X_train = get_embeddings(train_df[train_text_col].tolist())
    X_eval = get_embeddings(eval_df[eval_text_col].tolist())

    y_train = train_df[LABEL_COL].to_numpy()
    y_eval = eval_df[LABEL_COL].to_numpy()

    print("\nTraining winning Logistic Regression configuration...")
    print(f"C = {MODEL_C}")
    print(f"class_weight = {CLASS_WEIGHT}")

    model = create_model()
    model.fit(X_train, y_train)

    abuse_probabilities = get_abuse_probabilities(
        model, X_eval
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    summary_results = []
    all_predictions = []

    for threshold in THRESHOLDS:

        print(f"\nEvaluating threshold: {threshold:.2f}")

        y_pred = predict_using_threshold(
            abuse_probabilities,
            threshold
        )

        metrics = calculate_metrics(
            y_eval,
            y_pred,
            abuse_probabilities
        )

        summary_results.append({
            "threshold": threshold,
            **metrics
        })

        predictions_df = create_predictions_dataframe(
            eval_df,
            eval_text_col,
            y_eval,
            abuse_probabilities,
            y_pred
        )

        predictions_df[f"predicted_at_{threshold:.2f}"] = y_pred
        all_predictions.append(predictions_df)

        with mlflow.start_run(
            run_name=f"threshold_{threshold:.2f}"
        ):
            mlflow.log_params({
                "model": "LogisticRegression",
                "C": MODEL_C,
                "class_weight": "None",
                "embedding_model": EMBEDDING_MODEL_NAME,
                "threshold": threshold,
                "training_data": "train",
                "evaluation_data": "eval"
            })

            mlflow.log_metrics(metrics)

            categories = (
                eval_df[CATEGORY_COL]
                if CATEGORY_COL in eval_df.columns
                else None
            )

            log_category_metrics(
                y_eval,
                y_pred,
                categories
            )

            print(
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Abuse precision: {metrics['abuse_precision']:.4f} | "
                f"Abuse recall: {metrics['abuse_recall']:.4f}"
            )

    summary_df = pd.DataFrame(summary_results)

    summary_path = ARTIFACT_DIR / "threshold_summary.csv"
    predictions_path = ARTIFACT_DIR / "eval_predictions.csv"

    summary_df.to_csv(summary_path, index=False)

    # Save one copy of the evaluation predictions, including
    # probabilities and predictions at every threshold.
    combined_predictions = all_predictions[0].copy()

    for threshold, prediction_df in zip(
        THRESHOLDS,
        all_predictions
    ):
        column = f"predicted_at_{threshold:.2f}"
        if column not in combined_predictions.columns:
            combined_predictions[column] = (
                prediction_df[column].values
            )

    combined_predictions.to_csv(
        predictions_path,
        index=False
    )

    pr_path, threshold_path = create_threshold_plots(
        summary_df,
        y_eval,
        abuse_probabilities
    )

    # Log the overall sweep artifacts in a separate run
    with mlflow.start_run(run_name="threshold_sweep_summary"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": MODEL_C,
            "class_weight": "None",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "thresholds_tested": str(THRESHOLDS)
        })

        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(pr_path))
        mlflow.log_artifact(str(threshold_path))

    print("\nThreshold sweep complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"Predictions saved to: {predictions_path}")

    print("\nThreshold comparison:")
    print(
        summary_df[
            [
                "threshold",
                "accuracy",
                "macro_f1",
                "abuse_precision",
                "abuse_recall",
                "abuse_f1"
            ]
        ].to_string(index=False)
    )

#%%
# ============================================================
# STAGE 2: FINAL TRAIN+EVAL -> TEST EVALUATION
# ============================================================

def run_final_test(threshold):
    print("\nLoading datasets...")

    train_df, train_text_col = load_dataset(TRAIN_FILE)
    eval_df, eval_text_col = load_dataset(EVAL_FILE)
    test_df, test_text_col = load_dataset(TEST_FILE)

    # Combine train and evaluation data
    train_eval_df = pd.concat(
        [train_df, eval_df],
        ignore_index=True
    )

    train_eval_text_col = (
        train_text_col
        if train_text_col == eval_text_col
        else train_text_col
    )

    X_train_eval = get_embeddings(
        train_eval_df[train_eval_text_col].tolist()
    )

    X_test = get_embeddings(test_df[test_text_col].tolist())

    y_train_eval = train_eval_df[LABEL_COL].to_numpy()
    y_test = test_df[LABEL_COL].to_numpy()

    print("\nTraining on train + eval...")
    model = create_model()
    model.fit(X_train_eval, y_train_eval)

    abuse_probabilities = get_abuse_probabilities(
        model, X_test
    )

    y_pred = predict_using_threshold(
        abuse_probabilities,
        threshold
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        abuse_probabilities
    )

    predictions_df = create_predictions_dataframe(
        test_df,
        test_text_col,
        y_test,
        abuse_probabilities,
        y_pred
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = ARTIFACT_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["abuse", "genuine"],
        zero_division=0
    )

    report_path = ARTIFACT_DIR / "test_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_path = ARTIFACT_DIR / "test_confusion_matrix.csv"

    pd.DataFrame(
        cm,
        index=["actual_abuse", "actual_genuine"],
        columns=["predicted_abuse", "predicted_genuine"]
    ).to_csv(cm_path)

    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": MODEL_C,
            "class_weight": "None",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "threshold": threshold,
            "training_data": "train + eval",
            "test_data": "test"
        })

        mlflow.log_metrics(metrics)

        categories = (
            test_df[CATEGORY_COL]
            if CATEGORY_COL in test_df.columns
            else None
        )

        log_category_metrics(
            y_test,
            y_pred,
            categories
        )

        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_regression"
        )

    # Save a local copy of the final model
    joblib.dump(
        model,
        ARTIFACT_DIR / "final_logistic_regression.joblib"
    )

    print("\nFinal test evaluation complete.")
    print(f"Threshold used: {threshold:.2f}")
    print("\nClassification report:")
    print(report)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nTest predictions saved to: {predictions_path}")

#%%
# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Threshold experiments for the bot-abuse classifier"
    )

    parser.add_argument(
        "--stage",
        choices=["sweep", "final"],
        default="sweep",
        help="'sweep' evaluates thresholds on eval; "
             "'final' evaluates one selected threshold on test."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Selected abuse probability threshold for final test."
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.stage == "sweep":
        run_threshold_sweep()

    elif args.stage == "final":
        if args.threshold is None:
            raise ValueError(
                "Please supply a threshold, e.g. --threshold 0.45"
            )

        if not 0 < args.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

        run_final_test(args.threshold)


if __name__ == "__main__":
    main()
