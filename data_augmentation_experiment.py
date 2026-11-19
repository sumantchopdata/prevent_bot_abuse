
from pathlib import Path
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
    precision_recall_fscore_support,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("claude_datasets")
CACHE_DIR = Path("artifacts/embeddings")
OUTPUT_DIR = Path("artifacts/data_augmentation_experiment")

TRAIN_FILE = DATA_DIR / "combined_train_v4_balanced.csv"
EVAL_FILE = DATA_DIR / "combined_eval_v4_balanced.csv"

# Change these two paths to match your actual filenames.
A10_FILE = DATA_DIR / "abuse_mixed_intent_messages_v2.csv"
G3_FILE = DATA_DIR / "products_services_genuine_messages_v2.csv"

TEXT_CANDIDATES = ["text"]
LABEL_COL = "labels"
CATEGORY_COL = "category"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MODEL_C = 10
CLASS_WEIGHT = None
MAX_ITER = 2000
RANDOM_STATE = 42

# Selected in the previous threshold experiment.
THRESHOLD = 0.35

EXPERIMENT_NAME = "Prevent Bot Abuse - Data Augmentation A10 G3"
TRACKING_URI = "sqlite:///mlflow.db"


# ============================================================
# DATA LOADING
# ============================================================

def find_text_column(df, source_name):
    text_col = next(
        (col for col in TEXT_CANDIDATES if col in df.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            f"{source_name} must contain one of these columns: "
            f"{TEXT_CANDIDATES}"
        )

    return text_col


def load_main_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    text_col = find_text_column(df, str(path))

    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain '{LABEL_COL}'")

    result = pd.DataFrame({
        "text": df[text_col].fillna("").astype(str),
        "labels": df[LABEL_COL].astype(int),
    })

    if CATEGORY_COL in df.columns:
        result[CATEGORY_COL] = df[CATEGORY_COL]

    validate_labels(result, str(path))
    return result


def load_augmentation_file(path, expected_category, expected_label):
    if not path.exists():
        raise FileNotFoundError(
            f"Augmentation file not found: {path}\n"
            "Update A10_FILE or G3_FILE at the top of the script."
        )

    df = pd.read_csv(path)
    text_col = find_text_column(df, str(path))

    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain '{LABEL_COL}'")

    result = pd.DataFrame({
        "text": df[text_col].fillna("").astype(str),
        "labels": df[LABEL_COL].astype(int),
    })

    # If category is absent, assign it from the filename's purpose.
    if CATEGORY_COL in df.columns:
        result[CATEGORY_COL] = df[CATEGORY_COL].fillna(
            expected_category
        ).astype(str)

        unexpected = set(result[CATEGORY_COL].unique()) - {
            expected_category
        }

        if unexpected:
            raise ValueError(
                f"{path} contains unexpected categories: {unexpected}"
            )
    else:
        result[CATEGORY_COL] = expected_category

    validate_labels(result, str(path))

    if not (result["labels"] == expected_label).all():
        raise ValueError(
            f"{path}: expected every label to be {expected_label} "
            f"for category {expected_category}."
        )

    if result.empty:
        raise ValueError(f"{path} contains no examples.")

    print(
        f"Loaded {len(result)} {expected_category} examples "
        f"from {path}"
    )

    return result


def validate_labels(df, source_name):
    labels = set(df["labels"].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{source_name}: labels must be 0 (abuse) or 1 (genuine)."
        )


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings(texts, embedding_model):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Hash the model name and all texts to identify cached embeddings.
    content_hash = hashlib.sha256()

    content_hash.update(EMBEDDING_MODEL_NAME.encode("utf-8"))

    for text in texts:
        content_hash.update(text.encode("utf-8"))
        content_hash.update(b"\0")

    cache_key = content_hash.hexdigest()[:20]
    cache_path = CACHE_DIR / f"{cache_key}.npz"

    if cache_path.exists():
        print(f"Using cached embeddings: {cache_path}")
        with np.load(cache_path) as cached:
            return cached["embeddings"]

    print(f"Generating embeddings for {len(texts)} examples...")

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


# ============================================================
# MODEL AND PREDICTIONS
# ============================================================

def create_model():
    return LogisticRegression(
        C=MODEL_C,
        class_weight=CLASS_WEIGHT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE
    )


def get_abuse_probabilities(model, X):
    probabilities = model.predict_proba(X)

    # Explicitly identify class 0 (abuse).
    abuse_index = list(model.classes_).index(0)

    return probabilities[:, abuse_index]


def predict_at_threshold(abuse_probabilities):
    # Label 0 = abuse; label 1 = genuine.
    return np.where(
        abuse_probabilities >= THRESHOLD,
        0,
        1
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, abuse_probabilities):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    y_abuse_binary = (np.asarray(y_true) == 0).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "abuse_precision": precision[0],
        "abuse_recall": recall[0],
        "abuse_f1": f1[0],

        "genuine_precision": precision[1],
        "genuine_recall": recall[1],
        "genuine_f1": f1[1],

        "abuse_roc_auc": roc_auc_score(
            y_abuse_binary,
            abuse_probabilities
        ),

        "abuse_pr_auc": average_precision_score(
            y_abuse_binary,
            abuse_probabilities
        )
    }

    return metrics


def calculate_category_metrics(y_true, y_pred, categories):
    results = {}

    if categories is None:
        return results

    category_series = pd.Series(categories).reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for category in category_series.dropna().unique():
        mask = (category_series == category).to_numpy()

        category_true = y_true[mask]
        category_pred = y_pred[mask]

        safe_category = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            str(category)
        )

        results[f"category_{safe_category}_n"] = len(category_true)

        results[f"category_{safe_category}_accuracy"] = (
            accuracy_score(category_true, category_pred)
        )

        # Recall is calculated only if the category contains that class.
        if 0 in category_true:
            _, recall, _, _ = precision_recall_fscore_support(
                category_true,
                category_pred,
                labels=[0],
                zero_division=0
            )
            results[
                f"category_{safe_category}_abuse_recall"
            ] = recall[0]

        if 1 in category_true:
            _, recall, _, _ = precision_recall_fscore_support(
                category_true,
                category_pred,
                labels=[1],
                zero_division=0
            )
            results[
                f"category_{safe_category}_genuine_recall"
            ] = recall[0]

    return results


# ============================================================
# ARTIFACTS
# ============================================================

def create_prediction_dataframe(
    eval_df,
    y_pred,
    abuse_probabilities
):
    predictions = eval_df.copy()

    predictions["predicted_label"] = y_pred
    predictions["abuse_probability"] = abuse_probabilities
    predictions["genuine_probability"] = 1 - abuse_probabilities

    predictions["correct"] = (
        predictions["labels"] == predictions["predicted_label"]
    )

    return predictions


def save_run_artifacts(
    run_name,
    eval_df,
    y_true,
    y_pred,
    abuse_probabilities,
    metrics,
    category_metrics
):
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Predictions and probabilities
    predictions = create_prediction_dataframe(
        eval_df,
        y_pred,
        abuse_probabilities
    )

    predictions_path = run_dir / "eval_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["abuse", "genuine"],
        zero_division=0
    )

    report_path = run_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_path = run_dir / "confusion_matrix.csv"

    pd.DataFrame(
        cm,
        index=["actual_abuse", "actual_genuine"],
        columns=["predicted_abuse", "predicted_genuine"]
    ).to_csv(cm_path)

    # Category metrics
    category_path = run_dir / "category_metrics.csv"

    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in category_metrics.items()
        ]
    ).to_csv(category_path, index=False)

    return {
        "predictions": predictions_path,
        "report": report_path,
        "confusion_matrix": cm_path,
        "category_metrics": category_path
    }


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_model_experiment(
    run_name,
    train_df,
    eval_df,
    X_train,
    X_eval,
    embedding_model_name
):
    print(f"\n{'=' * 60}")
    print(f"Running: {run_name}")
    print(f"Training examples: {len(train_df)}")
    print(f"Evaluation examples: {len(eval_df)}")
    print(f"{'=' * 60}")

    y_train = train_df["labels"].to_numpy()
    y_eval = eval_df["labels"].to_numpy()

    model = create_model()
    model.fit(X_train, y_train)

    abuse_probabilities = get_abuse_probabilities(
        model,
        X_eval
    )

    y_pred = predict_at_threshold(abuse_probabilities)

    metrics = calculate_metrics(
        y_eval,
        y_pred,
        abuse_probabilities
    )

    categories = (
        eval_df[CATEGORY_COL]
        if CATEGORY_COL in eval_df.columns
        else None
    )

    category_metrics = calculate_category_metrics(
        y_eval,
        y_pred,
        categories
    )

    artifacts = save_run_artifacts(
        run_name,
        eval_df,
        y_eval,
        y_pred,
        abuse_probabilities,
        metrics,
        category_metrics
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": MODEL_C,
            "class_weight": "None",
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "embedding_model": embedding_model_name,
            "threshold": THRESHOLD,
            "training_examples": len(train_df),
            "evaluation_examples": len(eval_df),
            "training_data": run_name
        })

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(category_metrics)

        for artifact_path in artifacts.values():
            mlflow.log_artifact(str(artifact_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

    print("\nOverall metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nCategory metrics:")
    for name, value in category_metrics.items():
        print(f"{name}: {value}")

    return {
        **metrics,
        **category_metrics
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading datasets...")

    train_df = load_main_dataset(TRAIN_FILE)
    eval_df = load_main_dataset(EVAL_FILE)

    a10_df = load_augmentation_file(
        A10_FILE,
        expected_category="A10",
        expected_label=0
    )

    g3_df = load_augmentation_file(
        G3_FILE,
        expected_category="G3",
        expected_label=1
    )

    augmented_examples = pd.concat(
        [a10_df, g3_df],
        ignore_index=True
    )

    augmented_train_df = pd.concat(
        [train_df, augmented_examples],
        ignore_index=True
    )

    print("\nDataset sizes:")
    print(f"Original training set: {len(train_df)}")
    print(f"A10 additions: {len(a10_df)}")
    print(f"G3 additions: {len(g3_df)}")
    print(f"Augmented training set: {len(augmented_train_df)}")
    print(f"Evaluation set: {len(eval_df)}")

    print("\nTraining category counts after augmentation:")
    if CATEGORY_COL in augmented_train_df.columns:
        print(
            augmented_train_df[CATEGORY_COL]
            .value_counts(dropna=False)
            .to_string()
        )

    # Load the embedding model once and reuse it.
    print("\nLoading sentence-transformer model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    eval_texts = eval_df["text"].tolist()
    train_texts = train_df["text"].tolist()
    augmented_texts = augmented_train_df["text"].tolist()

    # Embeddings are cached independently for each text collection.
    X_eval = get_embeddings(eval_texts, embedding_model)
    X_train = get_embeddings(train_texts, embedding_model)
    X_augmented_train = get_embeddings(
        augmented_texts,
        embedding_model
    )

    # --------------------------------------------------------
    # Baseline: original training data
    # --------------------------------------------------------

    baseline_results = run_model_experiment(
        run_name="baseline_original_training_data",
        train_df=train_df,
        eval_df=eval_df,
        X_train=X_train,
        X_eval=X_eval,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )

    # --------------------------------------------------------
    # Experiment A: original + A10 + G3
    # --------------------------------------------------------

    augmented_results = run_model_experiment(
        run_name="experiment_A_A10_G3_augmentation",
        train_df=augmented_train_df,
        eval_df=eval_df,
        X_train=X_augmented_train,
        X_eval=X_eval,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )

    # --------------------------------------------------------
    # Save side-by-side comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "metric": baseline_results.keys(),
        "baseline": baseline_results.values(),
        "experiment_A": [
            augmented_results[key]
            for key in baseline_results.keys()
        ]
    })

    comparison["change"] = (
        comparison["experiment_A"] - comparison["baseline"]
    )

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    with mlflow.start_run(run_name="experiment_A_comparison_summary"):
        mlflow.log_params({
            "experiment": "Baseline vs A10/G3 augmentation",
            "C": MODEL_C,
            "class_weight": "None",
            "threshold": THRESHOLD,
            "embedding_model": EMBEDDING_MODEL_NAME
        })

        mlflow.log_artifact(str(comparison_path))

    print("\n" + "=" * 60)
    print("BASELINE VS EXPERIMENT A")
    print("=" * 60)
    print(comparison.to_string(index=False))

    print(f"\nComparison saved to: {comparison_path}")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
