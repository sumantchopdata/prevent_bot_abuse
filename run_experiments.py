# ML Flow experiments for Preventing Bot Abuse using Logistic Regression on Sentence Embeddings
#%%
import hashlib
import json
import re
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path("claude_datasets")
CACHE_DIR = Path("artifacts/embeddings")
OUTPUT_DIR = Path("artifacts")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_CANDIDATES = ["user_input", "text"]
LABEL_COL = "labels"
CATEGORY_COL = "category"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

EXPERIMENT_NAME = "Prevent Bot Abuse - Logistic Regression"

RANDOM_STATE = 42
MAX_ITER = 2000

# Label definitions
ABUSE_LABEL = 0
GENUINE_LABEL = 1

# Model configurations to compare
EXPERIMENTS = [
    {
        "name": "baseline",
        "C": 1.0,
        "class_weight": None
    },
    {
        "name": "regularisation_C_0.1",
        "C": 0.1,
        "class_weight": None
    },
    {
        "name": "regularisation_C_0.01",
        "C": 0.01,
        "class_weight": None
    },
    {
        "name": "regularisation_C_10",
        "C": 10.0,
        "class_weight": None
    },
    {
        "name": "class_weight_balanced",
        "C": 1.0,
        "class_weight": "balanced"
    },
    {
        "name": "regularisation_and_balanced",
        "C": 0.1,
        "class_weight": "balanced"
    }
]
#%%

# ============================================================
# 2. LOAD DATA
# ============================================================

def load_split(split_name):

    path = DATA_DIR / f"combined_{split_name}_v4_balanced.csv"

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path)

    text_col = next(
        (col for col in TEXT_CANDIDATES if col in df.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            f"No text column found in {path}. "
            f"Expected one of {TEXT_CANDIDATES}"
        )

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing '{LABEL_COL}' in {path}")

    if df[text_col].isna().any():
        raise ValueError(f"Missing text values in {path}")

    if df[LABEL_COL].isna().any():
        raise ValueError(f"Missing labels in {path}")

    df = df.copy()
    df[text_col] = df[text_col].astype(str)

    labels = set(df[LABEL_COL].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels in {path}: {labels}"
        )

    df = df.rename(columns={text_col: "text"})

    print(f"\n{split_name.upper()} SET")
    print(f"Rows: {len(df)}")
    print("Label distribution:")
    print(df[LABEL_COL].value_counts().sort_index())

    return df


train_df = load_split("train")
eval_df = load_split("eval")
test_df = load_split("test")
#%%
# ============================================================
# 3. GENERATE / LOAD CACHED EMBEDDINGS
# ============================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embeddings(df, split_name):

    texts = df["text"].tolist()

    # Hash text content to detect changes in a split
    text_hash = hashlib.sha256(
        "\n".join(texts).encode("utf-8")
    ).hexdigest()

    cache_path = CACHE_DIR / f"{split_name}.npz"

    if cache_path.exists():

        cached = np.load(cache_path)

        if (
            str(cached["text_hash"]) == text_hash
            and str(cached["model_name"]) == EMBEDDING_MODEL_NAME
        ):
            print(f"Using cached embeddings: {split_name}")

            return cached["embeddings"]

    print(f"Generating embeddings: {split_name}")

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        text_hash=text_hash,
        model_name=EMBEDDING_MODEL_NAME
    )

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


X_train = get_embeddings(train_df, "train")
X_eval = get_embeddings(eval_df, "eval")
X_test = get_embeddings(test_df, "test")

y_train = train_df[LABEL_COL].to_numpy()
y_eval = eval_df[LABEL_COL].to_numpy()
y_test = test_df[LABEL_COL].to_numpy()
#%%
# ============================================================
# 4. METRIC CALCULATION
# ============================================================
def calculate_metrics(y_true, y_pred, abuse_prob):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_f1": f1_score(
            y_true, y_pred, average="macro"
        ),

        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted"
        ),

        "abuse_precision": precision_score(
            y_true, y_pred,
            pos_label=ABUSE_LABEL,
            zero_division=0
        ),

        "abuse_recall": recall_score(
            y_true, y_pred,
            pos_label=ABUSE_LABEL,
            zero_division=0
        ),

        "abuse_f1": f1_score(
            y_true, y_pred,
            pos_label=ABUSE_LABEL,
            zero_division=0
        ),

        "genuine_precision": precision_score(
            y_true, y_pred,
            pos_label=GENUINE_LABEL,
            zero_division=0
        ),

        "genuine_recall": recall_score(
            y_true, y_pred,
            pos_label=GENUINE_LABEL,
            zero_division=0
        ),

        "genuine_f1": f1_score(
            y_true, y_pred,
            pos_label=GENUINE_LABEL,
            zero_division=0
        )
    }

    # Probability-based metrics for the abuse class
    if len(np.unique(y_true)) == 2:

        y_abuse = (y_true == ABUSE_LABEL).astype(int)

        metrics["abuse_roc_auc"] = roc_auc_score(
            y_abuse, abuse_prob
        )

        metrics["abuse_pr_auc"] = average_precision_score(
            y_abuse, abuse_prob
        )

    return metrics

# ============================================================
# 5. CATEGORY-WISE EVALUATION
# ============================================================

def log_category_metrics(df, y_pred):

    if CATEGORY_COL not in df.columns:
        print("\nNo category column found. Skipping category metrics.")
        return

    results = {}

    categories = df[CATEGORY_COL].fillna("UNKNOWN")

    for category in categories.unique():

        mask = (categories == category).to_numpy()

        # Accuracy within each category
        category_accuracy = accuracy_score(
            df.loc[mask, LABEL_COL],
            y_pred[mask]
        )

        safe_name = re.sub(
            r"[^a-zA-Z0-9_]+", "_", str(category)
        )

        metric_name = f"category_{safe_name}_accuracy"

        mlflow.log_metric(metric_name, category_accuracy)

        results[str(category)] = category_accuracy

    print("\nCategory-wise accuracy:")
    print(json.dumps(results, indent=2))
#%%
# ============================================================
# 6. CREATE MLflow EXPERIMENT
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)
#%%

# ============================================================
# 7. RUN EXPERIMENTS
# ============================================================

experiment_results = []

for config in EXPERIMENTS:

    print("\n" + "=" * 60)
    print(f"Running experiment: {config['name']}")
    print("=" * 60)

    with mlflow.start_run(run_name=config["name"]):

        # Log parameters
        mlflow.log_params({
            "embedding_model": EMBEDDING_MODEL_NAME,
            "classifier": "LogisticRegression",
            "C": config["C"],
            "class_weight": str(config["class_weight"]),
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "threshold": 0.5
        })

        # Train model
        model = LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        # Evaluate on evaluation set
        y_pred = model.predict(X_eval)

        abuse_index = list(model.classes_).index(ABUSE_LABEL)

        abuse_prob = model.predict_proba(X_eval)[:, abuse_index]

        metrics = calculate_metrics(
            y_eval, y_pred, abuse_prob
        )

        # Log metrics
        mlflow.log_metrics({
            f"eval_{key}": value
            for key, value in metrics.items()
        })

        # Category-wise metrics
        log_category_metrics(eval_df, y_pred)

        # Log classification report
        report = classification_report(
            y_eval,
            y_pred,
            labels=[0, 1],
            target_names=["Abuse", "Genuine"],
            zero_division=0
        )

        report_path = OUTPUT_DIR / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")

        mlflow.log_artifact(str(report_path))

        # Log confusion matrix
        cm = confusion_matrix(
            y_eval, y_pred, labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.imshow(cm)
        ax.set_title(f"Evaluation - {config['name']}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], ["Abuse", "Genuine"])
        ax.set_yticks([0, 1], ["Abuse", "Genuine"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        fig.tight_layout()

        cm_path = OUTPUT_DIR / f"cm_{config['name']}.png"
        fig.savefig(cm_path)
        plt.close(fig)

        mlflow.log_artifact(str(cm_path))

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        experiment_results.append({
            "run_name": config["name"],
            "C": config["C"],
            "class_weight": config["class_weight"],
            **metrics
        })

        print("\nEvaluation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

#%%
# ============================================================
# 8. COMPARE EXPERIMENTS
# ============================================================

results_df = pd.DataFrame(experiment_results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
)

print("\n" + "=" * 60)
print("EXPERIMENT COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_path = OUTPUT_DIR / "experiment_comparison.csv"
results_df.to_csv(results_path, index=False)
#%%

# ============================================================
# 9. SELECT BEST CONFIGURATION
# ============================================================

best_result = results_df.iloc[0]

best_config = next(
    config for config in EXPERIMENTS
    if config["name"] == best_result["run_name"]
)

print("\nBest configuration:")
print(best_config)
#%%

# ============================================================
# 10. FINAL TRAINING ON TRAIN + EVAL
# ============================================================

print("\nTraining final model on train + evaluation data...")

X_train_final = np.vstack([X_train, X_eval])
y_train_final = np.concatenate([y_train, y_eval])

final_model = LogisticRegression(
    C=best_config["C"],
    class_weight=best_config["class_weight"],
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
)

final_model.fit(X_train_final, y_train_final)


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on untouched test set...")

test_pred = final_model.predict(X_test)

abuse_index = list(final_model.classes_).index(ABUSE_LABEL)

test_abuse_prob = final_model.predict_proba(X_test)[:, abuse_index]

test_metrics = calculate_metrics(
    y_test, test_pred, test_abuse_prob
)

print("\nFINAL TEST METRICS")

for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")


# Log final model in a separate run
with mlflow.start_run(run_name="final_test_evaluation"):

    mlflow.log_params({
        "selected_experiment": best_config["name"],
        "C": best_config["C"],
        "class_weight": str(best_config["class_weight"]),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "training_data": "train + eval"
    })

    mlflow.log_metrics({
        f"test_{key}": value
        for key, value in test_metrics.items()
    })

    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="final_model"
    )

    mlflow.log_artifact(str(results_path))


print("\nAll experiments completed!")
print("Run 'mlflow ui' to view the results.")
