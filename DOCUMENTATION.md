# Prevent Bot Abuse: Technical Documentation

## 1. Problem Statement

Large language model based chatbots are susceptible to abusive and adversarial user inputs.

Examples include:

- Toxic prompts
- Prompt injections
- Irrelevant requests
- Attempts to manipulate system behavior

This project develops a binary classifier to identify such inputs before they are processed by a banking chatbot.

---

# 2. Objective

Classify user queries into:

### Genuine Query (1)

Examples:

- "What is my account balance?"
- "How do I reset my debit card PIN?"

### Misuse Query (0)

Examples:

- Toxic prompts
- Adversarial instructions
- Irrelevant conversations

---

# 3. Dataset Construction

Three datasets are combined.

## Genuine Class

### cleaned_data.csv

Contains legitimate banking customer queries.

Assigned label:

```
1
```

---

## Abuse Class

### toxic_chat_dataset.csv

General toxic conversations.

Assigned label:

```
0
```

### banking_chatbot_adversarial_prompts.csv

Adversarial and misuse prompts.

Assigned label:

```
0
```

---

# 4. Dataset Processing

Performed in:

```
combine_datasets.py
```

Steps:

1. Load datasets
2. Standardize columns
3. Assign labels
4. Remove duplicates
5. Remove empty rows
6. Shuffle data
7. Perform 80-20 split

Outputs:

```
train.csv
test.csv
```

---

# 5. Embedding Generation

File:

```
create_embeddings.py
```

Model:

```
all-MiniLM-L6-v2
```

Library:

```python
sentence-transformers
```

Each text sample is converted into a semantic vector representation.

Embedding dimension:

```
384
```

Outputs:

```
train_embeddings.csv
test_embeddings.csv
```

---

# 6. Classification Model

File:

```
train_model.py
```

Model:

```python
LogisticRegression(
    max_iter=2000
)
```

Why Logistic Regression?

- Fast training
- Interpretable
- Strong baseline for text embeddings
- Low computational requirements

---

# 7. Training Procedure

Input:

```
train_embeddings.csv
```

Features:

```
384-dimensional embeddings
```

Target:

```
labels
```

The model learns a linear decision boundary separating:

- Genuine requests
- Abuse requests

---

# 8. Evaluation

Input:

```
test_embeddings.csv
```

Metrics:

- Accuracy
- Precision
- Recall
- F1-score

---

## Test Accuracy

```
94.44%
```

### Classification Report

| Class | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Misuse | 0.96 | 0.88 | 0.92 |
| Genuine | 0.94 | 0.98 | 0.96 |

---

# 9. Error Analysis

The system stores:

### False Positives

Misuse predicted as genuine.

Output:

```
misclassified_false_positives.csv
```

### False Negatives

Genuine predicted as misuse.

Output:

```
misclassified_false_negatives.csv
```

These files help identify weaknesses in the classifier.

---

# 10. Embedding Visualization

File:

```
plot_embeddings.py
```

Technique:

### UMAP

High-dimensional embeddings are reduced to two dimensions.

Generated plots:

### Cluster Distribution

```
umap_projection.png
```

### Predicted Misuse

```
umap_projection_predictions.png
```

### Predicted Genuine

```
umap_projection_genuine_predictions.png
```

### Misclassification Analysis

```
umap_projection_misclassifications.png
```

---

# 11. Overall Architecture

```
            User Input
                 ↓
         Dataset Construction
                 ↓
            Train/Test Split
                 ↓
      Sentence Transformer Encoder
          (all-MiniLM-L6-v2)
                 ↓
         384-Dimensional Vector
                 ↓
          Logistic Regression
                 ↓
              Prediction
                 ↓
        Genuine / Misuse Label
```

---

# 12. Strengths

- Lightweight architecture
- Fast inference
- High recall for genuine queries
- Easy deployment
- Interpretable model
- Requires little compute

---

# 13. Limitations

- Binary classification only
- No confidence calibration
- No transformer fine-tuning
- Dataset imbalance may affect performance
- Adversarial prompts evolve over time

---

# 14. Potential Improvements

## Better Models

- XGBoost
- LightGBM
- Random Forest

## Deep Learning

- Fine-tune MiniLM
- DistilBERT
- BERT

## Production Enhancements

- Confidence thresholding
- Human feedback loop
- Continuous retraining
- Drift detection

## Explainability

- SHAP values
- Feature importance
- Probability calibration

---

# 15. Dependencies

```text
pandas
scikit-learn
sentence-transformers
umap-learn
matplotlib
joblib
tqdm
```

---

# 16. Conclusion

This project demonstrates that semantic embeddings combined with a simple Logistic Regression classifier can effectively detect abusive and adversarial chatbot inputs with high accuracy while remaining computationally efficient and easy to deploy.
