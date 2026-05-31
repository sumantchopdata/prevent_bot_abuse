---
language: 
- en
license: mit
tags:
- text-classification
- intent-classification
- intent-detection
- chatbot
- dialogue-system
- conversation-ai
- sentence-classification
- supervised-learning
- synthetic-data
- tanaos
pretty_name: tanaos-intent-classifier-v1 Training Dataset
task_categories:
- text-classification
task_ids:
- hate-speech-detection
- sentiment-classification
size_categories:
- 10K<n<20K
---

<p align="center">
    <img src="https://raw.githubusercontent.com/tanaos/.github/master/assets/logo.png" width="250px" alt="Tanaos – Train task specific LLMs without training data, for offline NLP and Text Classification">
</p>

# Tanaos Intent Classifier Training Dataset

This dataset was created synthetically by Tanaos with the [Artifex](https://github.com/tanaos/artifex) Python library.

The dataset is designed to **train and evaluate intent classification systems** — models that can identify user intents in conversational AI applications, such as chatbots and virtual assistants. 

Our flagship intent classification model, [tanaos-intent-classifier-v1](https://huggingface.co/tanaos/tanaos-intent-classifier-v1), was trained on this dataset.

## Dataset Summary

The dataset contains text samples with numeric labels from 0 to 11. The numeric labels have the following meanings:

| Label | Category | Description |
|-------|----------|-------------|
| 0 | `greeting` | Greeting or saying hello. |
| 1 | `farewell` | Saying goodbye or farewell. |
| 2 | `thank_you` | Expressing gratitude or thanks. |
| 3 | `affirmation` | Agreeing or confirming something. |
| 4 | `negation` | Disagreeing or denying something. |
| 5 | `small_talk` | Engaging in casual or light conversation with no specific purpose. |
| 6 | `bot_capabilities` | Inquiries about the bot's features or abilities. |
| 7 | `feedback_positive` | Providing positive feedback about the bot, service, or experience. |
| 8 | `feedback_negative` | Providing negative feedback about the bot, service, or experience. |
| 9 | `clarification` | Asking for clarification or more information about a previous statement or question. |
| 10 | `suggestion` | Offering a suggestion or recommendation for improvement. |
| 11 | `language_change` | Requesting a change in the language being used by the bot or information about language options. |

## How to Use

```python
from datasets import load_dataset

dataset = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1")

print(dataset["train"][0])
```

## Intended Use

This dataset is meant for **training, fine-tuning, and evaluating** models that perform intent classification in conversational AI systems.
Common use cases:
- Chatbots
- Virtual Assistants
- Customer Support Systems
- Dialogue Systems