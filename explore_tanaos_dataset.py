#%%
# This code explores the Tanaos Intent Classifier Training Dataset.
'''
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


We'll remove label 6 (bot_capabilities) from the dataset, as it can be used in
both positive and negative contexts, making it difficult to classify as either.
Then we'll mark label 5 (small_talk) as 0 (negative) and all other labels as 1
(positive). We'll combine this data with more data from other sources to create
a larger dataset.
'''
#%%
import pandas as pd
import re

df = pd.read_csv("data/data.csv")

# Basic info
print("SHAPE:", df.shape)
print("\nCOLUMNS:\n", df.columns.tolist())

print(df.info())
print(df.head(10).to_string(index=False))

# Summary statistics for numeric cols
print(df.describe(include=["number"]).T)

# Summary for object/categorical columns
obj = df.select_dtypes(include=["object", "category"])
if not obj.empty:
    print("\nTOP VALUE COUNTS FOR CATEGORICAL COLUMNS:")
    for c in obj.columns:
        print(f"\n-- {c} --")
        print(obj[c].value_counts(dropna=False).head(10).to_string())

# Missing values
print(df.isna().sum())
print(df['labels'].value_counts(dropna=False))

# check if some texts have multiple labels
text_label_counts = df.groupby('text')['labels'].nunique()
texts_with_mult_labels = text_label_counts[text_label_counts > 1]
print(f"Number of unique texts with multiple labels: {len(texts_with_mult_labels)}")

# read the texts with multiple labels
print(df[df['text'] == 'Can you help me with something?'])
# included in small_talk, clarification, bot_capabilities and affirmation,
# which makes sense as it can be used in all those contexts.

# bot capabilities (label 6) is difficult to classify as positive or negative,
# as it can be used in both contexts. So we'll exclude it from the dataset.

data = df[df['labels'] != 6]
print(data.shape)

# remove punctuation or special characters from the text column, make everything lowercase
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x)).str.lower()

# find out text length distribution
df['text_length'] = df['text'].apply(len)
print(df['text_length'].describe())

# remove duplicates
before_dedup = df.shape[0]
df = df.drop_duplicates(subset=['text'])
after_dedup = df.shape[0]
print(f"Removed {before_dedup - after_dedup} duplicate rows. New shape: {df.shape}")


# read the 5 smallest inputs
print(df.nsmallest(5, 'text_length')[['text', 'labels']].to_string(index=False))

# find out the word count distribution
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
print(df['word_count'].describe())

# read the 5 inputs with the fewest words
print("5 texts with the fewest words:")
print(df.nsmallest(5, 'word_count')[['text', 'labels']].to_string(index=False))

# read the 5 longest inputs
print("5 longest texts:")
print(df.nlargest(5, 'text_length')[['text', 'labels']].to_string(index=False))

# remove the text_length and word_count columns before saving
df = df.drop(columns=['text_length', 'word_count'])

# make all labels 5 as 0 and all others as 1, so we have a binary classification problem
df['labels'] = df['labels'].apply(lambda x: 0 if x == 5 else 1).astype(int)

# find distribution of the new binary labels
print("Distribution of binary labels:")
print(df['labels'].value_counts())

# save the final cleaned and labeled data to a new CSV file
df.to_csv("data/cleaned_data.csv", index=False)