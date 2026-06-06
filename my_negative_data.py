#%%
'''
This script loads the adversarial inputs dataset made by Claude and performs
EDA to understand the characteristics of the inputs. It includes statistics
on prompt lengths, word counts, and the presence of specific keywords related
to banking.
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter

CSV_FILENAME = "banking_chatbot_adversarial_inputs.csv"

def load_dataset(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), CSV_FILENAME)
    return pd.read_csv(path, header=None, names=["text"] , dtype=str)

def explore_dataset(df):
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("sample rows:")
    print(df.head(10).to_string(index=False))
    print("\nunique values:", df["text"].nunique())
    print("missing values:", df["text"].isna().sum())
    print("\ntext length stats:")
    lengths = df["text"].fillna("").str.len()
    print(lengths.describe())

df = load_dataset()
explore_dataset(df)

# the number of inputs that contain the word "ignore"
print(df["text"].str.contains("ignore", case=False, na=False).sum())

# the number of inputs that contain the word "disregard"
print(df["text"].str.contains("disregard", case=False, na=False).sum())

# the number of inputs that contain the word "forget"
print(df["text"].str.contains("forget", case=False, na=False).sum())

# the number of inputs that contain the word "pretend"
print(df["text"].str.contains("pretend", case=False, na=False).sum())

# print the top 5 longest inputs in the dataset
df["text_length"] = df["text"].fillna("").str.len()
top_longest = df.sort_values("text_length", ascending=False).head(5)
print("Top 5 longest inputs:")
print(top_longest[["text", "text_length"]].to_string(index=False))

# print the top 5 shortest inputs in the dataset
top_shortest = df.sort_values("text_length", ascending=True).head(5)
print("Top 5 shortest inputs:")
print(top_shortest[["text", "text_length"]].to_string(index=False))

# plot the text length distribution
plt.hist(df["text_length"].fillna(0), bins=20, edgecolor='black')
plt.title("Distribution of Prompt Lengths")
plt.xlabel("Prompt Length (characters)")
plt.ylabel("Frequency")
plt.show()

# plot the number of words in each prompt
df["word_count"] = df["text"].fillna("").str.split().str.len()
plt.hist(df["word_count"], bins=20, edgecolor='black')
plt.title("Distribution of Prompt Word Counts")
plt.xlabel("Number of Words in Prompt")
plt.ylabel("Frequency")
plt.show()

# print the top 25 most common words in the inputs
all_words = df["text"].fillna("").str.split().explode()
most_common = Counter(all_words).most_common(25)
print("Top 25 most common words:")
for word, count in most_common:
    print(f"{word}: {count}")

# print the number of inputs that contain the words
# 'bank' or 'account' or 'credit' or 'debit' or 'balance' or 'transaction' or
# 'loan' or 'payment' or 'card'
keywords = ["bank", "account", "credit", "debit", "balance", "transaction",
            "loan", "payment", "card"]
print(df["text"].str.contains("|".join(keywords), case=False, na=False).sum())

# print the inputs with the above keywords
keyword_inputs = df[df["text"].str.contains("|".join(keywords), case=False, na=False)]
print("inputs containing any of the keywords:")
print(keyword_inputs["text"].to_string(index=False))