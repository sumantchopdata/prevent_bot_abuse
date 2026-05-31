#%%
import pandas as pd
import os

CSV_FILENAME = "banking_chatbot_adversarial_prompts.csv"

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


try:
    df = load_dataset()
except FileNotFoundError:
    print(f"File not found: {CSV_FILENAME}")
else:
    explore_dataset(df)

# %%
# give me the number of prompts that contain the word "ignore"
num_ignore = df["text"].str.contains("ignore", case=False, na=False).sum()
print(f"Number of prompts containing 'ignore': {num_ignore}")
# %%
# give me the number of prompts that contain the word "disregard"
num_disregard = df["text"].str.contains("disregard", case=False, na=False).sum()
print(f"Number of prompts containing 'disregard': {num_disregard}")
# %%
# give me the number of prompts that contain the word "forget"
num_forget = df["text"].str.contains("forget", case=False, na=False).sum()
print(f"Number of prompts containing 'forget': {num_forget}")
# %%
# give me the number of prompts that contain the word "pretend"
num_pretend = df["text"].str.contains("pretend", case=False, na=False).sum()
print(f"Number of prompts containing 'pretend': {num_pretend}")
# %%
# print the top 5 longest prompts in the dataset
df["text_length"] = df["text"].fillna("").str.len()
top_longest = df.sort_values("text_length", ascending=False).head(5)
print("Top 5 longest prompts:")
print(top_longest[["text", "text_length"]].to_string(index=False))
# %%
# print the top 5 shortest prompts in the dataset
top_shortest = df.sort_values("text_length", ascending=True).head(5)
print("Top 5 shortest prompts:")
print(top_shortest[["text", "text_length"]].to_string(index=False))
# %%
# plot the text length distribution
import matplotlib.pyplot as plt
plt.hist(df["text_length"].fillna(0), bins=20, edgecolor='black')
plt.title("Distribution of Prompt Lengths")
plt.xlabel("Prompt Length (characters)")
plt.ylabel("Frequency")
plt.show()
# %%
# plot the number of words in each prompt
df["word_count"] = df["text"].fillna("").str.split().str.len()
plt.hist(df["word_count"], bins=20, edgecolor='black')
plt.title("Distribution of Prompt Word Counts")
plt.xlabel("Number of Words in Prompt")
plt.ylabel("Frequency")
plt.show()
# %%
# print the top 25 most common words in the prompts
from collections import Counter
all_words = df["text"].fillna("").str.split().explode()
word_counts = Counter(all_words)
most_common = word_counts.most_common(25)
print("Top 25 most common words:")
for word, count in most_common:
    print(f"{word}: {count}")
# %%
# print the number of prompts that contain the word 'bank' or 'account' or 'credit' or 'debit' or 'balance' or 'transaction' or 'loan' or 'payment' or 'card'
keywords = ["bank", "account", "credit", "debit", "balance", "transaction", "loan", "payment", "card"]
num_keywords = df["text"].str.contains("|".join(keywords), case=False, na=False).sum()
print(f"Number of prompts containing any of the keywords: {num_keywords}")
# %%
# print the prompts with the above keywords
keyword_prompts = df[df["text"].str.contains("|".join(keywords), case=False, na=False)]
print("Prompts containing any of the keywords:")
print(keyword_prompts["text"].to_string(index=False))