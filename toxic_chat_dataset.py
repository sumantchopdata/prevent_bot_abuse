#%%
# This script loads the toxic chat dataset from Hugging Face, combines the two
# versions of the dataset, and saves the combined dataset to a CSV file. It
# then cleans the dataset by removing empty rows, punctuation, and special
# characters, and analyzes the text length and word count distribution. Finally,
# it saves the cleaned dataset to a new CSV file.

import pandas as pd
from datasets import load_dataset
import re
import matplotlib.pyplot as plt
from collections import Counter

dataset_1 = load_dataset("lmsys/toxic-chat", 'toxicchat0124')

df = dataset_1['train'].to_pandas()
print(type(df))

# we'll keep the toxic and jailbreaking examples
d1_train_toxic = df[df['toxicity'] == 1]
d1_train_jailbreak = df[df['jailbreaking'] == 1]
d1_train = pd.concat([d1_train_toxic, d1_train_jailbreak]).drop_duplicates()
print(d1_train.shape)

df = dataset_1['test'].to_pandas()
d1_test_toxic = df[df['toxicity'] == 1]
d1_test_jailbreak = df[df['jailbreaking'] == 1]
d1_test = pd.concat([d1_test_toxic, d1_test_jailbreak]).drop_duplicates()
print(d1_test.shape)

dataset_2 = load_dataset("lmsys/toxic-chat", 'toxicchat1123')
dataset_2['train'] = dataset_2['train'].to_pandas()

df = dataset_2['train']
d2_train_toxic = df[df['toxicity'] == 1]
d2_train_jailbreak = df[df['jailbreaking'] == 1]
d2_train = pd.concat([d2_train_toxic, d2_train_jailbreak]).drop_duplicates()
print(d2_train.shape)

df = dataset_2['test'].to_pandas()
d2_test_toxic = df[df['toxicity'] == 1]
d2_test_jailbreak = df[df['jailbreaking'] == 1]
d2_test = pd.concat([d2_test_toxic, d2_test_jailbreak]).drop_duplicates()
print(d2_test.shape)

# we'll keep only the user inputs
d1_train = d1_train['user_input']
d1_test = d1_test['user_input']
d2_train = d2_train['user_input']
d2_test = d2_test['user_input']

# combine the two datasets and remove duplicates
combined_train = pd.concat([d1_train, d2_train]).drop_duplicates()
combined_test = pd.concat([d1_test, d2_test]).drop_duplicates()

# combine train and test for final dataset
final_dataset = pd.concat([combined_train, combined_test]).drop_duplicates()
print(f"Final dataset shape: {final_dataset.shape}")

# remove any rows that are empty or contain only whitespace
df = df[df['user_input'].str.strip() != '']

# remove punctuation or special characters from the user_input column and make all text lowercase
df['user_input'] = df['user_input'].apply(lambda x: re.sub(r'[^\w\s]', '', x)).str.lower()

# find out text length distribution
df['text_length'] = df['user_input'].apply(len)
print(df['text_length'].describe())

# read the 5 smallest inputs
print(df.nsmallest(5, 'text_length')[['user_input', 'text_length']])

# print the text length distribution
plt.hist(df['text_length'], bins=50, edgecolor='black')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# find out the word count for each input and plot the distribution
df['word_count'] = df['user_input'].apply(lambda x: len(x.split()))
plt.hist(df['word_count'], bins=50, edgecolor='black')
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# find out the most common words in the dataset
all_words = ' '.join(df['user_input']).split()
most_common_words = Counter(all_words).most_common(30)
print("Most common words:")
for word, count in most_common_words:
    print(f"{word}: {count}")

# what percent of inputs have more than 200 words?
long_inputs = df[df['word_count'] > 200]
percent_long = len(long_inputs) / len(df) * 100
print(f"Percentage of inputs with more than 200 words: {percent_long:.2f}%")

# remove the inputs with more than 200 words
df = df[df['word_count'] <= 200]

# remove the text_length and word_count columns and save the cleaned dataset
df = df.drop(columns=['text_length', 'word_count'])
df.to_csv("data/toxic_chat_dataset.csv", index=False)