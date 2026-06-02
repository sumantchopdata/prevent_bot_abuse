#%%
import pandas as pd
from datasets import load_dataset
# %%
from datasets import load_dataset

dataset_1 = load_dataset("lmsys/toxic-chat", 'toxicchat0124')

print(dataset_1["train"][0])
#%%
df = dataset_1['train'].to_pandas()
print(type(df))
# %%

# we'll look for toxic and jailbreaking examples
d1_train_toxic = df[df['toxicity'] == 1]
d1_train_jailbreak = df[df['jailbreaking'] == 1]
d1_train = pd.concat([d1_train_toxic, d1_train_jailbreak]).drop_duplicates()
print(d1_train.shape)
# %%
df = dataset_1['test'].to_pandas()
d1_test_toxic = df[df['toxicity'] == 1]
d1_test_jailbreak = df[df['jailbreaking'] == 1]
d1_test = pd.concat([d1_test_toxic, d1_test_jailbreak]).drop_duplicates()
print(d1_test.shape)
# %%
dataset_2 = load_dataset("lmsys/toxic-chat", 'toxicchat1123')
dataset_2['train'] = dataset_2['train'].to_pandas()
#%%
df = dataset_2['train']
d2_train_toxic = df[df['toxicity'] == 1]
d2_train_jailbreak = df[df['jailbreaking'] == 1]
d2_train = pd.concat([d2_train_toxic, d2_train_jailbreak]).drop_duplicates()
print(d2_train.shape)
# %%
df = dataset_2['test'].to_pandas()
d2_test_toxic = df[df['toxicity'] == 1]
d2_test_jailbreak = df[df['jailbreaking'] == 1]
d2_test = pd.concat([d2_test_toxic, d2_test_jailbreak]).drop_duplicates()
print(d2_test.shape)
#%%
d1_train = d1_train['user_input']
d1_test = d1_test['user_input']
d2_train = d2_train['user_input']
d2_test = d2_test['user_input']
# %%
# check if there are any duplicates between the two datasets
train_duplicates = set(d1_train).intersection(set(d2_train))
test_duplicates = set(d1_test).intersection(set(d2_test))
print(f"Number of duplicate examples in train: {len(train_duplicates)}")
print(f"Number of duplicate examples in test: {len(test_duplicates)}")
# %%
# combine the two datasets and remove duplicates
combined_train = pd.concat([d1_train, d2_train]).drop_duplicates()
combined_test = pd.concat([d1_test, d2_test]).drop_duplicates()
print(f"Combined train shape: {combined_train.shape}")
print(f"Combined test shape: {combined_test.shape}")
# %%
# combine train and test for final dataset
final_dataset = pd.concat([combined_train, combined_test]).drop_duplicates()
print(f"Final dataset shape: {final_dataset.shape}")
# %%
final_dataset.to_csv("toxic_chat_dataset.csv", index=False)
#%%
# clean toxic chat dataset
import pandas as pd

df = pd.read_csv("toxic_chat_dataset.csv", dtype=str, keep_default_na=False)
df
# %%
# clean the dataset by removing any rows that are empty or contain only whitespace
df = df[df['user_input'].str.strip() != '']
print(f"Cleaned dataset shape: {df.shape}")
# %%
# remove punctuation or special characters from the user_input column
import re
df['user_input'] = df['user_input'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
print("Removed punctuation and special characters.")
# %%
# make all text lowercase
df['user_input'] = df['user_input'].str.lower()
print("Converted text to lowercase.")
# %%
# find out text length distribution
df['text_length'] = df['user_input'].apply(len)
print("Calculated text length distribution.")
df['text_length'].describe()
# %%
# read the 5 smallest inputs
print("5 smallest inputs:")
print(df.nsmallest(5, 'text_length')[['user_input', 'text_length']])

# %%
# plot the text length distribution
import matplotlib.pyplot as plt
plt.hist(df['text_length'], bins=50, edgecolor='black')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()
# %%
# find out the word count for each input and plot the distribution
df['word_count'] = df['user_input'].apply(lambda x: len(x.split()))
print("Calculated word count distribution.")
plt.hist(df['word_count'], bins=50, edgecolor='black')
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()
# %%
# find out the most common words in the dataset
from collections import Counter
all_words = ' '.join(df['user_input']).split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(30)
print("Most common words:")
for word, count in most_common_words:
    print(f"{word}: {count}")
# %%
# what percent of inputs have more than 200 words?
long_inputs = df[df['word_count'] > 200]
percent_long = len(long_inputs) / len(df) * 100
print(f"Percentage of inputs with more than 200 words: {percent_long:.2f}%")
# %%
# remove the inputs with more than 200 words
df = df[df['word_count'] <= 200]
print(f"Removed inputs with more than 200 words. New dataset shape: {df.shape}")
# %%
# remove the text_length and word_count columns and save the cleaned dataset
df = df.drop(columns=['text_length', 'word_count'])
df
#%%
df.to_csv("toxic_chat_dataset.csv", index=False)
# %%
import pandas as pd
df = pd.read_csv("toxic_chat_dataset.csv", dtype=str, keep_default_na=False)

# print the 5 smallest inputs
# %%
print("5 smallest inputs:")
print(df.nsmallest(5, 'user_input')[['user_input']])
# %%
# convert user_input to string type
df['user_input'] = df['user_input'].astype(str)
print("Converted user_input to string type.")
# %%
# which rows don't have user_input as a string?
non_string_inputs = df[~df['user_input'].apply(lambda x: isinstance(x, str))]
print(f"Number of non-string inputs: {len(non_string_inputs)}")
# %%
# find out smallest inputs again
print("5 smallest inputs after conversion:")
print(df.nsmallest(5, 'user_input')[['user_input']])
# %%
# deduplicate the dataset again just in case
df = df.drop_duplicates()
print(f"Deduplicated dataset shape: {df.shape}")
# %%
df.user_input = df.user_input.str.strip()
print("Stripped whitespace from user_input.")
df.user_input.str.len().describe()

# %%
# find out the 5 smallest inputs again
print("5 smallest inputs after stripping whitespace:")
print(df.nsmallest(5, 'user_input')[['user_input']])
# %%
df.iloc[0:5]
# %%
# sort the dataset by user_input length and find out the 5 smallest inputs again
df['input_length'] = df['user_input'].apply(len)
df = df.sort_values(by='input_length')
print("5 smallest inputs after sorting by length:")
print(df.iloc[0:5][['user_input', 'input_length']])
# %%
# save the cleaned dataset again without the input_length column
df = df.drop(columns=['input_length'])
df.to_csv("toxic_chat_dataset.csv", index=False)
# %%
