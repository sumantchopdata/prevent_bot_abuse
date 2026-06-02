#%%
# assign label 0 to toxic chat dataset
# and to banking chatbot adversarial prompts dataset,
# and keep the cleaned data as is

import pandas as pd

def load_dataset(path, label):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'user_input' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'user_input'})
    if 'labels' not in df.columns:
        df['labels'] = label
    return pd.DataFrame({
        'user_input': df['user_input'].astype(str),
        'labels': df['labels']
    })

datasets = [
    ('cleaned_data.csv', 1),
    ('toxic_chat_dataset.csv', 0),
    ('banking_chatbot_adversarial_prompts.csv', 0),
]
combined = pd.concat([load_dataset(path, label) for path, label in datasets],
                     ignore_index=True)
combined
#%%
# clean combined dataset by removing duplicates and empty user_input
combined = combined.drop_duplicates(subset='user_input')
combined = combined[combined['user_input'].str.strip() != '']
print(f'After cleaning, {len(combined)} rows remain in the combined dataset.')
#%%
# find out label distribution
print("Label distribution in the combined dataset:")
print(combined['labels'].value_counts())
# %%
# convert labels to integers
combined['labels'] = combined['labels'].astype(int)
# find the label distribution again after conversion as a ratio of the total dataset
print("Label distribution in the combined dataset (as a ratio):")
print(combined['labels'].value_counts(normalize=True))

# %%
# shuffle the combined dataset
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
print("Shuffled the combined dataset.")
# %%
# split the dataset into train, validation, and test sets (80/10/10)
train_size = int(0.8 * len(combined))
val_size = int(0.1 * len(combined))
train = combined[:train_size]
val = combined[train_size:train_size + val_size]
test = combined[train_size + val_size:]
print(f"Train set size: {len(train)}")
print(f"Validation set size: {len(val)}")
print(f"Test set size: {len(test)}")

# %%
# save the train, validation, and test sets to separate CSV files
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
print("Saved train, validation, and test sets to train.csv, val.csv, and test.csv respectively.")
# %%
# find out the label distribution in the train, validation, and test sets
print("Label distribution in the train set:")
print(train['labels'].value_counts(normalize=True))
print("Label distribution in the validation set:")
print(val['labels'].value_counts(normalize=True))
print("Label distribution in the test set:")
print(test['labels'].value_counts(normalize=True))
# %%
# find out the word count distribution in the train, validation and test sets
train['word_count'] = train['user_input'].apply(lambda x: len(x.split()))
val['word_count'] = val['user_input'].apply(lambda x: len(x.split()))
test['word_count'] = test['user_input'].apply(lambda x: len(x.split()))
print("Word count distribution in the train set:")
print(train['word_count'].describe())
print("Word count distribution in the validation set:")
print(val['word_count'].describe())
print("Word count distribution in the test set:")
print(test['word_count'].describe())
#%%
# find out the top 10 word counts in the train, validation and test sets
print("Top 10 word counts in the train set:")
print(train['word_count'].value_counts().head(10))
print("Top 10 word counts in the validation set:")
print(val['word_count'].value_counts().head(10))
print("Top 10 word counts in the test set:")
print(test['word_count'].value_counts().head(10))
# %%
# plot word count distribution in the combined dataset
import matplotlib.pyplot as plt
combined['word_count'] = combined['user_input'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
plt.hist(combined['word_count'], bins=50, color='blue', alpha=0.7)
plt.title('Word Count Distribution in the Combined Dataset')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
# %%
# plot word count distribution in the train, validation and test sets
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(train['word_count'], bins=50, color='blue', alpha=0.7)
plt.title('Word Count Distribution in the Train Set')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.subplot(1, 3, 2)
plt.hist(val['word_count'], bins=50, color='orange', alpha=0.7)
plt.title('Word Count Distribution in the Validation Set')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.subplot(1, 3, 3)
plt.hist(test['word_count'], bins=50, color='green', alpha=0.7)
plt.title('Word Count Distribution in the Test Set')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
# %%
# there are more longer prompts in train set than in validation and test sets,
# so we will add more longer prompts to the validation and test sets from the combined dataset
long_prompts = combined[combined['word_count'] > 10]
print(len(long_prompts), "long prompts found in the combined dataset.")
#%%
# add 400 long prompts to the validation set and 400 long prompts to the test set
val = pd.concat([val, long_prompts.sample(n=400, random_state=42)], ignore_index=True)
test = pd.concat([test, long_prompts.sample(n=400, random_state=43)], ignore_index=True)
print(f"After adding long prompts, validation set size: {len(val)}, test set size: {len(test)}")

# %%
# now we will shuffle the validation and test sets again after adding long prompts
val = val.sample(frac=1, random_state=42).reset_index(drop=True)
test = test.sample(frac=1, random_state=43).reset_index(drop=True)
print("Shuffled the validation and test sets after adding long prompts.")
# %%
# now check the word count distribution in the validation and test sets again after adding long prompts
print("Word count distribution in the validation set after adding long prompts:")
print(val['word_count'].describe())
print("Word count distribution in the test set after adding long prompts:")
print(test['word_count'].describe())
# %%
# find the lengths of 2% longest prompts in the train dataset
longest_2_percent = train['word_count'].quantile(0.98)
print(f"The length of the 2% longest prompts in the train dataset is: {longest_2_percent} words.")
# %%
# remove the longest prompts from the train set
train = train[train['word_count'] <= longest_2_percent]
print(f"After removing the longest prompts, train set size: {len(train)}")
# %%
# save the train, validation, and test sets to separate CSV files
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
print("Saved the final train, validation, and test sets to train.csv, val.csv, and test.csv respectively.")
# %%
# find the distribution of labels in the final train, validation, and test sets
print("Final label distribution in the train set:")
print(train['labels'].value_counts(normalize=True))
print("Final label distribution in the validation set:")
print(val['labels'].value_counts(normalize=True))
print("Final label distribution in the test set:")
print(test['labels'].value_counts(normalize=True))
# %%
