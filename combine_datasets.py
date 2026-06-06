#%%
# assign label 0 to toxic chat dataset
# and to banking chatbot adversarial prompts dataset,
# and keep the cleaned data as is

import pandas as pd
import matplotlib.pyplot as plt

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
    ('data/cleaned_data.csv', 1),
    ('data/toxic_chat_dataset.csv', 0),
    ('data/banking_chatbot_adversarial_prompts.csv', 0),
]
combined = pd.concat([load_dataset(path, label) for path, label in datasets],
                     ignore_index=True)

# clean combined dataset by removing duplicates and empty user_input
combined = combined.drop_duplicates(subset='user_input')
combined = combined[combined['user_input'].str.strip() != '']

# find out label distribution
combined['labels'] = combined['labels'].astype(int)
print(combined['labels'].value_counts(normalize=True))

# shuffle the combined dataset
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# split the dataset into train and test sets (80/20)
train_size = int(0.8 * len(combined))
train = combined[:train_size]
test = combined[train_size:]
print(f"Train set size: {len(train)}")
print(f"Test set size: {len(test)}")

# save the train, validation, and test sets to separate CSV files
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

# find out the label distribution in the train, validation, and test sets
print(train['labels'].value_counts(normalize=True))
print(test['labels'].value_counts(normalize=True))

# find out the word count distribution in the train, test sets
train['word_count'] = train['user_input'].apply(lambda x: len(x.split()))
test['word_count'] = test['user_input'].apply(lambda x: len(x.split()))

print(train['word_count'].describe())
print(test['word_count'].describe())

# find out the top 10 word counts in the train, test sets
print(train['word_count'].value_counts().head(10))
print(test['word_count'].value_counts().head(10))

# plot word count distribution in the combined dataset
combined['word_count'] = combined['user_input'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
plt.hist(combined['word_count'], bins=50, color='blue', alpha=0.7)
plt.title('Word Count Distribution in the Combined Dataset')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# plot word count distribution in the train and test sets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(train['word_count'], bins=50, color='blue', alpha=0.7)
plt.title('Word Count Distribution in the Train Set')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.subplot(1, 2, 2)
plt.hist(test['word_count'], bins=50, color='green', alpha=0.7)
plt.title('Word Count Distribution in the Test Set')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# there are more longer prompts in train set than in test sets,
# so we will add more longer prompts to the test sets from the combined dataset
long_prompts = combined[combined['word_count'] > 15]
print(len(long_prompts), "long prompts found in the combined dataset.")

# add 200 long prompts to the test set
test = pd.concat([test, long_prompts.sample(n=200, random_state=43)],
                 ignore_index=True)
print(f"After adding long prompts, test set size: {len(test)}")

# now we will shuffle the test sets again after adding long prompts
test = test.sample(frac=1, random_state=43).reset_index(drop=True)

# now check the word count distribution in the test sets again after adding long prompts
print("Word count distribution in the test set after adding long prompts:")
print(test['word_count'].describe())

# find the lengths of 2% longest prompts in the train dataset
longest_2_percent = train['word_count'].quantile(0.98)
print(f"The length of the 2% longest prompts in the train dataset is: {longest_2_percent} words.")

# remove the longest prompts from the train set
train = train[train['word_count'] <= longest_2_percent]
print(f"After removing the longest prompts, train set size: {len(train)}")

# remove the inputs from train which are present in the test
print(train.shape)
train = train[~train['user_input'].isin(test['user_input'])]
print(train.shape)

# find the lengths of 2% longest prompts in the test dataset
longest_2_percent = test['word_count'].quantile(0.98)
print(f"The length of the 2% longest prompts in the test dataset is: {longest_2_percent} words.")

# remove the longest prompts from the test set
test = test[test['word_count'] <= longest_2_percent]
print(f"After removing the longest prompts, test set size: {len(test)}")

# find the distribution of labels in the final train, validation, and test sets
print("Final label distribution in the train set:")
print(train['labels'].value_counts(normalize=True))
print("Final label distribution in the test set:")
print(test['labels'].value_counts(normalize=True))

# remove the word_count column from the train, test sets before saving
train.drop(columns=['word_count']).to_csv("data/train.csv", index=False)
test.drop(columns=['word_count']).to_csv("data/test.csv", index=False)