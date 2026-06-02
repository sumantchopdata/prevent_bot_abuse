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
