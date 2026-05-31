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