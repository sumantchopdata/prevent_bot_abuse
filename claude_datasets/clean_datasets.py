# clean all the datasets in this folder and save as train, eval and test sets.
#%%
import pandas as pd
import os
#%%
from tqdm import tqdm
# read name category mapping
with open("name_category_mapping.txt", "r") as f:
    name_category_dict = dict(line.strip().split("-") for line in f)

# create a column for category in each dataset and save as cleaned csv in a new folder called cleaned_datasets
for filename in tqdm(os.listdir()):
    if not filename.endswith(".csv"):
        continue
    df = pd.read_csv(filename)
    category = name_category_dict.get(filename)
    df["category"] = category
    new_filename = f"{category}_{filename}"
    df.to_csv(f"cleaned_datasets/{new_filename}_cleaned.csv", index=False)
# %%
# split each cleaned dataset into train, eval and test sets and save as csv
from sklearn.model_selection import train_test_split

for filename in tqdm(os.listdir("cleaned_datasets")):
    if not filename.endswith("_cleaned.csv"):
        continue
    df = pd.read_csv(f"cleaned_datasets/{filename}")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df.to_csv(f"split_datasets/train_{filename}", index=False)
    eval_df.to_csv(f"split_datasets/eval_{filename}", index=False)
    test_df.to_csv(f"split_datasets/test_{filename}", index=False)
#%%
# combine all the train, eval and test sets into one csv each and
# save as combined_train.csv, combined_eval.csv and combined_test.csv
train_dfs = []
eval_dfs = []
test_dfs = []

for filename in tqdm(os.listdir("split_datasets")):
    if filename.startswith("train_") and filename.endswith("_cleaned.csv"):
        train_dfs.append(pd.read_csv(f"split_datasets/{filename}"))
    elif filename.startswith("eval_") and filename.endswith("_cleaned.csv"):
        eval_dfs.append(pd.read_csv(f"split_datasets/{filename}"))
    elif filename.startswith("test_") and filename.endswith("_cleaned.csv"):
        test_dfs.append(pd.read_csv(f"split_datasets/{filename}"))

combined_train_df = pd.concat(train_dfs, ignore_index=True)
combined_eval_df = pd.concat(eval_dfs, ignore_index=True)
combined_test_df = pd.concat(test_dfs, ignore_index=True)

combined_train_df.to_csv("combined_train.csv", index=False)
combined_eval_df.to_csv("combined_eval.csv", index=False)
combined_test_df.to_csv("combined_test.csv", index=False)
#%%
# turn the labels from float to int if they are not NA or inf
for filename in ["combined_train.csv", "combined_eval.csv", "combined_test.csv"]:
    df = pd.read_csv(filename)
    if df["labels"].dtype == float:
        df["labels"] = df["labels"].apply(lambda x: int(x) if pd.notna(x) and x != float("inf") else x)
    df.to_csv(filename, index=False)
#%%
# inspect the combined csv files and print the number of rows and columns
for filename in ["combined_train.csv", "combined_eval.csv", "combined_test.csv"]:
    df = pd.read_csv(filename)
    print(f"{filename}: {df.shape[0]} rows, {df.shape[1]} columns")
#%%
# count the number of rows with NA or inf in the labels column and print the percentage of such rows
for filename in ["combined_train.csv", "combined_eval.csv", "combined_test.csv"]:
    df = pd.read_csv(filename)
    na_inf_count = df["labels"].apply(lambda x: pd.isna(x) or x == float("inf")).sum()
    percentage = na_inf_count / df.shape[0] * 100
    print(f"{filename}: {na_inf_count} rows with NA or inf in labels column ({percentage:.2f}%)")
#%%
# print the number of unique categories in each combined csv file
for filename in ["combined_train.csv", "combined_eval.csv", "combined_test.csv"]:
    df = pd.read_csv(filename)
    unique_categories = df["category"].nunique()
    print(f"{filename}: {unique_categories} unique categories")

#%%
# print the number of unique labels in each combined csv file
for filename in ["combined_train.csv", "combined_eval.csv", "combined_test.csv"]:
    df = pd.read_csv(filename)
    unique_labels = df["labels"].nunique()
    print(f"{filename}: {unique_labels} unique labels")

#%%
# print the percentage of rows with label 1 and those with label 0
for filename in ["combined_train.csv", "combined_eval.csv", "combined_test.csv"]:
    df = pd.read_csv(filename)
    label_1_count = (df["labels"] == 1).sum()
    label_0_count = (df["labels"] == 0).sum()
    total_count = df.shape[0]
    label_1_percentage = label_1_count / total_count * 100
    label_0_percentage = label_0_count / total_count * 100
    print(f"{filename}: {label_1_percentage:.2f}% rows with label 1, {label_0_percentage:.2f}% rows with label 0")
#%%
from tqdm import tqdm
# Add the category name to each csv file
for file in tqdm(os.listdir()):
    if "cleaned" in file and file.endswith(".csv"):
        df = pd.read_csv(file)
        category = name_category_dict.get(file.replace("cleaned_", "").replace("train_", "").replace("eval_", "").replace("test_", ""))
        new_filename = file.replace("cleaned_", f"{category}_cleaned_")
        df.to_csv(new_filename, index=False)
        os.remove(file)