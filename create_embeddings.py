#%%
# create embeddings for train and test sets

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
model = SentenceTransformer("all-MiniLM-L6-v2")

for split in tqdm(['train', 'test']):
    df = pd.read_csv(f"data/{split}.csv")
    texts = df["user_input"].tolist()
    labels = df["labels"].tolist()
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )
    print(embeddings.shape)
    # save the embeddings and labels to a new CSV file without any column headers
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['labels'] = labels
    embeddings_df.to_csv(f"data/{split}_embeddings.csv", index=False, header=False)
#%%
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
model = SentenceTransformer("all-MiniLM-L6-v2")

for split in tqdm(['train', 'eval', 'test']):
    df = pd.read_csv(f"claude_datasets/combined_{split}_v4_balanced.csv")
    texts = df["text"].tolist()
    labels = df["labels"].tolist()
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )
    print(embeddings.shape)

    # save the embeddings and labels to a new CSV file without any column headers
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['labels'] = labels
    embeddings_df.to_csv(f"claude_datasets/{split}_embeddings.csv", index=False, header=False)