#%%
# create embeddings for train, val and test sets

import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]

model = SentenceTransformer("all-MiniLM-L6-v2")

#%%
for split in tqdm(['train', 'val', 'test']):
    df = pd.read_csv(f"{split}.csv")
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
    embeddings_df.to_csv(f"{split}_embeddings.csv", index=False, header=False)