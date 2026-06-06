#%%
# reduce the embeddings to 2d and plot them
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

reducer = UMAP(
    n_components=2,
    random_state=42
)

embeddings = pd.read_csv("data/train_embeddings.csv")
embeddings = embeddings.iloc[:, :-1].values
embeddings_2d = reducer.fit_transform(embeddings)
# %%
# plot the 2d embeddings, color by the labels
labels = np.array(pd.read_csv("data/train_embeddings.csv").iloc[:, -1].values)

plt.figure(figsize=(10,8))

plt.scatter(
    embeddings_2d[labels == 0, 0],
    embeddings_2d[labels == 0, 1],
    alpha=0.6,
    label="Misuse"
)

plt.scatter(
    embeddings_2d[labels == 1, 0],
    embeddings_2d[labels == 1, 1],
    alpha=0.6,
    label="Genuine"
)

plt.legend()
plt.title("UMAP projection of text embeddings")
plt.savefig("plots/umap_projection.png")
plt.show()
# %%
clf = joblib.load("logistic_regression_model.joblib")

preds = np.array(clf.predict(embeddings))

plt.figure(figsize=(10,8))

plt.scatter(
    embeddings_2d[labels == 0, 0],
    embeddings_2d[labels == 0, 1],
    alpha=0.2,
    label="Misuse"
)

plt.scatter(
    embeddings_2d[preds == 0, 0],
    embeddings_2d[preds == 0, 1],
    alpha=0.2,
    label="Predicted Misuse"
)

plt.legend()
plt.title("UMAP projection of misuse predictions")
plt.savefig("plots/umap_projection_predictions.png")
plt.show()
#%%
# plot genuine predictions
plt.figure(figsize=(10,8))
plt.scatter(
    embeddings_2d[labels == 1, 0],
    embeddings_2d[labels == 1, 1],
    alpha=0.1,
    label="Genuine"
)

plt.scatter(
    embeddings_2d[preds == 1, 0],
    embeddings_2d[preds == 1, 1],
    alpha=0.1,
    label="Predicted Genuine"
)

plt.legend()
plt.title("UMAP projection of genuine predictions")
plt.savefig("plots/umap_projection_genuine_predictions.png")
plt.show()
#%%
# highlight false positive errors and false negative errors in the plot
false_positives = (labels == 0) & (preds == 1)
false_negatives = (labels == 1) & (preds == 0)

plt.figure(figsize=(10,8))

plt.scatter(
    embeddings_2d[labels == 0, 0],
    embeddings_2d[labels == 0, 1],
    alpha=0.2,
    label="Misuse",
    color='green'
)

plt.scatter(
    embeddings_2d[labels == 1, 0],
    embeddings_2d[labels == 1, 1],
    alpha=0.2,
    label="Genuine",
    color='violet'
)

plt.scatter(
    embeddings_2d[false_positives, 0],
    embeddings_2d[false_positives, 1],
    alpha=0.2,
    label="False Positives",
    color='red'
)

plt.scatter(
    embeddings_2d[false_negatives, 0],
    embeddings_2d[false_negatives, 1],
    alpha=0.2,
    label="False Negatives",
    color='blue'
)

plt.legend()
plt.title("UMAP projection of text embeddings with misclassifications highlighted")
plt.savefig("plots/umap_projection_misclassifications.png")
plt.show()