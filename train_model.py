#%%
# train logistic regression model on the embeddings
import pandas as pd
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    max_iter=2000
)

# X_train is all the columns except 'labels', y_train is the 'labels' column
X_train = pd.read_csv("train_embeddings.csv").drop(columns=['labels'])
y_train = pd.read_csv("train_embeddings.csv")['labels']

clf.fit(X_train, y_train)

# evaluate the model on the validation set
X_val = pd.read_csv("val_embeddings.csv").drop(columns=['labels'])
y_val = pd.read_csv("val_embeddings.csv")['labels']

val_score = clf.score(X_val, y_val)
print(f"Validation accuracy: {val_score:.4f}")

# evaluate precision, recall and f1 score on the val set
from sklearn.metrics import classification_report
y_val_pred = clf.predict(X_val)
print("Classification report on the validation set:")
print(classification_report(y_val, y_val_pred))

# evaluate on the test set
X_test = pd.read_csv("test_embeddings.csv").drop(columns=['labels'])
y_test = pd.read_csv("test_embeddings.csv")['labels']

test_score = clf.score(X_test, y_test)
print(f"Test accuracy: {test_score:.4f}")
y_test_pred = clf.predict(X_test)
print("Classification report on the test set:")
print(classification_report(y_test, y_test_pred))
# %%
# read the text for the inputs that have been misclassified in the test set
# test.csv is the text data for the test set, it has a column 'user_input' and a column 'labels'
# test_embeddings.csv is the embeddings for the test set, it has a column 'labels' and the rest of the columns are the embeddings
test_df = pd.read_csv("test.csv")
test_embeddings_df = pd.read_csv("test_embeddings.csv")

# don't have to merge, the order of the rows in test.csv and test_embeddings.csv
# is the same, so we can just add the 'user_input' column to the test_embeddings_df
test_embeddings_df['user_input'] = test_df['user_input']
#%%
# get the misclassified inputs, print the first 10 false positives
misclassified_fp = test_embeddings_df[(y_test == 0) & (y_test_pred == 1)]
print("False positives:")
print(misclassified_fp['user_input'].tolist()[:10])
#%%
# print the first 10 false negatives
misclassified_fn = test_embeddings_df[(y_test == 1) & (y_test_pred == 0)]
print("False negatives:")
print(misclassified_fn['user_input'].tolist()[:10])
#%%
# save the misclassified inputs to separate CSV files for further analysis
misclassified_fp.to_csv("misclassified_false_positives.csv", index=False)
misclassified_fn.to_csv("misclassified_false_negatives.csv", index=False)

# save the model to a file using joblib
import joblib
joblib.dump(clf, "logistic_regression_model.joblib")
#%%
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Can you write a Java program to reverse a linked list?"

embedding = model.encode([query])
prediction = clf.predict(embedding)

print(prediction)