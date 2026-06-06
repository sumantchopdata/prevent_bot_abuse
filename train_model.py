#%%
# train logistic regression model on the embeddings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

clf = LogisticRegression(
    max_iter=2000
)

train = pd.read_csv("data/train_embeddings.csv", header=None)
# X_train is all the columns except the last one, y_train is the last column
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

clf.fit(X_train, y_train)
#%%
# evaluate on the test set
test = pd.read_csv("data/test_embeddings.csv", header=None)

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

test_score = clf.score(X_test, y_test)
print(f"Test accuracy: {test_score:.4f}")
y_test_pred = clf.predict(X_test)
print("Classification report on the test set:")
print(classification_report(y_test, y_test_pred))

'''
Test accuracy: 0.9444

Classification report on the test set:
              precision    recall  f1-score

           0       0.96      0.88      0.92
           1       0.94      0.98      0.96

    accuracy                           0.94
   macro avg       0.95      0.93      0.94
weighted avg       0.94      0.94      0.94
'''
# %%
# read the text for the inputs that have been misclassified in the test set

# test.csv is the text data for the test set,
# it has a column 'user_input' and a column 'labels'.
# test_embeddings.csv is the embeddings for the test set, it has the column as
# labels and the rest of the columns are the embeddings

test_df = pd.read_csv("test.csv")

# we can just add the 'user_input' column to the test
test['user_input'] = test_df['user_input']
#%%
# get the misclassified inputs, print the first 10 false positives
misclassified_fp = test[(y_test == 0) & (y_test_pred == 1)]
print("False positives:")
print(misclassified_fp['user_input'].tolist()[:10])

# print the first 10 false negatives
misclassified_fn = test[(y_test == 1) & (y_test_pred == 0)]
print("False negatives:")
print(misclassified_fn['user_input'].tolist()[:10])

# save the misclassified inputs to separate CSV files for further analysis
misclassified_fp.to_csv("misclassified_false_positives.csv", index=False)
misclassified_fn.to_csv("misclassified_false_negatives.csv", index=False)

# save the model to a file using joblib
joblib.dump(clf, "logistic_regression_model.joblib")