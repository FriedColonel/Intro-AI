from random import random
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest
import pandas as pd
# import standardScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("PSN_Dataset_06-12_cleaned.csv")

X = data.drop(columns=['Label']).values
y = data['Label'].values

# normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42
)

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

clf = RandomForest(n_trees=20, n_feature=50, max_depth=100)
clf.fit(X_train, y_train)

# save model to file
clf.save_model("model.pkl")

predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(f"Độ chính xác của mô hình: {acc}")
