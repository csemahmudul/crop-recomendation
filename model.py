#All Code was created by joy


import pandas as pd
from pandas._libs import arrays
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle

data = pd.read_csv('datasets.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

new_features = [[12, 22, 55, 28.6604, 53.31891, 10.399136, 156.9263]]

predicted_crop = model.predict(new_features)

print("Predicted crop:", predicted_crop)

pickle.dump(model, open("joy.pkl", "wb"))