import pandas as pd
import sklearn.tree

melb_path = "./melb_data.csv"
dataFrame_melb = pd.read_csv(melb_path)

y = dataFrame_melb.Price


melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = dataFrame_melb[melb_features]


# print(X.describe())
# print("HEAD _-------------")
# print(X.head())


# CRIAR MODELO
# 
#

from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))