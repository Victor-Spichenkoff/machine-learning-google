import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# data = {
#     'n1': [1, 2, 3, 4, 5],
#     'n2': [2, 3, 4, 5, 6],
#     'result': [3, 5, 7, 9, 11]}
data = {
    'n1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'n2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
'result': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
}

df = pd.DataFrame(data)

y = df["result"]
features = ["n1", "n2"]
X = df[features]

model = DecisionTreeRegressor(random_state=17, max_leaf_nodes=1000)
model.fit(X, y)

val_X = pd.DataFrame({'n1': [1, 2, 3], 'n2': [1, 3, 6]})
val_y = [2, 5, 9]

prediction = model.predict(val_X)
prediction_mae = mean_absolute_error(val_y, prediction)



print("O erro médio foi: ", prediction_mae)
print(prediction)


# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=17)
# model.fit(train_X, train_y)