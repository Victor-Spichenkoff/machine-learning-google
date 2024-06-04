import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


data = {
    'n1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'n2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
'result': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
}

df = pd.DataFrame(data)

#feature enginearing


y = df["result"]
features = ["n1", "n2"]
X = df[["n1", "n2"]]

model = RandomForestRegressor()
model.fit(X, y)

val_X = pd.DataFrame({'n1': [1, 2, 3], 'n2': [1, 3, 6]})
# val_X = pd.DataFrame({'n1': [1, 2, 3], 'n2': [1, 3, 6]})
val_y = [2, 5, 9]


#Model
# prediction = model.predict(val_X)
# prediction_mae = mean_absolute_error(val_y, prediction)



# print("O erro médio foi: ", prediction_mae)
# print(prediction)





print("DEPENDECIA")
from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# score = make_mi_scores(X, y)
score = mutual_info_regression(X, y)
print(score)

# score_df = pd.DataFrame({ "Coluna": X.columns, "Pontos": score })
# score_df = score_df.sort_values(by="Pontos")
# print(score_df)


# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=17)
# model.fit(train_X, train_y)