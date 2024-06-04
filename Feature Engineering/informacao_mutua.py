import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
df = pd.read_csv(url)
df.head()


X = df.copy()
y = X.pop("strength")

# Treinando e avaliando o modelo de linha de base
baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)



from sklearn.feature_selection import mutual_info_regression

# Suponha que X é o dataframe de features e y é a variável alvo
mi_scores = mutual_info_regression(X, y)

# Convertendo os scores para um dataframe para facilitar a visualização
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)
print(mi_scores_df)