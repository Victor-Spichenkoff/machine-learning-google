# tem 2 colunas e deve descobrir como chega no valor final
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_t
from sklearn.tree import DecisionTreeRegressorest_split
from sklearn.metrics import mean_absolute_error

data = {
    'coluna1': [1, 2, 3, 4, 5],
    'coluna2': [2, 3, 4, 5, 6],
    'target': [3, 4, 5, 6, 7]}
df = pd.DataFrame(data)

X = df[['coluna1', 'coluna2']]
y = df['target']

train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 0)
my_model = DecisionTreeRegressor(random_state=1)

#pegar o melhor (nesse, não muda nada)

# def get_medium_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y)
#     preds_val = model.predict(val_X)
#     mae = mean_absolute_error(val_y, preds_val)
#     return (mae)
#
#
# def get_best():
#     best = 0
#     best_score=1000
#     last_leaft_tested = 2
#     for vezes in [100]:
#         current_mae = get_medium_absolute_error(last_leaft_tested, train_X, validation_X, train_y, validation_y)
#         if(current_mae < best_score):
#             best_score = current_mae
#             best = last_leaft_tested
#     return best
#
#
# best = get_best()



my_model.fit(X=train_X, y=train_y)

#modelo já treinado
#salva o Y a partir de dados de validação (testa a revisão)
validation_predictions = my_model.predict(validation_X)

#Ver o desvio/erro medio
mae = mean_absolute_error(validation_y, validation_predictions)
print(f"O erro médio foi: {mae}")
print(f"Previsão: {my_model.predict(X)}")