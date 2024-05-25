# tentar encontrar o mehor node tree limiter
#Resumo de como criar o modelo
#1 - Criar o Data Frame -> pd.read_csv()
#2 - Remover os vazios: filtered_melb_df = melb_df.dropna(axis=0)
#3 - definir y(qual vai prever) e X (colunas que vai usar) df["c1", "c4"] --> normal, não tem separação
#4 - Criar modelo(a partir do tree decison regressor) e usar o FIT nele (Scikit-learn)
#--- fit == ajusta internamente o model para prever y a partir de uma linha de X (de treinamento)
#5 - Evitar a previsão interna
#---




import pandas as pd

melb_path = "./melb_data.csv"
melb_df = pd.read_csv(melb_path)



filtered_melb_df = melb_df.dropna(axis=0)


from sklearn.tree import DecisionTreeRegressor

#Criar o terget e o que vai usar para calculcar
melb_features =  ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
y = filtered_melb_df.Price
X = filtered_melb_df[melb_features]

#Criar o modelo
melb_model = DecisionTreeRegressor(random_state=1)

melb_model.fit(X, y)
# print("Previsão:")
# print(melb_model.predict(X.head(5)))#faz a revisão de preço final para os 5 primeiros

predicted_home_prices = melb_model.predict(X)


#Verificar precisão
from sklearn.metrics import mean_absolute_error
print("Valor de desvio(interno)",mean_absolute_error(y, predicted_home_prices))#Verifcar o quanto desvia na media, com relação ao valor real


#Lidar com o problema de previsões "internas"
from sklearn.model_selection import train_test_split

#separa em grupos de controle e de treino
train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 0)

#melb_model = DecisionTreeRegressor()#criação
melb_model.fit(train_X, train_y)#adiciona ao modelo o validationor X e y (substitui)

validation_predictions = melb_model.predict(validation_X)

#recebe o valor previsto e o valor real
print(f"Valor real de desvio: {mean_absolute_error(validation_y, validation_predictions)}")



#lidar com FITTINGS errados
#faz o mesmo que o restante (cria o modelo->fazer o fit -> previsão de controle -> pegar o erro medio/)
def get_medium_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


#escolhe um valor inicial e vai testando, o melhor foi o 791
best = 0
best_mae = 10000000000
test_leaf_num = 700#inicial
while test_leaf_num < 800:
    last_mae = get_medium_absolute_error(test_leaf_num, train_X, validation_X, train_y, validation_y)
    if last_mae < best_mae:
        best = test_leaf_num
        best_mae = last_mae
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (test_leaf_num, last_mae))
    test_leaf_num +=1

print(f"O melhor foi: {best}")
#melhor: Max leaf nodes: 791  		 Mean Absolute Error:  259763