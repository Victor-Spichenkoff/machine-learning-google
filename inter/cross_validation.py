import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Carregar o conjunto de dados Iris
dados = load_iris()
X = dados.data
y = dados.target
# Definir o modelo
modelo = DecisionTreeClassifier()


resultados = cross_val_score(modelo, X, y, cv=5)

# Exibir os resultados
print("Acurácias em cada fold: ", resultados)#Resultado bruto
print("Acurácia média: ", np.mean(resultados))#media de acertos
print("Desvio padrão: ", np.std(resultados))#desvio padrão

