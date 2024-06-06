import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
# Carregando o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Visualizando os dados
print("Features originais:", iris.feature_names)
print("Primeiras 5 linhas de dados:\n", X[:5])



# Aplicando PCA para reduzir as features para 2 componentes principaisf
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizando a variância explicada pelos componentes principais
print("Variância explicada pelos componentes")



