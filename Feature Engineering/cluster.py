import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Criando um conjunto de dados de exemplo
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualizando os dados
# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Dados de Exemplo para Clusterização')
# plt.show()



# Aplicando K-means
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(X)

# Obtendo os centroides e rótulos dos clusters
centroides = kmeans.cluster_centers_
rótulos = kmeans.labels_

# Visualizando os clusters
plt.scatter(X[:, 0], X[:, 1], c=rótulos, s=50, cmap='viridis')
plt.scatter(centroides[:, 0], centroides[:, 1], s=200, c='red', alpha=0.75, marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clusters Formados pelo K-means')
plt.show()
