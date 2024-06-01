import pandas as pd

casas_df = pd.read_csv('./data/houses.csv')
valores_ausentes = casas_df.isnull().sum()
print(valores_ausentes)

#IMPUTAÇõES
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

imputer = KNNImputer(n_neighbors=5)
df_imputado = imputer.fit_transform(casas_df)

# Exemplo simplificado para imputação por regressão
# Selecionando colunas de features e a coluna alvo
features = casas_df[['feature1', 'feature2']]
alvo = casas_df['coluna'].dropna()
features = features.loc[alvo.index]

# Treinando o modelo de regressão
modelo = LinearRegression()
modelo.fit(features, alvo)

# Prevendo valores ausentes
predicoes = modelo.predict(casas_df[['feature1', 'feature2']][casas_df['coluna'].isna()])

# Substituindo valores ausentes pelas previsões
casas_df.loc[casas_df['coluna'].isna(), 'coluna'] = predicoes