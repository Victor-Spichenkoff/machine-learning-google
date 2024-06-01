import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Gerando um exemplo de dataset
data = {
     'idade': [25, np.nan, 30, 35, np.nan, 40, 45],
     'salario': [50000, 60000, np.nan, 80000, 90000, 100000, np.nan],
     'cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'São Paulo'],
     'compras': [1, 2, 3, 4, 5, 6, 7]
}

df = pd.DataFrame(data)


#Separando os dados
X = df[['idade', 'salario', 'cidade']]
y = df['compras']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo transformações para as colunas numéricas e categóricas
numeric_features = ['idade', 'salario']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputação de valores faltantes com a média
    ('scaler', StandardScaler())  # Normalização dos dados
])

categorical_features = ['cidade']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputação de valores faltantes com um valor constante
    ('onehot', OneHotEncoder())  # Codificação OneHot das variáveis categóricas
])

# Combinando transformações numéricas e categóricas
preprocessor = ColumnTransformer(
    transformers=[
         ('num', numeric_transformer, numeric_features),
         ('cat', categorical_transformer, categorical_features)
    ])

# Criando o Pipeline completo com pré-processamento e modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Treinando o Pipeline
pipeline.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliando o modelo usando MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')