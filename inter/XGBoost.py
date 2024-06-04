# Importando bibliotecas
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import numpy as np

# Carregar o dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Definir os parâmetros para a busca
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Criar o modelo XGBRegressor
xgb_model = XGBRegressor(random_state=42)

# Configurar o GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=1)

# Executar o GridSearchCV
grid_search.fit(X_train, y_train)

# Obter os melhores parâmetros
best_params = grid_search.best_params_
print(f'Melhores Parâmetros: {best_params}')

# Criar o modelo com os melhores parâmetros
best_xgb_model = XGBRegressor(**best_params, random_state=42)

# Treinar o modelo
best_xgb_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = best_xgb_model.predict(X_test)

# Calcular o Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error com Melhores Parâmetros: {mae}')