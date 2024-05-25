import pandas as pd

# Carregar dados
melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Filtar linhas com valores ausentes
melbourne_data = melbourne_data.dropna(axis=0)

# Escolher alvo e variaveis
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# dividir os dados em dados de treinamento e validação, tanto para características quanto para alvo
# A divisão é baseada em um gerador de números aleatórios. Fornecer um valor numérico para
# o argumento random_state garante que obtemos a mesma divisão todas as vezes que
# rodamos o script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)








# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))