import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([1, 2, 3, 4, 5], dtype=float)#açucar em gramas
y = np.array([92.5, 95, 97.5, 100, 102.5], dtype=float)#calorias por porção



model = Sequential()

# Adicionando uma camada densa com uma unidade (neurônio) e input_shape = [1]
# A função de ativação é linear por padrão, então não precisamos especificar
model.add(Dense(units=1, input_shape=[1]))

# Compilando o modelo com otimizador e função de perda
model.compile(optimizer='sgd', loss='mean_squared_error')

# Treinando o modelo (ajustando os pesos)
model.fit(X, y, epochs=500, verbose=0)

# Fazendo uma previsão com o modelo treinado
# Por exemplo, queremos prever as calorias para um cereal com 5 gramas de açúcar
sugar_amount = 5.0
predicted_calories = model.predict([sugar_amount])

print(f"Calorias previstas para um cereal com {sugar_amount} gramas de açúcar: {predicted_calories[0][0]:.2f}")