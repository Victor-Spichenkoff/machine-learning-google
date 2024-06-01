#CRIANDO A BASE
import pandas as pd

# Dados de exemplo
dados = {
    'Nome': ['Ana', 'Bruno', 'Carla', 'Daniel'],
    'Cor dos Olhos': ['azul', 'verde', 'castanho', 'azul'],
    'Tipo de Animal': ['gato', 'cachorro', 'pássaro', 'gato'],
    'Avaliação': ['bom', 'excelente', 'ruim', 'bom']
}

df = pd.DataFrame(dados)




from sklearn.preprocessing import OneHotEncoder

# Inicializando o OneHotEncoder
# one_hot_encoder = OneHotEncoder(sparse=False)
#OU:
one_hot_encoder = OneHotEncoder(drop='first', sparse=False)

# Selecionando as colunas categóricas
colunas_categoricas = ['Cor dos Olhos', 'Tipo de Animal']

# Ajustando e transformando os dados categóricos
one_hot_encoded = one_hot_encoder.fit_transform(df[colunas_categoricas])

# Criando um DataFrame com os dados transformados
df_one_hot = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(colunas_categoricas))

# Concatenando com o DataFrame original, removendo as colunas categóricas originais
df_final = pd.concat([df.drop(columns=colunas_categoricas), df_one_hot], axis=1)


# USANDO ORDINAL ENCONDING
from sklearn.preprocessing import OrdinalEncoder

# Inicializando o OrdinalEncoder com a ordem das categorias
ordem_avalicao = ['péssimo', 'ruim', 'bom', 'excelente']
ordinal_encoder = OrdinalEncoder(categories=[ordem_avalicao])

# Aplicando o Ordinal Encoding
df['Avaliação Encoded'] = ordinal_encoder.fit_transform(df[['Avaliação']])

print(df)

