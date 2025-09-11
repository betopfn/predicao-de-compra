import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', 500)
data = pd.read_excel('Online Retail.xlsx')

#Preparação do dataset
# Excluir Nan 
data_limpo = data.dropna().copy()
# Remover duplicaas
data_limpo = data_limpo.drop_duplicates
# Retira faturas canceladas
data_limpo = data_limpo[data_limpo['InvoiceNo'].str.contains('C') == False].copy()

# Preparação de dados
# Encontrar a ultima data geral
ultima_data = data_limpo['InvoiceDate'].max()
# Criar data de referencia, que sera 1 dia depois da ultima data
data_referencia = ultima_data + pd.DateOffset(days=1)

# Calculo Recencia
# Encontrar a ultima data de compra por pessoa
ultima_compra_por_pessoa = data_limpo.groupby('CustomerID')['InvoiceDate'].max().reset_index()
# Calculo de recencia(diferença entre a data de referencia e última compra)
ultima_compra_por_pessoa['Recencia'] = (data_referencia - ultima_compra_por_pessoa['InvoiceDate']).dt.days

# Calculo de Frquência
frequencia_por_pessoa = data_limpo.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequencia_por_pessoa.columns = ['CustomerID', 'Frequencia']

# Calculo de monetário
# Primeirpo: Coluna de total por transação
data_limpo['Receita'] = data_limpo['Quantity'] * data_limpo['UnitPrice']
# Segundo: Agrupamento de dados
monetário_por_pessoa = data_limpo.groupby('CustomerID')['Receita'].sum().reset_index()

# Criação do data frame para ensino do modelo de learning
rf_data = pd.merge(ultima_compra_por_pessoa, frequencia_por_pessoa, on='CustomerID')
rfm_completo = pd.merge(rf_data, monetário_por_pessoa, on='CustomerID')
# Definição de data de corte
data_de_corte = '2011-12-01'
# Data do período de treinamento
df_treinamento = data_limpo[data_limpo['InvoiceDate'] < data_de_corte].copy()
# Data de previsão
df_previsao = data_limpo[data_limpo['InvoiceDate'] >= data_de_corte].copy()
# Agrupar por CustomerID e contar faturas únicas
clientes_que_compraram_futuro = df_previsao.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
# Lista do DF dos clientes que compraram no futuro 
lista_clientes_futuro = clientes_que_compraram_futuro['CustomerID'].tolist()
# Coluna de verificação de compra
rfm_completo['Fez_compra'] = rfm_completo['CustomerID'].isin(lista_clientes_futuro).astype(int)

# Ensinando o modelo
# Variáveis independentes
X = rfm_completo[["Recencia", "Frequencia", "Receita"]]
# Variavél dependente
Y = rfm_completo["Fez_compra"]

# Divisão do adataset em treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42 )

# Prevendo com o Random Forest Classifier
# Criação da instância
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
# Ensinando o modelo
modelo.fit(X_treino, Y_treino)
# Prvisões no conjunto de testes
previsoes = modelo.predict(X_teste)

# Avalie o desempenho do modelo
print('Acurácia do modelo:', accuracy_score(Y_teste, previsoes))
print('\nRelatório de Classificação:\n', classification_report(Y_teste, previsoes))

# Criação dos data frames finais
rfm_completo.to_csv('rfm_completo.csv', index=False)
print(rfm_completo.head())
