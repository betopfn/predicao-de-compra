import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', 500)
data = pd.read_excel('Online Retail.xlsx')

# --- 1. Pré-processamento e limpeza de todo o dataset ---
# A ordem é crucial. Remova nulos primeiro, depois converta tipos.
data.dropna(subset=['CustomerID', 'InvoiceNo'], inplace=True)
data['CustomerID'] = data['CustomerID'].astype(int)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Receita'] = data['Quantity'] * data['UnitPrice']
data = data[data['InvoiceNo'].str.contains('C', na=False) == False]
data = data.drop_duplicates()

# 2. Definição da data de corte e divisão dos dados
data_de_corte = '2011-12-01'
df_treinamento = data[data['InvoiceDate'] < data_de_corte].copy()
df_previsao = data[data['InvoiceDate'] >= data_de_corte].copy()

# 3. Cálculo das métricas RFM APENAS com os dados de TREINAMENTO
ultima_data_treinamento = df_treinamento['InvoiceDate'].max()
data_referencia_treinamento = ultima_data_treinamento + pd.DateOffset(days=1)

recencia_df = df_treinamento.groupby('CustomerID')['InvoiceDate'].max().reset_index()
recencia_df['Recencia'] = (data_referencia_treinamento - recencia_df['InvoiceDate']).dt.days

frequencia_df = df_treinamento.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequencia_df.columns = ['CustomerID', 'Frequencia']

monetario_df = df_treinamento.groupby('CustomerID')['Receita'].sum().reset_index()
monetario_df.columns = ['CustomerID', 'Monetario']

# 4. Criação do DataFrame RFM COMPLETO e da variável alvo
rfm_completo = pd.merge(recencia_df, frequencia_df, on='CustomerID')
rfm_completo = pd.merge(rfm_completo, monetario_df, on='CustomerID')

clientes_que_compraram_futuro = df_previsao.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
lista_clientes_futuro = clientes_que_compraram_futuro['CustomerID'].tolist()
rfm_completo['Fez_compra'] = rfm_completo['CustomerID'].isin(lista_clientes_futuro).astype(int)

# Verificação de dados antes de treinar
print(f"Número de clientes no DataFrame RFM: {len(rfm_completo)}")
print(f"Número de clientes que compraram no futuro: {len(lista_clientes_futuro)}")

# 5. Treinamento e avaliação do modelo
X = rfm_completo[["Recencia", "Frequencia", "Monetario"]]
Y = rfm_completo["Fez_compra"]

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, Y_treino)

previsoes = modelo.predict(X_teste)

print('---')
print('Acurácia do modelo:', accuracy_score(Y_teste, previsoes))
print('\nRelatório de Classificação:\n', classification_report(Y_teste, previsoes))
print('---')

# 6. Criação da tabela de resultados
resultados_df = X_teste.copy()
resultados_df['Fez_compra_REAL'] = Y_teste
resultados_df['Fez_compra_PREVISAO'] = previsoes

print('\n--- Tabela de Resultados do Modelo ---')
print(resultados_df.head())

resultados_df.to_csv('relatorio_previsoes.csv', index=False)