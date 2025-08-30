import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', 500)
data = pd.read_excel('Online Retail.xlsx')

#Preparação do dataset
# Excluir Nan 
data_limpo = data.dropna().copy()
# Remover duplicaas
data_limpo = data_limpo.drop_duplicates
# Retira faturas canceladas
data_limpo = data_limpo[data_limpo['InvoiceNo'].str.contains('C') == False].copy()
# Calculo de Recencia
# Encontrar a ultima data geral
ultima_data = data_limpo['InvoiceDate'].max()
# Criar data de referencia, que sera 1 dia depois da ultima data
data_referencia = ultima_data + pd.DateOffset(days=1)
# Encontrar a ultima data de compra por pessoa
ultima_compra_por_pessoa = data_limpo.groupby('CustomerID')['InvoiceDate'].max().reset_index()
#Calculo de recencia(diferença entre a data de referencia e última compra)
ultima_compra_por_pessoa['Recencia'] = (data_referencia - ultima_compra_por_pessoa['InvoiceDate']).dt.days
print(data.info())