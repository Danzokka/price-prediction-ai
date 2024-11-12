import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Carregar os datasets
df_train = pd.read_csv('data/bitcoin_price_training.csv')
df_test = pd.read_csv('data/btc_test.csv')

# Converter a coluna 'Date' para o tipo datetime
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

# Remover vírgulas e converter as colunas 'Volume' e 'Market Cap' para numérico
df_train['Volume'] = pd.to_numeric(df_train['Volume'].astype(str).str.replace(',', ''), errors='coerce')
df_train['Market Cap'] = pd.to_numeric(df_train['Market Cap'].astype(str).str.replace(',', ''), errors='coerce')

# No conjunto de teste, 'Market Cap' não está presente
df_test['Volume'] = pd.to_numeric(df_test['Volume'].astype(str).str.replace(',', ''), errors='coerce')

# Selecionar as features para o clustering (removendo 'Market Cap')
features = ['Open', 'High', 'Low', 'Close', 'Volume']

X_train = df_train[features].fillna(0)
X_test = df_test[features].fillna(0)

# Padronizar as features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gerar o linkage matrix para o dendrograma
linked = linkage(X_train_scaled, method='ward')

# Plotar o dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.savefig('img/hierarchical/hierarchical_dendrogram.png')

# Definir o número de clusters com base no dendrograma
n_clusters = 32 # Ajuste conforme necessário

# Treinar o modelo de clustering hierárquico
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df_train['Cluster'] = hc.fit_predict(X_train_scaled)

# Aplicar o modelo aos dados de teste
df_test['Cluster'] = hc.fit_predict(X_test_scaled)

# Calcular o Coeficiente de Silhueta e o Índice de Davies-Bouldin para o conjunto de teste
silhouette_avg = silhouette_score(X_test_scaled, df_test['Cluster'])
davies_bouldin_avg = davies_bouldin_score(X_test_scaled, df_test['Cluster'])

print("\n===============================")
print("\nResultados do Clustering Hierárquico:")
print(f'\nCoeficiente de Silhueta (Teste): {silhouette_avg}')
print(f'\nÍndice de Davies-Bouldin (Teste): {davies_bouldin_avg}')

# Calcular as médias das features para cada cluster no conjunto de treinamento
cluster_means_train = df_train.groupby('Cluster')[features].mean()

# Calcular as médias das features para cada cluster no conjunto de teste
cluster_means_test = df_test.groupby('Cluster')[features].mean()

# Determinar a direção real do preço no conjunto de teste
df_test['Direction'] = np.where(df_test['Close'].diff() > 0, 'alta', 
                                np.where(df_test['Close'].diff() < 0, 'baixa', 'estável'))

# Atribuir uma tendência a cada cluster com base nos dados de treinamento
cluster_trend = df_train.groupby('Cluster')['Close'].apply(lambda x: 'alta' if x.diff().mean() > 0 
                                                           else ('baixa' if x.diff().mean() < 0 else 'estável'))

# Aplicar as previsões de tendência ao conjunto de teste com base nos clusters
df_test['Predicted_Direction'] = df_test['Cluster'].map(cluster_trend)

# Calcular a acurácia comparando a direção prevista com a direção real
accuracy = (df_test['Direction'] == df_test['Predicted_Direction']).mean() * 100
print(f"\nAcurácia: {accuracy:.2f}%")
print("\n===============================\n")

# Plotar o gráfico final do preço de fechamento com os clusters identificados
plt.figure(figsize=(14, 7))
sns.scatterplot(data=df_test, x='Date', y='Close', hue='Cluster', palette='tab20', legend='full', s=10)
plt.title('Variação do Preço de Fechamento do Bitcoin com Clusters Identificados')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('img/hierarchical/hierarchical_clusters.png')

# Comparação entre a direção real e a direção prevista
df_test['Prediction_Accuracy'] = np.where(df_test['Direction'] == df_test['Predicted_Direction'], 'Correto', 'Incorreto')

# Plotar gráfico da precisão das previsões
plt.figure(figsize=(14, 7))
sns.scatterplot(data=df_test, x='Date', y='Close', hue='Prediction_Accuracy', style='Prediction_Accuracy', palette={'Correto': 'green', 'Incorreto': 'red'}, s=30)
plt.title('Precisão das Previsões de Direção do Preço do Bitcoin')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend(title='Precisão da Previsão', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig('img/hierarchical/hierarchical_precision.png')
