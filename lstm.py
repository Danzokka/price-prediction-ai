import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Carregar os datasets
df_train = pd.read_csv('data/bitcoin_price_training.csv')
df_test = pd.read_csv('data/btc_test.csv')

# Converter a coluna 'Date' para o tipo datetime
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

# Selecionar a coluna 'Close' para previsão
train_data = df_train[['Close']].values
test_data = df_test[['Close']].values

# Normalizar os dados entre [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Preparar as sequências para LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# Redimensionar os dados para [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Construção do modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=70, batch_size=1024, validation_data=(X_test, y_test))

# Previsões com o conjunto de teste
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Comparar as direções de preço
df_test['Real_Direction'] = np.where(df_test['Close'].diff() > 0, 'alta', 'baixa')
predicted_directions = np.where(np.diff(predicted_prices.flatten()) > 0, 'alta', 'baixa')
df_test['Predicted_Direction'] = [None] * (window_size + 1) + list(predicted_directions)
df_test['Accuracy'] = np.where(df_test['Real_Direction'] == df_test['Predicted_Direction'], 'Correto', 'Incorreto')

# Remover os valores iniciais onde não há previsão
df_test_filtered = df_test.dropna(subset=['Predicted_Direction'])

# Calcular a acurácia
accuracy = (df_test_filtered['Real_Direction'] == df_test_filtered['Predicted_Direction']).mean() * 100
print("\n===============================")
print(f"Acurácia das Previsões: {accuracy:.2f}%")
print("\n===============================\n")

# Plotar a previsão do preço de fechamento com direções corretas/incorretas
plt.figure(figsize=(14, 7))
sns.scatterplot(data=df_test_filtered, x='Date', y='Close', hue='Accuracy', style='Accuracy', palette={'Correto': 'green', 'Incorreto': 'red'}, s=30)
plt.title('Precisão das Previsões de Direção do Preço do Bitcoin com LSTM')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend(title='Precisão', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig('img/lstm/lstm_precision.png')
