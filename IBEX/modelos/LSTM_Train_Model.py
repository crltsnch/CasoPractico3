import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Cargar los datos
file_path = "IBEX/data/ibex_data_clean.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)
df_filtered = df.dropna(subset=['Close'])

# Escalar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_filtered[['Close']])

# Crear secuencias de datos
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 365
X, y = create_sequences(scaled_data, seq_length)

# Dividir los datos
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construir y entrenar el modelo
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo
model.save("IBEX/resultados/modelo_lstm_ibex.keras")

# Hacer predicciones y escalado inverso
train_predictions = scaler.inverse_transform(model.predict(X_train))
test_predictions = scaler.inverse_transform(model.predict(X_test))
y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Guardar predicciones y datos en un archivo CSV
results = pd.DataFrame({
    'Date': df_filtered.index[-len(y_test_unscaled):],
    'Real': y_test_unscaled.flatten(),
    'Predicted': test_predictions.flatten()
})
results.to_csv("IBEX/resultados/lstm_predictions.csv", index=False)


# ==============================
# Predicciones para los próximos 7 días
# ==============================

# Número de días a predecir
prediction_days = 7

# Última secuencia de datos para hacer las predicciones
last_sequence = X_test[-1].reshape(-1, 1)

# Lista para almacenar las predicciones futuras
future_predictions = []

# Generar predicciones para los próximos 7 días
for _ in range(prediction_days):
    next_pred = model.predict(np.expand_dims(last_sequence, axis=0))
    future_predictions.append(next_pred[0][0])
    last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)

# Invertir el escalado de las predicciones
future_predictions_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Crear fechas para las predicciones
last_date = df_filtered.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')

# Crear DataFrame de predicciones futuras
future_predictions_df = pd.DataFrame(data=future_predictions_unscaled, index=future_dates, columns=['Predicted Close'])

# Guardar las predicciones futuras en un archivo CSV
future_predictions_df.to_csv("IBEX/resultados/lstm_future_predictions.csv")

print("Predicciones para los próximos 7 días guardadas en 'IBEX/resultados/lstm_future_predictions.csv'")