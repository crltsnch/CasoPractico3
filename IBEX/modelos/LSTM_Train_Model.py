import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import holidays

# Cargar los datos desde el archivo CSV
file_path = "IBEX/data/ibex_data_clean.csv"
df = pd.read_csv(file_path)

# Mostrar las primeras filas del DataFrame
print("Datos originales:")
print(df.head())

# Convertir la columna 'Date' a tipo datetime y establecerla como índice
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Asegúrate de que el formato coincida
df.set_index('Date', inplace=True)

# Filtrar todas las filas donde 'Close' no sea nulo o NaN
df_filtered = df.dropna(subset=['Close'])

# Seleccionar la columna objetivo
data = df_filtered[['Close']]

# Escalar los datos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Crear secuencias de datos para entrenamiento
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Definir la longitud de la secuencias
seq_length = 365
X, y = create_sequences(scaled_data, seq_length)

# Dividir los datos en entrenamiento y prueba (80% - 20%)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Guardar el modelo
model.save("IBEX/resultados/modelo_lstm_ibex.keras")


## Hacer predicciones con el modelo LSTM
predicted_prices = model.predict(X_test)

# Función para calcular métricas de evaluación
def calcular_metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

# Calcular las métricas para el conjunto de prueba
lstm_metrics = calcular_metricas(y_test, predicted_prices)

# Imprimir las métricas
print("Métricas del Modelo LSTM:", lstm_metrics)
print(f"MSE: {lstm_metrics[0]}, MAE: {lstm_metrics[1]}, RMSE: {lstm_metrics[2]}, R²: {lstm_metrics[3]}")

# Escalado inverso
predicted_prices_unscaled = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular los residuos
residuals = y_test_unscaled.flatten() - predicted_prices_unscaled.flatten()

# Desviación estándar de los residuos
std_residuals = np.std(residuals)

# Intervalo de confianza del 70% (±1 desviación estándar)
z_value = 1.04  # Valor crítico para un intervalo de confianza del 70%
confidence_interval_upper = predicted_prices_unscaled.flatten() + z_value * std_residuals
confidence_interval_lower = predicted_prices_unscaled.flatten() - z_value * std_residuals

# Guardar los resultados en un DataFrame
results = pd.DataFrame({
    'Date': df_filtered.index[-len(y_test_unscaled):],  # Fecha correspondiente a las predicciones
    'Real': y_test_unscaled.flatten(),
    'Predicted': predicted_prices_unscaled.flatten(),
    'Residuals': residuals,
    'Confidence Interval Upper': confidence_interval_upper,
    'Confidence Interval Lower': confidence_interval_lower
})

# Guardar el DataFrame en un archivo CSV
results.to_csv("IBEX/resultados/lstm_predictions.csv", index=False)


# ==============================
# Predicciones para los próximos 7 días
# ==============================


# Obtener los días festivos de España (opcional)
es_holidays = holidays.Spain(years=range(2012, 2025))


# Número de días a predecir
prediction_days = 5

# Última secuencia de datos para hacer las predicciones
last_sequence = X_test[-1].reshape(-1, 1)

# Lista para almacenar las predicciones futuras y fechas
future_predictions = []
future_dates = []

# Obtener la última fecha de tu DataFrame
last_date = df_filtered.index[-1]

# Generar predicciones para los próximos 7 días hábiles
while len(future_predictions) < prediction_days:
    # Avanzar un día
    last_date += pd.Timedelta(days=1)

    # Saltar si es fin de semana
    if last_date.weekday() in [5, 6]:  # 5 = Sábado, 6 = Domingo
        continue

    # Saltar si es un día festivo
    if last_date in es_holidays:
        continue

    # Hacer la predicción
    next_pred = model.predict(np.expand_dims(last_sequence, axis=0))
    future_predictions.append(next_pred[0][0])
    future_dates.append(last_date)

    # Actualizar la secuencia con la nueva predicción
    last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)

# Invertir el escalado de las predicciones
future_predictions_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Crear DataFrame de predicciones futuras
future_predictions_df = pd.DataFrame(data=future_predictions_unscaled, index=future_dates, columns=['Predicted Close'])

# Inicializar las columnas 'Price Change' y 'Percentage Change'
future_predictions_df['Price Change'] = 0.0
future_predictions_df['Percentage Change'] = 0.0

# Último valor real del precio de cierre
first_real_value = df_filtered['Close'].iloc[-1]
print(f"Último valor real: {first_real_value}")

# Calcular 'Price Change' y 'Percentage Change' para la primera predicción
future_predictions_df.at[future_dates[0], 'Price Change'] = future_predictions_df.at[future_dates[0], 'Predicted Close'] - first_real_value
future_predictions_df.at[future_dates[0], 'Percentage Change'] = (future_predictions_df.at[future_dates[0], 'Price Change'] / first_real_value) * 100

# Calcular 'Price Change' y 'Percentage Change' para los días siguientes
for i in range(1, len(future_predictions_df)):
    current_close = future_predictions_df['Predicted Close'].iloc[i]
    previous_close = future_predictions_df['Predicted Close'].iloc[i - 1]
    
    # Cambio de precio respecto al día anterior
    future_predictions_df.at[future_dates[i], 'Price Change'] = current_close - previous_close
    
    # Porcentaje de cambio respecto al día anterior
    future_predictions_df.at[future_dates[i], 'Percentage Change'] = (future_predictions_df.at[future_dates[i], 'Price Change'] / previous_close) * 100

# Guardar las predicciones futuras en un archivo CSV
future_predictions_df.to_csv("IBEX/resultados/lstm_future_predictions.csv")

# Imprimir el DataFrame con las nuevas columnas
print(future_predictions_df)

print("Predicciones para los próximos 5 días hábiles guardadas en 'IBEX/resultados/lstm_future_predictions.csv'")


