import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import holidays

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


# Calcular las métricas de evaluación
# Error cuadrático medio (MSE)
mse_train = mean_squared_error(y_train_unscaled, train_predictions)
mse_test = mean_squared_error(y_test_unscaled, test_predictions)

# Coeficiente de determinación R²
r2_train = r2_score(y_train_unscaled, train_predictions)
r2_test = r2_score(y_test_unscaled, test_predictions)

# Residuo medio absoluto (MAE)
mae_train = mean_absolute_error(y_train_unscaled, train_predictions)
mae_test = mean_absolute_error(y_test_unscaled, test_predictions)

# Crear DataFrame con las métricas
metrics = {
    'Metric': ['MSE Train', 'MSE Test', 'R2 Train', 'R2 Test', 'MAE Train', 'MAE Test'],
    'Value': [mse_train, mse_test, r2_train, r2_test, mae_train, mae_test]
}

metrics_df = pd.DataFrame(metrics)

# Guardar las métricas en un archivo CSV
metrics_df.to_csv("IBEX/resultados/model_metrics.csv", index=False)

# Imprimir el DataFrame con las métricas
print(metrics_df)




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


