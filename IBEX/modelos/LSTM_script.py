import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Cargar los datos desde el archivo CSV
file_path = "../data/ibex_data_clean.csv"
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

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Hacer predicciones con el modelo LSTM
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

# Calcular intervalo de confianza del 70%
z_value = 1.04  # Valor crítico para un intervalo de confianza del 70%
confidence_interval = z_value * np.std(y_test_unscaled - predicted_prices_unscaled)

# Crear los intervalos superior e inferior
upper_bound = predicted_prices_unscaled + confidence_interval
lower_bound = predicted_prices_unscaled - confidence_interval

# Extraer las fechas para el conjunto de prueba
test_dates = df_filtered.index[-len(y_test_unscaled):]

# Graficar los resultados con los intervalos de confianza
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_unscaled, label='Valores Reales', color='blue')
plt.plot(test_dates, predicted_prices_unscaled, label=f'Predicciones LSTM - RMSE: {lstm_metrics[2]:.2f}', linestyle='--', color='red')
plt.fill_between(test_dates, lower_bound.flatten(), upper_bound.flatten(), color='gray', alpha=0.3, label='Intervalo de Confianza (70%)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Predicciones del Modelo LSTM con Intervalo de Confianza del 70%')
plt.legend()
plt.show()


# Graficar los residuos
lstm_residuals = y_test_unscaled - predicted_prices_unscaled
plt.figure(figsize=(12, 6))
plt.plot(lstm_residuals, label='LSTM Residuos', color='red')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Días')
plt.ylabel('Residuos')
plt.title('Residuos del Modelo LSTM')
plt.legend()
plt.show()

# Graficar histograma de los residuos
plt.figure(figsize=(12, 6))
plt.hist(lstm_residuals, bins=20, color='red', alpha=0.5, label='LSTM Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos del Modelo LSTM')
plt.legend()
plt.grid(True)
plt.show()

# Crear un DataFrame para los residuos
residuals_df = pd.DataFrame({
    'Día': np.arange(len(y_test_unscaled)),
    'LSTM Residuos': lstm_residuals.flatten()
})

# Mostrar la tabla de resultados de residuos
print(residuals_df)

# Calcular y mostrar el coeficiente R² en entrenamiento (opcional)
train_predictions = model.predict(X_train)
train_predictions_unscaled = scaler.inverse_transform(train_predictions)
r2_train = r2_score(scaler.inverse_transform(y_train), train_predictions_unscaled)
print(f'Coeficiente de determinación R² en entrenamiento: {r2_train:.2f}')
print(f'Coeficiente de determinación R² en prueba: {lstm_metrics[3]:.2f}')



#Guardar modelo
save_path = "modelo_lstm_ibex.h5"
model.save(save_path)
print(f"Modelo guardado en: {save_path}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
load_path = "modelo_lstm_ibex.h5"
model = load_model(load_path)

# Número de días a predecir (un mes completo)
prediction_days = 30

# Crear una lista para almacenar las predicciones
predictions = []

# Usar la última secuencia de datos para generar la primera predicción
last_sequence = X_test[-1].reshape(-1, 1)

# Generar predicciones día a día
for _ in range(prediction_days):
    # Hacer una predicción para el siguiente día
    next_pred = model.predict(np.expand_dims(last_sequence, axis=0))
    predictions.append(next_pred[0][0])  # Guardar la predicción
    
    # Actualizar la secuencia deslizante
    last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)

# Invertir la escala de las predicciones
predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Crear un rango de fechas para el próximo mes
last_date = df.index[-1]  # Última fecha en el DataFrame original
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')

# Crear un DataFrame para las predicciones futuras
predictions_df = pd.DataFrame(data=predictions_unscaled, index=future_dates, columns=['Predicted Close'])

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Datos Originales', color='blue')
plt.axvline(x=last_date, color='gray', linestyle='--', label='Inicio de Predicciones')
plt.plot(predictions_df, label='Predicciones LSTM (30 días)', color='red', linestyle='--')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Predicciones del Modelo LSTM para el Próximo Mes')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar el modelo LSTM
load_path = "modelo_lstm_ibex.h5"
model = load_model(load_path)

# Número de días a predecir
prediction_days = 7

# Lista para almacenar las predicciones
predictions = []

# Última secuencia para hacer las predicciones
last_sequence = X_test[-1].reshape(-1, 1)

# Generar predicciones
for _ in range(prediction_days):
    next_pred = model.predict(np.expand_dims(last_sequence, axis=0))
    predictions.append(next_pred[0][0])
    last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)

# Invertir el escalado de las predicciones
predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Crear fechas para las predicciones
last_date = df.index[-1]
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='D')

# DataFrame de predicciones
predictions_df = pd.DataFrame(data=predictions_unscaled, index=future_dates, columns=['Predicted Close'])

# Crear una gráfica combinada de los últimos datos y las predicciones
plt.figure(figsize=(14, 7))

# Mostrar los últimos 6 meses de datos
last_6_months = df['Close'].iloc[-180:]  # Últimos 180 días (aproximadamente 6 meses)
plt.plot(last_6_months, label='Datos Originales (Últimos 6 Meses)', color='blue')

# Añadir las predicciones
plt.plot(predictions_df, label='Predicciones LSTM (30 días)', color='red', linestyle='--')

# Líneas verticales para marcar el inicio de las predicciones
plt.axvline(x=last_date, color='gray', linestyle='--', label='Inicio de Predicciones')

# Añadir etiquetas y título
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Predicciones del Modelo LSTM con Datos Recientes')
plt.legend()
plt.show()

# Gráfica de zoom en las predicciones
plt.figure(figsize=(14, 7))
plt.plot(df['Close'].iloc[-30:], label='Datos Originales (Últimos 30 Días)', color='blue')
plt.plot(predictions_df, label='Predicciones LSTM (30 días)', color='red', linestyle='--')
plt.axvline(x=last_date, color='gray', linestyle='--', label='Inicio de Predicciones')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Zoom en las Predicciones del Modelo LSTM')
plt.legend()
plt.show()
