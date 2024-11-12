import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar los datos
data = pd.read_csv('IBEX/data/ibex_data_clean.csv', index_col='Date', parse_dates=True)

# Previsualizar los datos
print(data.head())

# Selección de características (por ejemplo, el cierre ajustado)
dataset = data[['Close']].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Definir el tamaño del conjunto de entrenamiento
training_data_len = int(np.ceil(len(dataset) * 0.8))

# Crear el conjunto de datos de entrenamiento
train_data = scaled_data[0:training_data_len, :]

# Crear el conjunto de datos de validación
val_data = scaled_data[training_data_len - 60:, :]

# Dividir los datos en conjuntos x_train y y_train
x_train = []
y_train = []
x_val = []
y_val = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

for i in range(60, len(val_data)):
    x_val.append(val_data[i-60:i, 0])
    y_val.append(val_data[i, 0])

# Convertir x_train, y_train, x_val, y_val a arrays de numpy
x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)

# Redimensionar los datos
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

# Construir el modelo LSTM con Dropout
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo y validar en el conjunto de validación
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=40)

# Visualizar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predecir valores
train_predictions = model.predict(x_train)
val_predictions = model.predict(x_val)

# Desnormalizar las predicciones
train_predictions = scaler.inverse_transform(train_predictions)
val_predictions = scaler.inverse_transform(val_predictions)

# Evaluar el modelo
mse = mean_squared_error(dataset[60:training_data_len], train_predictions)
mae = mean_absolute_error(dataset[60:training_data_len], train_predictions)
rmse = np.sqrt(mse)
print(f'Training MSE: {mse}')
print(f'Training MAE: {mae}')
print(f'Training RMSE: {rmse}')

mse_val = mean_squared_error(dataset[training_data_len:], val_predictions)
mae_val = mean_absolute_error(dataset[training_data_len:], val_predictions)
rmse_val = np.sqrt(mse_val)
print(f'Validation MSE: {mse_val}')
print(f'Validation MAE: {mae_val}')
print(f'Validation RMSE: {rmse_val}')

# Visualizar resultados
plt.figure(figsize=(16, 8))
plt.plot(data.index[60:training_data_len], dataset[60:training_data_len], color='blue', label='Actual IBEX 35 Training')
plt.plot(data.index[60:training_data_len], train_predictions, color='red', label='Predicted IBEX 35 Training')
plt.plot(data.index[training_data_len:], dataset[training_data_len:], color='green', label='Actual IBEX 35 Validation')
plt.plot(data.index[training_data_len:], val_predictions, color='orange', label='Predicted IBEX 35 Validation')
plt.title('IBEX 35 Prediction')
plt.xlabel('Date')
plt.ylabel('IBEX 35 Value')
plt.legend()
plt.show()


# Esta parte está regulinchi


# Predecir los próximos 20 días
forecast_length = 20
predictions = []
lower_bounds = []
upper_bounds = []

# Obtener los últimos 60 días del conjunto de datos para iniciar la previsión
last_sequence = scaled_data[-60:, :]
current_sequence = last_sequence

for _ in range(forecast_length):
    current_sequence = np.reshape(current_sequence, (1, 60, 1))
    predicted_value = model.predict(current_sequence)
    predictions.append(predicted_value[0, 0])

    # Calcular intervalo de confianza
    uncertainty = 0.15 * predicted_value[0, 0]
    lower_bounds.append(predicted_value[0, 0] - uncertainty)
    upper_bounds.append(predicted_value[0, 0] + uncertainty)

    # Actualizar la secuencia actual
    current_sequence = np.append(current_sequence[:, 1:, :], [[predicted_value]], axis=1)

# Desnormalizar
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
lower_bounds = scaler.inverse_transform(np.array(lower_bounds).reshape(-1, 1))
upper_bounds = scaler.inverse_transform(np.array(upper_bounds).reshape(-1, 1))

# Crear DataFrame para el pronóstico
fecha_primera_prediccion = data.index[-1] + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=fecha_primera_prediccion, periods=forecast_length, freq='B')
predicted_df = pd.DataFrame(data=predictions, index=prediction_dates, columns=['Predicted Close'])
predicted_df['Lower Bound'] = lower_bounds
predicted_df['Upper Bound'] = upper_bounds

# Visualizar
plt.figure(figsize=(16, 8))
plt.plot(data.index[-250:], data['Close'][-250:], color='blue', label='Actual IBEX 35')
plt.plot(predicted_df.index, predicted_df['Predicted Close'], color='red', label='Predicted IBEX 35')
plt.fill_between(predicted_df.index, predicted_df['Lower Bound'], predicted_df['Upper Bound'], color='pink', alpha=0.3)
plt.axvline(fecha_primera_prediccion, color='green', linestyle='--', label='Start of Predictions')
plt.title('IBEX 35 Prediction for Next 20 Days')
plt.xlabel('Date')
plt.ylabel('IBEX 35 Value')
plt.legend()
plt.xticks(rotation=45)
plt.show()
