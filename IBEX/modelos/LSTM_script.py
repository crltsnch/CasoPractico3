import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("modelo_lstm_ibex.h5")

# Número de días a predecir (un mes completo)
prediction_days = 7

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

# Graficar los resultados de las predicciones futuras
st.write('### Predicciones LSTM para el Próximo Mes')
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(df['Close'], label='Datos Originales', color='blue')
ax4.axvline(x=last_date, color='gray', linestyle='--', label='Inicio de Predicciones')
ax4.plot(predictions_df, label='Predicciones LSTM (30 días)', color='red', linestyle='--')
ax4.set_xlabel('Fecha')
ax4.set_ylabel('Precio de Cierre')
ax4.set_title('Predicciones del Modelo LSTM para el Próximo Mes')
ax4.legend()
st.pyplot(fig4)

