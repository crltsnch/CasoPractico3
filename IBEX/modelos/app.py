import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Cargar los datos de predicciones
file_path = "IBEX/resultados/lstm_predictions.csv"
results = pd.read_csv(file_path)

# Cargar las predicciones futuras
future_predictions_path = "IBEX/resultados/lstm_future_predictions.csv"
future_predictions = pd.read_csv(future_predictions_path, index_col=0, parse_dates=True)

results['Date'] = pd.to_datetime(results['Date'])
results.index = pd.to_datetime(results.index)

residuals = results['Real'] - results['Predicted']
std_residuals = residuals.std()

# Crear el intervalo de confianza del 70% (±1 desviación estándar)
confidence_interval_upper = results['Predicted'] + std_residuals
confidence_interval_lower = results['Predicted'] - std_residuals

st.write("### Gráfico de Predicciones con Intervalo de Confianza del 70%")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results['Date'], results['Real'], label='Valores Reales', color='blue')
ax.plot(results['Date'], results['Predicted'], label='Predicciones LSTM', linestyle='--', color='red')
ax.fill_between(results['Date'], confidence_interval_lower, confidence_interval_upper, color='gray', alpha=0.3, label='Intervalo de Confianza 70%')
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio de Cierre')
ax.set_title('Comparación de Valores Reales y Predicciones LSTM con Intervalo de Confianza')
ax.legend()
st.pyplot(fig)

# Graficar residuos
st.write("### Gráfico de Residuos")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(results['Date'], residuals, label='Residuos', color='red')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Residuos')
ax2.set_title('Residuos del Modelo LSTM')
ax2.legend()

# Mostrar el gráfico de residuos
st.pyplot(fig2)

# Mostrar histograma de residuos
st.write("### Histograma de Residuos")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.hist(residuals, bins=20, color='red', alpha=0.5)
ax3.set_xlabel('Residuos')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Histograma de Residuos del Modelo LSTM')
st.pyplot(fig3)

st.write("### Predicciones para los próximos 7 días del IBEX35")

# Convertir a datetime
future_predictions.index = pd.to_datetime(future_predictions.index)

# Verificar longitud de los datos reales
n_rows = len(results)

# Graficar las predicciones futuras para los últimos 3 meses
fig4, ax4 = plt.subplots(figsize=(14, 7))

if n_rows < 90:
    last_3_months = results['Real'].iloc[-n_rows:]
    dates = results['Date'].iloc[-n_rows:]
else:
    last_3_months = results['Real'].iloc[-90:]
    dates = results['Date'].iloc[-90:]

# Graficar los datos reales de los últimos 3 meses
ax4.plot(dates, last_3_months, label='Datos Originales (Últimos 3 Meses)', color='blue')

# Añadir las predicciones futuras
ax4.plot(future_predictions.index, future_predictions['Predicted Close'], label='Predicciones LSTM (7 días)', color='red', linestyle='--')

# Conectar el último dato real con el primer dato predicho con una línea discontinua roja
last_real_date = dates.iloc[-1]
first_predicted_date = future_predictions.index[0]
first_predicted_value = future_predictions['Predicted Close'].iloc[0]

# Graficar la línea discontinua que conecta el último valor real con el primero predicho
ax4.plot([last_real_date, first_predicted_date], [last_3_months.iloc[-1], first_predicted_value], color='red', linestyle='--')

# Etiquetas y título
ax4.set_xlabel('Fecha')
ax4.set_ylabel('Precio de Cierre')
ax4.set_title('Predicciones Futuras del Modelo LSTM para los Próximos 7 Días')
ax4.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig4)

# Filtrar los últimos 30 días de los datos reales y las predicciones
last_30_days_data = results['Real'].iloc[-30:]
last_30_days_predictions = future_predictions['Predicted Close'].iloc[:30]  # Selecciona las primeras 30 predicciones

# Graficar el zoom de las predicciones de los últimos 30 días
fig5, ax5 = plt.subplots(figsize=(14, 7))

# Datos reales (últimos 30 días)
ax5.plot(results['Date'].iloc[-30:], last_30_days_data, label='Datos Originales (Últimos 30 Días)', color='blue')

# Predicciones LSTM (últimos 30 días)
ax5.plot(future_predictions.index[:30], last_30_days_predictions, label='Predicciones LSTM (Próximos 7 Días)', color='red', linestyle='--')

# Conectar el último dato real con el primer dato predicho con una línea discontinua roja
last_real_date = results['Date'].iloc[-1]
first_predicted_date = future_predictions.index[0]
first_predicted_value = future_predictions['Predicted Close'].iloc[0]

# Graficar la línea discontinua que conecta el último valor real con el primero predicho
ax5.plot([last_real_date, first_predicted_date], [last_30_days_data.iloc[-1], first_predicted_value], color='red', linestyle='--')

# Etiquetas y título
ax5.set_xlabel('Fecha')
ax5.set_ylabel('Precio de Cierre')
ax5.set_title('Zoom en las Predicciones para los Próximos 7 Días')
ax5.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig5)



# Convertir la columna de fecha para que solo contenga la fecha (sin la hora)
future_predictions['Fecha'] = future_predictions.index.strftime('%Y-%m-%d')  # Esto extrae solo la fecha en formato 'YYYY-MM-DD'

# Restablecer el índice y eliminar la columna del índice
future_predictions = future_predictions.reset_index(drop=True)

# Renombrar la columna para mayor claridad (si es necesario)
future_predictions = future_predictions.rename(columns={'Predicted Close': 'Predicción de Cierre'})

# Reordenar las columnas para que la 'Fecha' esté primero
future_predictions = future_predictions[['Fecha', 'Predicción de Cierre']]

# Mostrar la tabla sin la columna de índice
st.dataframe(future_predictions)



