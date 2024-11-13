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



st.write("### Predicciones para los próximos 5 días del IBEX35")

# Convertir a datetime
future_predictions.index = pd.to_datetime(future_predictions.index)

# Filtrar los últimos 30 días de los datos reales y las predicciones
last_30_days_data = results['Real'].iloc[-30:]
last_30_days_predictions = future_predictions['Predicted Close'].iloc[:30]  # Selecciona las primeras 30 predicciones

# Graficar el zoom de las predicciones de los últimos 30 días
fig5, ax5 = plt.subplots(figsize=(14, 7))

# Datos reales (últimos 30 días)
ax5.plot(results['Date'].iloc[-30:], last_30_days_data, label='Datos de los últimos 30 Días)', color='blue')

# Predicciones LSTM (últimos 30 días)
ax5.plot(future_predictions.index[:30], last_30_days_predictions, label='Predicciones LSTM (Próximos 5 Días)', color='red', linestyle='--')

# Conectar el último dato real con el primer dato predicho con una línea discontinua roja
last_real_date = results['Date'].iloc[-1]
first_predicted_date = future_predictions.index[0]
first_predicted_value = future_predictions['Predicted Close'].iloc[0]

# Graficar la línea discontinua que conecta el último valor real con el primero predicho
ax5.plot([last_real_date, first_predicted_date], [last_30_days_data.iloc[-1], first_predicted_value], color='red', linestyle='--')

# Etiquetas y título
ax5.set_xlabel('Fecha')
ax5.set_ylabel('Precio de Cierre')
ax5.set_title('Predicciones para los Próximos 5 Días')
ax5.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig5)




# Suponiendo que 'future_predictions' ya tiene los datos que mencionaste
future_predictions['Fecha'] = future_predictions.index.strftime('%d %b %Y').str.lower()

# Restablecer el índice y eliminar la columna del índice
future_predictions = future_predictions.reset_index(drop=True)

# Renombrar las columnas para mayor claridad
future_predictions = future_predictions.rename(columns={
    'Predicted Close': 'Predicción de Cierre',
    'Price Change': 'Cambio',
    'Percentage Change': 'Cambio Porcentual (%)'
})

# Establecer 'Fecha' como índice
future_predictions.set_index('Fecha', inplace=True)

# Función para redondear y eliminar ceros innecesarios
def format_column(val):
    if isinstance(val, (int, float)):
        # Redondear a 2 decimales y eliminar ceros innecesarios
        return f'{val:.2f}'.rstrip('0').rstrip('.')
    return val

# Aplicar la función de formato a todas las columnas
future_predictions = future_predictions.applymap(format_column)

# Estilo para cambiar el color según el valor
def color_change(val):
    try:
        # Convertir a float para asegurar que la comparación funcione
        float_val = float(val)
        color = 'red' if float_val < 0 else 'green'
        return f'color: {color}'
    except:
        return ''  # Para valores que no sean numéricos, no aplicar color

# Aplicar el estilo a las columnas 'Cambio' y 'Cambio Porcentual (%)'
styled_df = future_predictions.style.applymap(color_change, subset=['Cambio', 'Cambio Porcentual (%)'])

# Mostrar la tabla estilizada
st.write(styled_df)


