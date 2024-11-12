import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Cargar los datos de predicciones
file_path = "IBEX/resultados/lstm_predictions.csv"
results = pd.read_csv(file_path)

# Mostrar datos de predicciones
st.write("### Predicciones del Modelo LSTM para el IBEX35")
st.write(results.head())

# Graficar resultados
st.write("### Gráfico de Predicciones")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results['Date'], results['Real'], label='Valores Reales', color='blue')
ax.plot(results['Date'], results['Predicted'], label='Predicciones LSTM', linestyle='--', color='red')
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio de Cierre')
ax.set_title('Comparación de Valores Reales y Predicciones LSTM')
ax.legend()
st.pyplot(fig)

# Graficar residuos
residuals = results['Real'] - results['Predicted']
st.write("### Gráfico de Residuos")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(residuals, label='Residuos', color='red')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax2.set_xlabel('Días')
ax2.set_ylabel('Residuos')
ax2.set_title('Residuos del Modelo LSTM')
ax2.legend()
st.pyplot(fig2)

# Mostrar histograma de residuos
st.write("### Histograma de Residuos")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.hist(residuals, bins=20, color='red', alpha=0.5)
ax3.set_xlabel('Residuos')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Histograma de Residuos del Modelo LSTM')
st.pyplot(fig3)
