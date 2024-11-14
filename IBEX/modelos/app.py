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


# Función para redondear y eliminar ceros innecesarios
def format_column(val):
    if isinstance(val, (int, float)):
        # Redondear a 2 decimales y eliminar ceros innecesarios
        return f'{val:.2f}'.rstrip('0').rstrip('.')
    return val

# Estilo para cambiar el color según el valor
def color_change(val):
    try:
        # Convertir a float para asegurar que la comparación funcione
        float_val = float(val)
        color = 'red' if float_val < 0 else 'green'
        return f'color: {color}'
    except:
        return ''  # Para valores que no sean numéricos, no aplicar color


st.set_page_config(layout="wide")

# Encabezado de la página
st.markdown(
    """
    <div style="
        background-color: #424863;
        padding: 15px;
        text-align: center;
        border-radius: 8px;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 60px;
        ">
        <h1>Predicciones IBEX35 con LSTM</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Menú de navegación superior
page = st.selectbox("Selecciona una página", ["Predicciones Futuras", "Nuestro Análisis"], index=0)

# Agregar un margen inferior mediante markdown
st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)

# Página 1: Nuestro Análisis
if page == "Nuestro Análisis":
    st.markdown(
    """
    <h2 style='text-align: center; margin-bottom: 40px;'>
    ¿Cuánto de bueno es nuestro modelo de predicción?
    </h2>
    """, 
    unsafe_allow_html=True
)
    
    # Crear primera fila de gráficos con dos columnas
    col1, col2 = st.columns([1, 1])

    with col1:
        texto_explicativo = """
        <div style="
            color: #81a800;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            margin-bottom: 10px;
            margin-top: 10px;
        ">
           <h5>¿Cómo se ven nuestras predicciones respecto a los valores reales?</h5>
        </div>
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: justify;
            margin-top: 20px;
            padding: 20px;
            font-size: 14px;
            line-height: 1.6;
        ">
            <p>
            Esta gráfica compara los precios reales del IBEX35 (en azul) con las predicciones de nuestro modelo (en rojo, línea discontinua).
            También incluye un área sombreada en gris que representa un rango donde creemos que los precios deberían estar con un 70% de certeza.
            Si las líneas roja y azul están muy cerca, significa que nuestro modelo está haciendo predicciones precisas. En este caso, se puede ver
            que ambas líneas se superponen mucho, lo que indica que nuestro modelo es capaz de seguir de cerca las fluctuaciones reales del mercado,
            demostrando un entendimiento robusto de las tendencias y comportamientos del IBEX35.
            </p>
        </div>
        """
        st.markdown(texto_explicativo, unsafe_allow_html=True)
        

    with col2:
        # Gráfico de Predicciones con Intervalo de Confianza
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results['Date'], results['Real'], label='Valores Reales', color='blue')
        ax.plot(results['Date'], results['Predicted'], label='Predicciones LSTM', linestyle='--', color='red')
        ax.fill_between(results['Date'], results['Confidence Interval Lower'], results['Confidence Interval Upper'], color='gray', alpha=0.3, label='Intervalo de Confianza 70%')
        ax.set_xlabel('Fecha', fontsize=12)  # Ajuste del tamaño de la etiqueta del eje x
        ax.set_ylabel('Precio de Cierre', fontsize=12)  # Ajuste del tamaño de la etiqueta del eje y
        ax.set_title('Comparación de Valores Reales y Predicciones LSTM', fontsize=10)  # Título más grande
        ax.tick_params(axis='x', labelsize=10)  # Tamaño de fuente de los valores del eje x
        ax.tick_params(axis='y', labelsize=10)  # Tamaño de fuente de los valores del eje y
        ax.legend(fontsize=10)  # Tamaño de fuente de la leyenda
        fig.savefig("IBEX/resultados/predicciones_vs_reales.png", dpi=300)
        st.pyplot(fig)
        

    # Espacio en blanco entre las filas
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Crear segunda fila de gráficos con una columna y un espacio vacío
    col3, col4 = st.columns([1, 1])

    with col3:
        texto_explicativo = """
        <div style="
            color: #81a800;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            margin-top: 30px
            margin-bottom: 20px;
        ">
           <h5>¿Cómo son los errores que podría cometer nuestro modelo?</h5>
        </div>
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: justify;
        padding: 20px;
        margin-top: 20px;
        font-size: 14px;
        line-height: 1.6;
    ">
        <p>
        Los residuos son la diferencia entre los valores reales y las predicciones. Esta gráfica muestra esos residuos a lo largo del tiempo, con una línea negra que representa cero (es decir, donde el error sería nulo).
        La mayoría de los residuos están cerca de la línea de cero, lo que indica que los errores de nuestro modelo son
        pequeños. Además, no se observa un patrón claro en los residuos, lo que sugiere que el modelo no tiene un
        sesgo sistemático (no comete los mismos errores una y otra vez).
        </p>
    </div>
    """
        st.markdown(texto_explicativo, unsafe_allow_html=True)
    
    with col4:
        # Gráfico de Residuos
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(results['Date'], results['Residuals'], label='Residuos', color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Fecha', fontsize=12)  # Ajuste del tamaño de la etiqueta del eje x
        ax2.set_ylabel('Residuos', fontsize=12)  # Ajuste del tamaño de la etiqueta del eje y
        ax2.set_title('Residuos del Modelo LSTM', fontsize=10)  # Título más grande
        ax2.tick_params(axis='x', labelsize=10)  # Tamaño de fuente de los valores del eje x
        ax2.tick_params(axis='y', labelsize=10)  # Tamaño de fuente de los valores del eje y
        ax2.legend(fontsize=10)  # Tamaño de fuente de la leyenda
        fig2.savefig("IBEX/resultados/residuos.png", dpi=300)
        st.pyplot(fig2)



    # Espacio en blanco entre las filas
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    col5, col6 = st.columns([1, 1])

    with col5:
        # Histograma de Residuos
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.hist(results['Residuals'], bins=20, color='red', alpha=0.5)
        ax3.set_xlabel('Residuos', fontsize=10)  # Ajuste del tamaño de la etiqueta del eje x
        ax3.set_ylabel('Frecuencia', fontsize=10)  # Ajuste del tamaño de la etiqueta del eje y
        ax3.set_title('Histograma de Residuos del Modelo LSTM', fontsize=10)  # Título más grande
        ax3.tick_params(axis='x', labelsize=12)  # Tamaño de fuente de los valores del eje x
        ax3.tick_params(axis='y', labelsize=12)  # Tamaño de fuente de los valores del eje y
        ax3.grid(True)
        fig3.savefig("IBEX/resultados/histograma_residuos.png", dpi=300)
        st.pyplot(fig3)


    with col6:
        texto_explicativo = """
        <div style="
            color: #81a800;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            margin-bottom: 10px;
            margin-top: 10px
        ">
           <h5>¿Y cúanto de significativos son nuestros errores?</h5>
        </div>
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: justify;
            padding: 20px;
            margin-top: 20px;
            font-size: 14px;
            line-height: 1.6;
        ">
            <p>
            Este gráfico muestra la distribución de los errores en nuestras predicciones. La mayoría de los errores se concentran
            alrededor de cero, con muy pocos errores extremos. Esto 
            significa que nuestro modelo es consistentemente preciso, con pocos errores grandes. Esto nos da confianza en la
            estabilidad y fiabilidad del modelo.
            </p>
        </div>
    """
        st.markdown(texto_explicativo, unsafe_allow_html=True)
        



# Página 2: Predicciones Futuras
if page == "Predicciones Futuras":
    st.markdown(
    """
    <h2 style='text-align: center; margin-bottom: 40px;'>
    Predicciones para los próximos 5 días del IBEX35
    </h2>
    """, 
    unsafe_allow_html=True
)


    # Convertir a datetime
    future_predictions.index = pd.to_datetime(future_predictions.index)

    # Filtrar los últimos 30 días de los datos reales y las predicciones
    last_30_days_data = results['Real'].iloc[-30:]
    last_30_days_predictions = future_predictions['Predicted Close'].iloc[:30]  # Selecciona las primeras 30 predicciones

    # Graficar el zoom de las predicciones de los últimos 30 días
    fig5, ax5 = plt.subplots(figsize=(18, 8))

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
    fig5.savefig("IBEX/resultados/predicciones.png", dpi=300)
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig5)

    # Preparar y mostrar la tabla estilizada
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

    # Aplicar la función de formato a todas las columnas
    future_predictions = future_predictions.applymap(format_column)

    # Mostrar predicciones como tarjetas en un diseño de 5 columnas
    st.markdown("<h3 style='text-align: center;'>Detalles de las Predicciones</h3>", unsafe_allow_html=True)

    # Crear columnas para las tarjetas
    cols = st.columns(5)

    # Iterar a través de las filas y mostrar cada predicción en las columnas
    for i, (index, row) in enumerate(future_predictions.iterrows()):
        cambio_color = 'red' if float(row['Cambio']) < 0 else 'green'
        cambio_porcentual_color = 'red' if float(row['Cambio Porcentual (%)']) < 0 else 'green'

        # Distribuir las tarjetas en 5 columnas
        with cols[i % 5]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 10px;
                    background-color: #f9f9f9;
                    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                    width: 100%;
                ">
                    <h4 style="margin: 0; color: #424863; font-size: 18px;">{index}</h4>
                    <p style="margin: 5px 0; font-size: 18px;">
                        <strong>Cierre:</strong> {row['Predicción de Cierre']} 
                    </p>
                    <p style="margin: 5px 0; font-size: 18px; color: {cambio_color};">
                        <strong>
                            { '↑' if float(row['Cambio']) > 0 else '↓' }
                            {row['Cambio']}
                            ({row['Cambio Porcentual (%)']}%)
                        </strong>
                    </p>


                </div>
                """,
                unsafe_allow_html=True
            )





