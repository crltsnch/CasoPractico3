import yfinance as yf
from datetime import datetime, timedelta

# Define el símbolo del ticker y el periodo de tiempo
tiker = '^IBEX'
start_date = '2012-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Obtener los datos de las acciones
data = yf.download(tiker, start=start_date, end=end_date)

# Guardamos los datos
data.to_csv('./IBEX/data/ibex_data.csv')