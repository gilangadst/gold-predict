import yfinance as yf
import pandas as pd

# Simbol emas di Yahoo Finance adalah GC=F
symbol = 'GC=F'

# Rentang waktu: 1 Juni 2020 – 1 Juni 2025
start_date = '2005-01-01'
end_date = '2025-08-01'

# Ambil data
data_emas = yf.download(symbol, start=start_date, end=end_date, interval='1d')

# Tampilkan 5 data pertama
print(data_emas.head())

# Simpan ke file CSV
data_emas.to_csv('gold_dataset.csv')
