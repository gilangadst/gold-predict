import requests
import pandas as pd

# API Key
api_key = "MASUKKAN_API_KEY_ANDA"

# Endpoint
url = "https://api.twelvedata.com/time_series"

# Parameter
params = {
    "symbol": "XAU/USD",        # emas
    "interval": "1day",         # interval harian
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "apikey": api_key,
    "format": "JSON"            # bisa JSON atau CSV
}

# Request data
response = requests.get(url, params=params)
data = response.json()

# Convert ke DataFrame
df = pd.DataFrame(data['values'])
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

print(df.head())

# Simpan ke CSV
df.to_csv("emas_historis.csv", index=False)
