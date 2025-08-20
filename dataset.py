import requests
import pandas as pd

# API Key
api_key = "f18bce1f9c6e475f8852f67591d6ccc0"

# Endpoint
url = "https://api.twelvedata.com/time_series"

# Parameter
params = {
    "symbol": "XAU/USD",        # emas
    "interval": "1day",         # interval harian
    "start_date": "2006-07-31",
    "end_date": "2025-07-31",
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
df.to_csv("dataset.csv", index=False)
