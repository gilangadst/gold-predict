import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests

st.set_page_config(page_title="Prediksi Harga Emas LSTM", layout="wide")

# Header dengan styling
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #FFD700, #FFA500); border-radius: 10px; margin-bottom: 30px;">
    <h1 style="color: white; margin: 0;">🏆 Prediksi Harga Emas dengan LSTM</h1>
    <p style="color: white; margin: 5px 0 0 0;">Model Machine Learning untuk Prediksi Harga Emas</p>
</div>
""", unsafe_allow_html=True)

# Load model LSTM
@st.cache_resource
def load_lstm_model():
    model = load_model('./model.h5')
    return model

model = load_lstm_model()

# ====== KONFIGURASI API KEY TWELVE DATA ======
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY', 'f18bce1f9c6e475f8852f67591d6ccc0')  # Ganti dengan API key Anda

# ====== FUNGSI AMBIL DATA DARI TWELVE DATA ======
def get_gold_data_twelvedata(window_days=7):
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1day&outputsize={window_days}&apikey={TWELVE_DATA_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'values' in data:
            # Urutkan dari lama ke baru
            values = list(reversed(data['values']))
            close_prices = [float(v['close']) for v in values[-window_days:]]
            dates = [pd.to_datetime(v['datetime']).date() for v in values[-window_days:]]
            return close_prices, dates, 'Twelve Data'
        else:
            return None, None, None
    except Exception as e:
        return None, None, None

@st.cache_data(ttl=300)
def get_gold_data_twelvedata_cached(window_days=7):
    return get_gold_data_twelvedata(window_days)

@st.cache_data(ttl=300)
def get_gold_data_yahoo_cached(start_date, end_date, window_days=7):
    data = yf.download('GC=F', start=start_date, end=end_date, interval='1d')
    if len(data) >= window_days:
        close_prices = data['Close'].dropna().values[-window_days:].flatten()
        dates = data.index[-window_days:]
        return close_prices, dates
    else:
        return None, None

# Sidebar untuk input parameter
st.sidebar.markdown("### ⚙️ Parameter Prediksi")

# 1. Input tanggal target - hanya 1 hari ke depan
st.sidebar.markdown("#### 📅 Tanggal Target Prediksi")

def next_weekday(d):
    """Return the next weekday if the given date falls on a weekend"""
    result = d
    while result.weekday() >= 5:  # 5 = Sabtu, 6 = Minggu
        result += timedelta(days=1)
    return result

# Calculate the target date once to ensure consistency
tomorrow = datetime.now().date() + timedelta(days=1)
target_date_default = next_weekday(tomorrow)

# Ensure all date values are the same for the date_input
target_date = st.sidebar.date_input(
    "Tanggal yang akan diprediksi:",
    value=target_date_default,
    min_value=target_date_default,
    max_value=target_date_default,
    help="Tanggal untuk prediksi harga emas (hanya 1 hari ke depan)"
)

# Additional check for weekends (in case the date_input somehow returns a weekend)
if target_date.weekday() >= 5:
    st.sidebar.warning("Tanggal otomatis digeser ke hari kerja terdekat.")
    target_date = next_weekday(target_date)

# 2. Pilihan window data historis
# st.sidebar.markdown("#### 📊 Window Data Historis")
# window_option = st.sidebar.selectbox(
#     "Pilih periode data historis:",
#     options=[
#         ("7", "7 hari terakhir (Short-term)")
#     ],
#     format_func=lambda x: x[1]
# )
window_days = 7 # default: 30 hari terakhir

# Info tentang data availability
st.sidebar.markdown("#### ℹ️ Info Data")
st.sidebar.info("""
**Catatan:**
- Yahoo Finance menyediakan data trading days
- Hari libur dan weekend tidak termasuk
- **Prediksi hanya 1 hari ke depan**
""")

# 3. Info data source
st.sidebar.markdown("#### 📈 Sumber Data")
st.sidebar.info("""
**Data Source:**
- Yahoo Finance (GC=F)
- Data real-time harga emas
- Update otomatis setiap hari
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Dashboard Prediksi")
    
    # Ambil data dari Twelve Data, fallback ke Yahoo Finance
    with st.spinner("🔄 Mengambil data harga emas..."):
        # Hitung periode yang diperlukan dengan buffer lebih besar
        end_date = datetime.now().date() + timedelta(days=1)
        buffer_days = max(window_days * 2, 180)  # Minimal 180 hari buffer
        start_date = end_date - timedelta(days=buffer_days)

        # Coba ambil dari Twelve Data (cached)
        close_prices, dates, data_source = get_gold_data_twelvedata_cached(window_days)
        if close_prices is None or len(close_prices) < window_days:
            # Fallback ke Yahoo Finance (cached)
            close_prices, dates = get_gold_data_yahoo_cached(start_date, end_date, window_days)
            data_source = 'Yahoo Finance' if close_prices is not None else None

    if close_prices is not None and len(close_prices) >= window_days:
        # Display data info
        st.info(f"📊 Data {window_days} hari terakhir dari {data_source}")
        # Create dataframe untuk display
        df_display = pd.DataFrame({
            'Tanggal': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates],
            'Harga Emas (USD)': close_prices
        })
        st.dataframe(df_display, use_container_width=True)
        
        # Prediksi 1 hari ke depan
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_reshaped = np.array(close_prices).reshape(-1, 1)
        scaler.fit(close_prices_reshaped)
        close_scaled = scaler.transform(close_prices_reshaped)
        
        # Reshape untuk model (assuming model expects 30 days)
        if window_days != 30:
            # Pad atau truncate ke 30 days
            if window_days < 30:
                padded_data = np.pad(close_scaled.flatten(), (30 - window_days, 0), mode='edge')
            else:
                padded_data = close_scaled.flatten()[-30:]
            X_input = np.reshape(padded_data, (1, 30, 1))
        else:
            X_input = np.reshape(close_scaled, (1, 30, 1))
        
        # Prediksi 1 hari ke depan
        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        st.markdown("### 🎯 Hasil Prediksi")
        
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            # Calculate previous day change
            prev_change = close_prices[-1] - close_prices[-2] if len(close_prices) > 1 else 0
            prev_change_percent = ((prev_change) / close_prices[-2]) * 100 if len(close_prices) > 1 and close_prices[-2] != 0 else 0
            
            st.metric(
                label="Harga Terakhir",
                value=f"${close_prices[-1]:,.2f}",
                delta=round(prev_change, 2),
                help=f"{prev_change:+,.2f} ({prev_change_percent:+.2f}%)"
            )
        
        with col_pred2:
            # Calculate prediction change
            pred_change = predicted_price - close_prices[-1]
            pred_change_percent = ((pred_change) / close_prices[-1]) * 100
            
            st.metric(
                label=f"Prediksi {target_date.strftime('%d %B %Y')}",
                value=f"${predicted_price:,.2f}",
                delta=round(pred_change, 2),
                help=f"{pred_change:+,.2f} ({pred_change_percent:+.2f}%)"
            )
        
        with col_pred3:
            change_percent = ((predicted_price - close_prices[-1]) / close_prices[-1]) * 100
            
            st.metric(
                label="Perubahan (%)",
                value=f"{change_percent:+.2f}%",
                delta=round(change_percent, 2),
                help="Naik" if change_percent > 0 else ("Turun" if change_percent < 0 else "Stabil")
            )
        
        # Plot dengan Plotly
        fig = go.Figure()
        
        # Plot historis
        fig.add_trace(
            go.Scatter(
                x=[d.strftime('%Y-%m-%d') for d in dates],
                y=close_prices,
                mode='lines+markers',
                name='Harga Historis',
                line=dict(color='#FFD700', width=2),
                marker=dict(size=6)
            )
        )
        
        # Plot prediksi 1 hari
        fig.add_trace(
            go.Scatter(
                x=[dates[-1].strftime('%Y-%m-%d'), target_date.strftime('%Y-%m-%d')],
                y=[close_prices[-1], predicted_price],
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            )
        )
        
        fig.update_layout(
            height=500,
            title_text=f"Prediksi Harga Emas untuk {target_date.strftime('%d %B %Y')}",
            showlegend=True,
            hovermode='x unified',
            xaxis_title="Tanggal",
            yaxis_title="Harga (USD)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error(f"❌ Data tidak cukup! Hanya tersedia {len(data) if 'data' in locals() else 0} hari, dibutuhkan minimal {window_days} hari.")
        
        # Tampilkan opsi alternatif
        st.markdown("### 🔄 Opsi Alternatif:")
        
        # Hitung berapa hari yang tersedia
        available_days = len(data) if 'data' in locals() else 0
        
        if available_days >= 7:
            st.info(f"💡 **Saran**: Gunakan {available_days} hari yang tersedia atau pilih window yang lebih kecil")
            
            # Tampilkan data yang tersedia
            if available_days > 0:
                close_prices_available = data['Close'].dropna().values.flatten()
                dates_available = data.index
                
                st.write(f"📊 Data {available_days} hari yang tersedia:")
                df_available = pd.DataFrame({
                    'Tanggal': [d.strftime('%Y-%m-%d') for d in dates_available],
                    'Harga Emas (USD)': close_prices_available.tolist()
                })
                st.dataframe(df_available, use_container_width=True)
                
                # Opsi untuk menggunakan data yang tersedia
                if st.button(f"🔄 Gunakan {available_days} hari yang tersedia"):
                    # Gunakan data yang tersedia
                    close_prices = close_prices_available
                    dates = dates_available
                    
                    # Prediksi dengan data yang tersedia
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    close_prices_reshaped = np.array(close_prices).reshape(-1, 1)
                    scaler.fit(close_prices_reshaped)
                    close_scaled = scaler.transform(close_prices_reshaped)
                    
                    # Pad atau truncate ke 30 days untuk model
                    if len(close_scaled) < 30:
                        padded_data = np.pad(close_scaled.flatten(), (30 - len(close_scaled), 0), mode='edge')
                    else:
                        padded_data = close_scaled.flatten()[-30:]
                    
                    X_input = np.reshape(padded_data, (1, 30, 1))
                    prediction = model.predict(X_input)
                    predicted_price = scaler.inverse_transform(prediction)[0][0]
                    
                    # Display prediction
                    st.markdown("### 🎯 Hasil Prediksi (dengan data yang tersedia)")
                    col_pred1, col_pred2, col_pred3 = st.columns(3)
                    
                    with col_pred1:
                        # Calculate previous day change
                        prev_change = close_prices[-1] - close_prices[-2] if len(close_prices) > 1 else 0
                        prev_change_percent = ((prev_change) / close_prices[-2]) * 100 if len(close_prices) > 1 and close_prices[-2] != 0 else 0
                        st.metric(
                            label="Harga Terakhir",
                            value=f"${close_prices[-1]:,.2f}",
                            delta=round(prev_change, 2),
                            help=f"{prev_change:+,.2f} ({prev_change_percent:+.2f}%)"
                        )
                    
                    with col_pred2:
                        pred_change = predicted_price - close_prices[-1]
                        pred_change_percent = ((pred_change) / close_prices[-1]) * 100
                        st.metric(
                            label=f"Prediksi {target_date.strftime('%d %B %Y')}",
                            value=f"${predicted_price:,.2f}",
                            delta=round(pred_change, 2),
                            help=f"{pred_change:+,.2f} ({pred_change_percent:+.2f}%)"
                        )
                    
                    with col_pred3:
                        change_percent = ((predicted_price - close_prices[-1]) / close_prices[-1]) * 100
                        st.metric(
                            label="Perubahan (%)",
                            value=f"{change_percent:+.2f}%",
                            delta=round(change_percent, 2),
                            help="Naik" if change_percent > 0 else ("Turun" if change_percent < 0 else "Stabil")
                        )
                    
                    # Plot dengan data yang tersedia
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=[d.strftime('%Y-%m-%d') for d in dates],
                        y=close_prices,
                        mode='lines+markers',
                        name='Harga Historis',
                        line=dict(color='#FFD700', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[dates[-1].strftime('%Y-%m-%d'), target_date.strftime('%Y-%m-%d')],
                        y=[close_prices[-1], predicted_price],
                        mode='lines+markers',
                        name='Prediksi',
                        line=dict(color='red', width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        title=f"Prediksi Harga Emas untuk {target_date.strftime('%d %B %Y')} (dengan {available_days} hari data)",
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (USD)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.warning("⚠️ **Catatan**: Prediksi ini menggunakan data yang tersedia. Akurasi mungkin berbeda dari yang diharapkan.")
        
        else:
            st.error("❌ Data terlalu sedikit untuk melakukan prediksi yang akurat.")
            st.info("💡 **Saran**: Pilih window yang lebih kecil atau coba lagi nanti.")

with col2:
    st.markdown("### 📋 Informasi Model")
    
    st.info("""
    **Model LSTM yang digunakan:**
    - Arsitektur: Long Short-Term Memory
    - Input: 7 hari data historis
    - Output: Prediksi 1 hari ke depan
    - Metrik: MAE, MSE, MAPE, dan RMSE
    - **Batasan**: Hanya 1 hari prediksi ke depan
    """)
    
    st.markdown("### 📊 Statistik")
    
    if 'close_prices' in locals():
        data_for_stats = close_prices
        
        st.metric("Rata-rata", f"${np.mean(data_for_stats):,.2f}")
        st.metric("Minimum", f"${np.min(data_for_stats):,.2f}")
        st.metric("Maximum", f"${np.max(data_for_stats):,.2f}")
        st.metric("Volatilitas", f"{np.std(data_for_stats):,.2f}")
    

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2025 Prediksi Harga Emas LSTM | Dibuat dengan Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
