import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    model = load_model('./model/model.h5')
    return model

model = load_lstm_model()

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
    "Pilih tanggal yang ingin diprediksi:",
    value=target_date_default,
    min_value=target_date_default,
    max_value=target_date_default,
    help="Pilih tanggal untuk prediksi harga emas (hanya 1 hari ke depan)"
)

# Additional check for weekends (in case the date_input somehow returns a weekend)
if target_date.weekday() >= 5:
    st.sidebar.warning("Tanggal otomatis digeser ke hari kerja terdekat.")
    target_date = next_weekday(target_date)

# 2. Pilihan window data historis
st.sidebar.markdown("#### 📊 Window Data Historis")
window_option = st.sidebar.selectbox(
    "Pilih periode data historis:",
    options=[
        ("7", "7 hari terakhir (Short-term)"),
        ("30", "30 hari terakhir (Mid-term)")
    ],
    format_func=lambda x: x[1],
    help="Semakin panjang periode, semakin banyak pola yang dianalisis"
)

# Info tentang data availability
st.sidebar.markdown("#### ℹ️ Info Data")
st.sidebar.info("""
**Catatan:**
- Yahoo Finance menyediakan data trading days
- Hari libur dan weekend tidak termasuk
- **Prediksi hanya 1 hari ke depan**
""")

window_days = int(window_option[0])

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
    
    # Ambil data dari Yahoo Finance
    with st.spinner("🔄 Mengambil data dari Yahoo Finance..."):
        # Hitung periode yang diperlukan dengan buffer lebih besar
        end_date = datetime.now().date() + timedelta(days=1)
        # Tambah buffer lebih besar untuk mengatasi hari libur
        buffer_days = max(window_days * 2, 180)  # Minimal 180 hari buffer
        start_date = end_date - timedelta(days=buffer_days)
        
        data = yf.download('GC=F', start=start_date, end=end_date, interval='1d')
        
    if len(data) >= window_days:
        close_prices = data['Close'].dropna().values[-window_days:].flatten()
        dates = data.index[-window_days:]
        
        # Display data info
        st.info(f"📊 Data {window_days} hari terakhir dari Yahoo Finance (GC=F)")
        
        # Create dataframe untuk display
        df_display = pd.DataFrame({
            'Tanggal': [d.strftime('%Y-%m-%d') for d in dates],
            'Harga Emas (USD)': close_prices.tolist()
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
            
            # Create arrow indicator for previous day change
            if prev_change > 0:
                delta_arrow = "⬆️"
                delta_color = "normal"
            elif prev_change < 0:
                delta_arrow = "⬇️"
                delta_color = "inverse"
            else:
                delta_arrow = "➡️"
                delta_color = "normal"
            
            st.metric(
                label="Harga Terakhir",
                value=f"${close_prices[-1]:,.2f}",
                delta=f"{delta_arrow} ${prev_change:+,.2f} ({prev_change_percent:+.2f}%)",
                delta_color=delta_color
            )
        
        with col_pred2:
            # Calculate prediction change
            pred_change = predicted_price - close_prices[-1]
            pred_change_percent = ((pred_change) / close_prices[-1]) * 100
            
            # Create arrow indicator for prediction
            if pred_change > 0:
                delta_arrow = "⬆️"
                delta_color = "normal"
            elif pred_change < 0:
                delta_arrow = "⬇️"
                delta_color = "inverse"
            else:
                delta_arrow = "➡️"
                delta_color = "normal"
            
            st.metric(
                label=f"Prediksi {target_date.strftime('%d %B %Y')}",
                value=f"${predicted_price:,.2f}",
                delta=f"{delta_arrow} ${pred_change:+,.2f} ({pred_change_percent:+.2f}%)",
                delta_color=delta_color
            )
        
        with col_pred3:
            change_percent = ((predicted_price - close_prices[-1]) / close_prices[-1]) * 100
            
            # Create arrow indicator for percentage change
            if change_percent > 0:
                delta_arrow = "⬆️"
                delta_color = "normal"
                status_text = "Naik"
            elif change_percent < 0:
                delta_arrow = "⬇️"
                delta_color = "inverse"
                status_text = "Turun"
            else:
                delta_arrow = "➡️"
                delta_color = "normal"
                status_text = "Stabil"
            
            st.metric(
                label="Perubahan (%)",
                value=f"{change_percent:+.2f}%",
                delta=f"{delta_arrow} {status_text}",
                delta_color=delta_color
            )
        
        # Plot dengan Plotly
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Harga Emas Historis', 'Prediksi vs Aktual'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot historis
        fig.add_trace(
            go.Scatter(
                x=[d.strftime('%Y-%m-%d') for d in dates],
                y=close_prices,
                mode='lines+markers',
                name='Harga Historis',
                line=dict(color='#FFD700', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
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
            ),
            row=1, col=1
        )
        
        # Bar chart perubahan
        fig.add_trace(
            go.Bar(
                x=['Harga Terakhir', 'Prediksi'],
                y=[close_prices[-1], predicted_price],
                name='Perbandingan',
                marker_color=['#FFD700', '#FF6B6B']
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text=f"Prediksi Harga Emas untuk {target_date.strftime('%d %B %Y')}",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Tanggal", row=1, col=1)
        fig.update_yaxes(title_text="Harga (USD)", row=1, col=1)
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_yaxes(title_text="Harga (USD)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error(f"❌ Data tidak cukup! Hanya tersedia {len(data)} hari, dibutuhkan minimal {window_days} hari.")
        
        # Tampilkan opsi alternatif
        st.markdown("### 🔄 Opsi Alternatif:")
        
        # Hitung berapa hari yang tersedia
        available_days = len(data)
        
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
                        
                        # Create arrow indicator for previous day change
                        if prev_change > 0:
                            delta_arrow = "⬆️"
                            delta_color = "normal"
                        elif prev_change < 0:
                            delta_arrow = "⬇️"
                            delta_color = "inverse"
                        else:
                            delta_arrow = "➡️"
                            delta_color = "normal"
                        
                        st.metric(
                            label="Harga Terakhir",
                            value=f"${close_prices[-1]:,.2f}",
                            delta=f"{delta_arrow} ${prev_change:+,.2f} ({prev_change_percent:+.2f}%)",
                            delta_color=delta_color
                        )
                    
                    with col_pred2:
                        # Calculate prediction change
                        pred_change = predicted_price - close_prices[-1]
                        pred_change_percent = ((pred_change) / close_prices[-1]) * 100
                        
                        # Create arrow indicator for prediction
                        if pred_change > 0:
                            delta_arrow = "⬆️"
                            delta_color = "normal"
                        elif pred_change < 0:
                            delta_arrow = "⬇️"
                            delta_color = "inverse"
                        else:
                            delta_arrow = "➡️"
                            delta_color = "normal"
                        
                        st.metric(
                            label=f"Prediksi {target_date.strftime('%d %B %Y')}",
                            value=f"${predicted_price:,.2f}",
                            delta=f"{delta_arrow} ${pred_change:+,.2f} ({pred_change_percent:+.2f}%)",
                            delta_color=delta_color
                        )
                    
                    with col_pred3:
                        change_percent = ((predicted_price - close_prices[-1]) / close_prices[-1]) * 100
                        
                        # Create arrow indicator for percentage change
                        if change_percent > 0:
                            delta_arrow = "⬆️"
                            delta_color = "normal"
                            status_text = "Naik"
                        elif change_percent < 0:
                            delta_arrow = "⬇️"
                            delta_color = "inverse"
                            status_text = "Turun"
                        else:
                            delta_arrow = "➡️"
                            delta_color = "normal"
                            status_text = "Stabil"
                        
                        st.metric(
                            label="Perubahan (%)",
                            value=f"{change_percent:+.2f}%",
                            delta=f"{delta_arrow} {status_text}",
                            delta_color=delta_color
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
    - Input: 30 hari data historis
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
    <p>© 2024 Prediksi Harga Emas LSTM | Dibuat dengan Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
