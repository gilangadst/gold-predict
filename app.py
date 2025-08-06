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

# 1. Input tanggal target
st.sidebar.markdown("#### 📅 Tanggal Target Prediksi")

def next_weekday(d):
    while d.weekday() >= 5:  # 5 = Sabtu, 6 = Minggu
        d += timedelta(days=1)
    return d

target_date = st.sidebar.date_input(
    "Pilih tanggal yang ingin diprediksi:",
    value=next_weekday(datetime.now().date() + timedelta(days=1)),
    min_value=datetime.now().date() + timedelta(days=1),
    max_value=datetime.now().date() + timedelta(days=7),
    help="Pilih tanggal di masa depan untuk prediksi harga emas (maksimal 7 hari)"
)

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
- Untuk 90 hari, dibutuhkan ~130 hari kalender
- Jika data kurang, gunakan opsi alternatif
- **Prediksi maksimal 7 hari ke depan**
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
        
        # Prediksi
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
        
        # Hitung berapa hari ke depan yang diprediksi
        days_ahead = (target_date - datetime.now().date()).days
        
        # Validasi maksimal 7 hari
        if days_ahead > 7:
            st.error("❌ **Error**: Prediksi maksimal hanya 7 hari ke depan!")
            st.stop()
        
        # Multi-step prediction untuk tanggal yang lebih jauh
        if days_ahead == 1:
            # Prediksi 1 hari ke depan
            prediction = model.predict(X_input)
            predicted_price = scaler.inverse_transform(prediction)[0][0]
        else:
            # Prediksi multi-step untuk tanggal yang lebih jauh
            current_input = X_input.copy()
            predicted_prices = []
            
            for day in range(days_ahead):
                # Prediksi 1 hari ke depan
                prediction = model.predict(current_input)
                predicted_price_step = scaler.inverse_transform(prediction)[0][0]
                predicted_prices.append(predicted_price_step)
                
                # Update input untuk prediksi berikutnya (rolling window)
                # Tambahkan prediksi ke input dan geser window
                predicted_scaled = scaler.transform([[predicted_price_step]])
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = predicted_scaled[0, 0]
            
            # Ambil prediksi untuk tanggal target
            predicted_price = predicted_prices[-1]
        
        # Display prediction
        st.markdown("### 🎯 Hasil Prediksi")
        
        # Tampilkan info multi-step prediction jika lebih dari 1 hari
        if days_ahead > 1:
            st.info(f"📈 **Multi-step Prediction**: Model memprediksi {days_ahead} hari ke depan menggunakan rolling window approach")
        
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            st.metric(
                label="Harga Terakhir",
                value=f"${close_prices[-1]:,.2f}",
                delta=f"${close_prices[-1] - close_prices[-2]:,.2f}" if len(close_prices) > 1 else None
            )
        
        with col_pred2:
            st.metric(
                label=f"Prediksi {target_date.strftime('%d %B %Y')}",
                value=f"${predicted_price:,.2f}",
                delta=f"${predicted_price - close_prices[-1]:,.2f}"
            )
        
        with col_pred3:
            change_percent = ((predicted_price - close_prices[-1]) / close_prices[-1]) * 100
            st.metric(
                label="Perubahan (%)",
                value=f"{change_percent:+.2f}%",
                delta_color="normal" if abs(change_percent) < 2 else ("inverse" if change_percent < 0 else "normal")
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
        
        # Plot prediksi
        if days_ahead == 1:
            # Prediksi 1 hari
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
        else:
            # Multi-step prediction - tampilkan semua prediksi intermediate
            prediction_dates = []
            prediction_values = [close_prices[-1]]  # Mulai dari harga terakhir
            
            current_date = datetime.now().date()
            for i in range(days_ahead):
                current_date += timedelta(days=1)
                prediction_dates.append(current_date.strftime('%Y-%m-%d'))
                if i < len(predicted_prices):
                    prediction_values.append(predicted_prices[i])
            
            fig.add_trace(
                go.Scatter(
                    x=prediction_dates,
                    y=prediction_values,
                    mode='lines+markers',
                    name='Prediksi Multi-step',
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
        
        # Tampilkan tabel prediksi intermediate untuk multi-step
        if days_ahead > 1 and 'predicted_prices' in locals():
            st.markdown("### 📊 Prediksi Intermediate")
            st.info("Berikut adalah prediksi harga emas untuk setiap hari dari sekarang hingga tanggal target:")
            
            # Buat tabel prediksi intermediate
            intermediate_dates = []
            intermediate_prices = []
            
            current_date = datetime.now().date()
            for i in range(days_ahead):
                current_date += timedelta(days=1)
                intermediate_dates.append(current_date.strftime('%Y-%m-%d'))
                if i < len(predicted_prices):
                    intermediate_prices.append(predicted_prices[i])
            
            df_intermediate = pd.DataFrame({
                'Tanggal': intermediate_dates,
                'Prediksi Harga (USD)': intermediate_prices,
                'Perubahan': [f"{((predicted_prices[i] - close_prices[-1]) / close_prices[-1] * 100):+.2f}%" if i < len(predicted_prices) else "N/A" for i in range(days_ahead)]
            })
            
            st.dataframe(df_intermediate, use_container_width=True)
            
            st.warning("⚠️ **Peringatan**: Prediksi multi-step untuk tanggal jauh mungkin kurang akurat karena error compounding. Semakin jauh tanggal prediksi, semakin besar kemungkinan deviasi dari nilai sebenarnya. Prediksi dioptimalkan untuk maksimal 7 hari ke depan.")
        
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
                        st.metric(
                            label="Harga Terakhir",
                            value=f"${close_prices[-1]:,.2f}",
                            delta=f"${close_prices[-1] - close_prices[-2]:,.2f}" if len(close_prices) > 1 else None
                        )
                    
                    with col_pred2:
                        st.metric(
                            label=f"Prediksi {target_date.strftime('%d %B %Y')}",
                            value=f"${predicted_price:,.2f}",
                            delta=f"${predicted_price - close_prices[-1]:,.2f}"
                        )
                    
                    with col_pred3:
                        change_percent = ((predicted_price - close_prices[-1]) / close_prices[-1]) * 100
                        st.metric(
                            label="Perubahan (%)",
                            value=f"{change_percent:+.2f}%",
                            delta_color="normal" if abs(change_percent) < 2 else ("inverse" if change_percent < 0 else "normal")
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
    - Metrik: Mean Squared Error (MSE)
    - **Batasan**: Maksimal 7 hari prediksi ke depan
    """)
    
    st.markdown("### 📊 Statistik")
    
    if 'close_prices' in locals():
        data_for_stats = close_prices
        
        st.metric("Rata-rata", f"${np.mean(data_for_stats):,.2f}")
        st.metric("Minimum", f"${np.min(data_for_stats):,.2f}")
        st.metric("Maximum", f"${np.max(data_for_stats):,.2f}")
        st.metric("Volatilitas", f"{np.std(data_for_stats):,.2f}")
    
    st.markdown("### ⚠️ Disclaimer")
    st.warning("""
    **Peringatan:**
    - Prediksi ini hanya untuk tujuan edukasi
    - Harga emas dipengaruhi banyak faktor
    - Tidak ada jaminan akurasi prediksi
    - Konsultasikan dengan ahli keuangan untuk keputusan investasi
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2024 Prediksi Harga Emas LSTM | Dibuat dengan Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
