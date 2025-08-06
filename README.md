# 🏆 Prediksi Harga Emas dengan LSTM

Aplikasi web untuk memprediksi harga emas menggunakan model Long Short-Term Memory (LSTM) dengan interface yang user-friendly.

## ✨ Fitur Utama

### 📅 Input Tanggal Target
- User dapat memilih tanggal spesifik di masa depan untuk prediksi
- Interface kalender yang mudah digunakan
- Validasi tanggal (hanya tanggal masa depan yang diizinkan)

### 📊 Window Data Historis (Sliding Window)
- **7 hari terakhir** (Short-term) - untuk analisis tren jangka pendek
- **30 hari terakhir** (Mid-term) - untuk analisis tren menengah
- **90 hari terakhir** (Long-term) - untuk analisis tren jangka panjang

### 📈 Sumber Data
- **Yahoo Finance (Real-time)** - Data harga emas real-time dari GC=F
- Data update otomatis setiap hari trading

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 3. Buka Browser
Aplikasi akan terbuka di `http://localhost:8501`

## 🎯 Cara Penggunaan

### Langkah 1: Pilih Parameter di Sidebar
1. **Tanggal Target**: Pilih tanggal yang ingin diprediksi
2. **Window Data**: Pilih periode data historis (7, 30, atau 90 hari)
3. **Data Source**: Yahoo Finance (GC=F) - real-time

### Langkah 2: Lihat Hasil Prediksi
- Dashboard akan menampilkan data historis
- Metrik prediksi dengan perubahan harga
- Grafik interaktif dengan Plotly
- Statistik deskriptif

### Langkah 3: Analisis Hasil
- Bandingkan harga terakhir dengan prediksi
- Lihat persentase perubahan
- Analisis tren dari grafik

## 📊 Komponen Aplikasi

### Sidebar (Parameter Input)
- **Tanggal Target**: Date picker untuk memilih tanggal prediksi
- **Window Data**: Dropdown untuk memilih periode (7/30/90 hari)
- **Data Source**: Info Yahoo Finance (GC=F)

### Main Dashboard
- **Data Historis**: Tabel dan grafik data input
- **Hasil Prediksi**: Metrik dengan delta perubahan
- **Visualisasi**: Grafik interaktif dengan Plotly
- **Statistik**: Rata-rata, min, max, volatilitas

### Informasi Tambahan
- **Model Info**: Detail arsitektur LSTM
- **Disclaimer**: Peringatan penggunaan
- **Footer**: Informasi aplikasi

## 🔧 Teknis

### Model LSTM
- **Input**: 30 hari data historis (dipad/truncate jika berbeda)
- **Output**: Prediksi 1 hari ke depan
- **Arsitektur**: Long Short-Term Memory
- **Metrik**: Mean Squared Error (MSE)

### Data Processing
- **Normalisasi**: MinMaxScaler (0-1)
- **Padding**: Edge padding untuk data < 30 hari
- **Truncating**: Ambil 30 hari terakhir untuk data > 30 hari

### Visualisasi
- **Plotly**: Grafik interaktif dengan hover info
- **Subplots**: Kombinasi line chart dan bar chart
- **Responsive**: Menyesuaikan dengan ukuran layar

## 📁 Struktur File

```
SkripsiAyu2/
├── app.py                          # Aplikasi utama Streamlit
├── requirements.txt                # Dependencies Python
├── README.md                       # Dokumentasi ini
├── lstm_gold_price_model.h5       # Model LSTM (harus ada)
└── harga_emas_jun2020_jun2025.csv # Data training (opsional)
```

## ⚠️ Disclaimer

**Peringatan Penting:**
- Prediksi ini hanya untuk tujuan edukasi dan penelitian
- Harga emas dipengaruhi oleh banyak faktor kompleks
- Tidak ada jaminan akurasi prediksi
- Konsultasikan dengan ahli keuangan untuk keputusan investasi
- Model ini tidak memperhitungkan faktor fundamental ekonomi

## 🛠️ Troubleshooting

### Error: Model tidak ditemukan
- Pastikan file `lstm_gold_price_model.h5` ada di direktori yang sama
- Model harus kompatibel dengan TensorFlow versi yang digunakan

### Error: Data Yahoo Finance tidak cukup
- Coba pilih window yang lebih kecil (7 hari)
- Periksa koneksi internet
- Yahoo Finance mungkin sedang maintenance

### Error: Input manual tidak valid
- Pastikan format: angka dipisahkan koma
- Contoh: `950000,951000,952000`
- Tidak boleh ada spasi atau karakter khusus

## 📈 Contoh Penggunaan

### Skenario 1: Prediksi Jangka Pendek
1. Pilih tanggal: besok
2. Window: 7 hari terakhir
3. Sumber: Yahoo Finance
4. Analisis: Tren jangka pendek untuk trading

### Skenario 2: Prediksi Jangka Menengah
1. Pilih tanggal: 1 minggu ke depan
2. Window: 30 hari terakhir
3. Sumber: Yahoo Finance
4. Analisis: Tren menengah untuk investasi

### Skenario 3: Analisis Jangka Panjang
1. Pilih tanggal: 1 minggu ke depan
2. Window: 90 hari terakhir
3. Sumber: Yahoo Finance
4. Analisis: Tren jangka panjang untuk investasi

## 🔄 Update dan Maintenance

### Menambah Window Baru
1. Edit opsi di `window_option` selectbox
2. Tambahkan logika padding/truncating sesuai kebutuhan
3. Update dokumentasi

### Mengganti Model
1. Ganti file `lstm_gold_price_model.h5`
2. Sesuaikan input shape jika diperlukan
3. Test dengan data sample

### Menambah Sumber Data
1. Tambahkan fungsi fetch data baru
2. Implementasi preprocessing sesuai format
3. Update dokumentasi

## 📞 Support

Untuk pertanyaan atau masalah:
- Periksa troubleshooting di atas
- Pastikan semua dependencies terinstall
- Cek versi Python dan packages

---

**© 2024 Prediksi Harga Emas LSTM | Dibuat dengan Streamlit & TensorFlow** 