# 🔧 Troubleshooting Guide

## ❌ Masalah: "Data tidak cukup! Hanya tersedia X hari, dibutuhkan minimal Y hari"

### Penyebab Masalah:

1. **Hari Libur Pasar**
   - Yahoo Finance hanya menyediakan data pada hari trading
   - Weekend (Sabtu-Minggu) tidak termasuk
   - Hari libur nasional tidak termasuk
   - Untuk 90 hari trading, dibutuhkan ~130 hari kalender

2. **Batasan Yahoo Finance**
   - API Yahoo Finance mungkin membatasi data historis
   - Koneksi internet yang lambat
   - Rate limiting dari Yahoo Finance

3. **Periode Pengambilan Data**
   - Buffer waktu yang tidak cukup
   - Perhitungan tanggal yang tidak tepat

### ✅ Solusi:

#### 1. Gunakan Window yang Lebih Kecil
- Pilih "7 hari terakhir" untuk data minimal
- Pilih "30 hari terakhir" untuk data menengah
- Hindari "90 hari terakhir" jika sering error

#### 2. Gunakan Opsi Alternatif
- Klik tombol "Gunakan X hari yang tersedia"
- Aplikasi akan menggunakan data yang ada
- Prediksi tetap bisa dilakukan dengan akurasi yang mungkin berbeda

#### 3. Coba Lagi Nanti
- Yahoo Finance mungkin sedang maintenance
- Coba refresh halaman setelah beberapa menit
- Periksa koneksi internet

#### 4. Cek Koneksi Internet
- Pastikan koneksi internet stabil
- Coba refresh halaman
- Tunggu beberapa menit dan coba lagi

### 📊 Perhitungan Data:

| Window | Trading Days | Kalender Days | Buffer yang Dibutuhkan |
|--------|-------------|---------------|----------------------|
| 7 hari | 7 | ~10 | 20 hari |
| 30 hari | 30 | ~42 | 60 hari |
| 90 hari | 90 | ~130 | 180 hari |

### 🔄 Cara Kerja Aplikasi:

1. **Pengambilan Data**: Aplikasi mengambil data dengan buffer 2x window
2. **Validasi**: Mengecek apakah data cukup untuk window yang dipilih
3. **Fallback**: Jika data kurang, menawarkan opsi alternatif
4. **Prediksi**: Menggunakan data yang tersedia dengan padding/truncating

### 💡 Tips Penggunaan:

1. **Untuk Trading Jangka Pendek**: Gunakan 7 hari
2. **Untuk Analisis Menengah**: Gunakan 30 hari
3. **Untuk Analisis Jangka Panjang**: Gunakan 90 hari
4. **Untuk Testing**: Gunakan data yang tersedia dengan opsi alternatif

### 🚨 Jika Masalah Berlanjut:

1. Cek versi library:
   ```bash
   pip show yfinance
   pip show streamlit
   ```

2. Update library:
   ```bash
   pip install --upgrade yfinance streamlit
   ```

3. Coba sumber data alternatif:
   - Coba di waktu yang berbeda
   - Cek status Yahoo Finance
   - Gunakan opsi data yang tersedia

### 📞 Support:

Jika masalah masih berlanjut:
- Periksa log error di terminal
- Pastikan semua dependencies terinstall
- Coba restart aplikasi
- Hubungi support jika diperlukan

---

**Catatan**: Masalah data adalah hal yang umum terjadi dengan API finansial. Aplikasi ini dirancang untuk menangani kasus tersebut dengan graceful fallback. 