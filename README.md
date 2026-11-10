# 🎓 Testbed Klasifikasi Efikasi Diri Akademik Siswa SMP & SMA

Pipeline data mining *end-to-end* berbasis web untuk klasifikasi dan evaluasi tingkat efikasi diri akademik siswa SMP dan SMA di Indonesia berdasarkan domain kesiapan belajar menggunakan algoritma Machine Learning.

---

## 📌 Deskripsi Proyek
Penelitian dan aplikasi ini bertujuan untuk memetakan dan mengklasifikasikan tingkat efikasi diri akademik siswa (*Tinggi*, *Sedang*, *Rendah*) dengan memanfaatkan indikator kesiapan belajar:
- **Domain Sosial-Emosional**: Rasa Memiliki (A1), Regulasi Emosi (A2), Perilaku Sosial (A3)
- **Domain Perilaku Belajar**: Ketekunan (B1), Perencanaan Belajar (B2), Keterlibatan di Kelas (B3)
- **Domain Kognitif-Motivasional**: Efikasi Diri (C1 - Target), Nilai Belajar (C2), Growth Mindset (C3)
- **Data Demografi & Konteks**: Jenjang sekolah (SMP/SMA), Kelas, Gender, Lokasi, Transportasi, Kepuasan, dan Keyakinan Prestasi.

---

## 🚀 Fitur Utama Aplikasi
1. **📋 Ringkasan Dataset & Preprocessing**:
   - Pembalikan skor butir negatif (*reverse scoring*).
   - Perhitungan agregat subskala dan pembentukan target tercile (C1).
2. **📊 Eksplorasi Data Interaktif (EDA)**:
   - Distribusi target efikasi diri akademik.
   - Korelasi subskala dengan target efikasi diri.
   - Perbandingan proporsi jenjang sekolah (SMP vs SMA).
   - Heatmap korelasi Pearson antar-subskala.
3. **🧩 Analisis Klastering Siswa (Unsupervised Learning)**:
   - Evaluasi klaster optimal dengan *Elbow Method* dan *Silhouette Score*.
   - Proyeksi 2D menggunakan Principal Component Analysis (PCA).
   - *Profiling* karakteristik siswa tiap klaster.
4. **🤖 Pelatihan & Perbandingan Model (Supervised Learning)**:
   - Evaluasi 5 algoritma klasifikasi: Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF), dan K-Nearest Neighbors (KNN).
   - Evaluasi menggunakan *Stratified 5-Fold Cross-Validation*.
   - Metrik komparasi: Test Macro-F1, Akurasi, Balanced Accuracy, dan Cohen's Kappa.
5. **📈 Evaluasi Mendalam & Simulator Prediksi**:
   - Confusion Matrix & Classification Report per kelas.
   - *Permutation Feature Importance*.
   - **Simulator Interaktif**: Prediksi tingkat efikasi diri siswa baru secara *real-time* lengkap dengan distribusi probabilitas kelas.

---

## 🛠️ Instalasi & Menjalankan Secara Lokal

### 1. Clone Repository
```bash
git clone https://github.com/zamzami-ux/data_science.git
cd data_science
```

### 2. Buat Lingkungan Virtual (Opsional namun Disarankan)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser Anda di `http://localhost:8501`.

---

## 🌐 Panduan Deployment (Streamlit Community Cloud)

1. Buka [Streamlit Community Cloud](https://share.streamlit.io/) dan masuk menggunakan akun GitHub Anda.
2. Klik tombol **New app**.
3. Isi parameter deployment:
   - **Repository**: `zamzami-ux/data_science`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Klik **Deploy!** — Web app akan langsung aktif secara publik dengan URL gratis.

---

## 📦 Struktur Berkas
```text
├── .streamlit/
│   └── config.toml          # Konfigurasi server Streamlit
├── 11.12_Raw_data.xlsx      # Dataset mentah respon siswa
├── app.py                   # Kode utama aplikasi Streamlit
├── requirements.txt         # Daftar pustaka Python yang dibutuhkan
├── .gitignore               # Berkas yang diabaikan oleh Git
└── README.md                # Dokumentasi proyek
```

---

## 📚 Pustaka yang Digunakan
- **[Streamlit](https://streamlit.io/)** — Kerangka kerja web UI interaktif
- **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** — Manipulasi dan analisis data
- **[Scikit-Learn](https://scikit-learn.org/)** — Pipeline preprocessing, klastering, dan model klasifikasi
- **[Matplotlib](https://matplotlib.org/)** — Visualisasi data dan grafik statistik
- **[OpenPyXL](https://openpyxl.readthedocs.io/)** — Parser pembacaan berkas Excel (.xlsx)
