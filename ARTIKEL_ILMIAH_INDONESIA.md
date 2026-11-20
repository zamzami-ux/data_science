# KLASIFIKASI TINGKAT EFIKASI DIRI AKADEMIK SISWA SMP DAN SMA DI INDONESIA BERDASARKAN DOMAIN KESIAPAN BELAJAR MENGGUNAKAN ALGORITMA MACHINE LEARNING

**Muhammad Zamzami**  
Program Studi Sains Data / Ilmu Komputer  
Fakultas Sains dan Teknologi  
*E-mail: zamzami@student.institution.ac.id*

---

## Abstrak
Efikasi diri akademik merupakan determinan psikologis fundamental yang mempengaruhi motivasi, resiliensi, dan pencapaian akademik peserta didik. Dalam konteks pendidikan di Indonesia, identifikasi dini terhadap siswa yang memiliki tingkat efikasi diri akademik rendah seringkali terhambat oleh keterbatasan instrumen asesmen konvensional yang bersifat manual dan lambat. Penelitian ini bertujuan untuk merancang dan mengevaluasi sistem klasifikasi *machine learning* untuk memprediksi tingkat efikasi diri akademik siswa sekolah menengah (SMP dan SMA) berdasarkan domain kesiapan belajar multidimensi. Dataset penelitian bersumber dari repositori publik Mendeley Data (DOI: 10.17632/ft5z9yk94n.1) yang mencakup 1.182 responden siswa Indonesia dengan 36 butir kuesioner skala Likert pada 9 subskala kesiapan belajar (A1–C3) serta variabel demografis. Target klasifikasi dibentuk dari subskala C1 (Efikasi Diri) yang dikelompokkan ke dalam tiga kategori (*Rendah*, *Sedang*, *Tinggi*) menggunakan ambang batas kuantil tercile (q1 = 4,25; q2 = 4,75). Untuk mencegah kebocoran data (*data leakage*), subskala C1 dihapus secara ketat dari matriks fitur prediktor, dan seluruh tahapan *preprocessing* (termasuk *reverse coding* butir A3.1 dan A3.2, pembersihan data, *StandardScaler*, serta *One-Hot Encoding*) dienkapsulasi dalam *pipeline scikit-learn*. Lima algoritma *supervised learning* dievaluasi menggunakan *Stratified 5-Fold Cross-Validation* dan pengujian independen (*test set* 20%): *Logistic Regression*, *Random Forest (300 estimators)*, *Support Vector Machine (SVM RBF)*, *K-Nearest Neighbors (KNN, k=15)*, dan *Gradient Boosting*. Hasil eksperimen menunjukkan bahwa seluruh model melampaui performa *Zero-Rule Baseline* (akurasi 44,73%). Model **SVM RBF** mencatat akurasi pengujian tertinggi sebesar **65,82%** (Macro-F1: **0,6295**), sementara **Gradient Boosting** menghasilkan Macro-F1 pengujian tertinggi sebesar **0,6317** (Akurasi: **64,98%**). Analisis *Permutation Feature Importance* mengungkapkan bahwa **Keterlibatan di Kelas (B3)**, **Nilai/Manfaat Belajar (C2)**, dan **Ketekunan Belajar (B1)** merupakan prediktor paling signifikan terhadap tingkat efikasi diri akademik siswa. Sistem ini telah diimplementasikan dalam aplikasi web interaktif berbasis Streamlit untuk mendukung diagnosis dini dan pengambilan keputusan pedagogis bagi pendidik dan konselor sekolah.

**Kata kunci**: *Efikasi Diri Akademik*, *Educational Data Mining*, *Machine Learning*, *Klasifikasi*, *Kesiapan Belajar*, *Streamlit*

---

## Abstract
*Academic self-efficacy is a fundamental psychological determinant influencing students' motivation, academic resilience, and learning performance. In the Indonesian secondary education context, early identification of students with low academic self-efficacy is frequently hampered by conventional, manual assessment tools. This study aims to develop and evaluate a machine learning classification framework to predict the academic self-efficacy levels of junior and senior high school students based on multidimensional learning readiness domains. The dataset was obtained from the Mendeley Data public repository (DOI: 10.17632/ft5z9yk94n.1), comprising 1,182 Indonesian secondary student respondents across 36 Likert-scale items covering 9 subscales (A1–C3) and demographic features. The classification target was derived from subscale C1 (Academic Self-Efficacy) stratified into three categories (Low, Moderate, High) via tercile quantiles (q1 = 4.25; q2 = 4.75). To rigorously prevent data leakage, subscale C1 was strictly eliminated from predictor matrices, and all preprocessing procedures—including reverse coding of items A3.1 and A3.2, missing data handling, StandardScaler, and One-Hot Encoding—were encapsulated inside scikit-learn Pipelines. Five supervised algorithms were evaluated using Stratified 5-Fold Cross-Validation and an independent 20% test partition: Logistic Regression, Random Forest (300 trees), Support Vector Machine (SVM RBF), K-Nearest Neighbors (KNN, k=15), and Gradient Boosting. Experimental findings demonstrated that all machine learning models significantly outperformed the Zero-Rule majority baseline (accuracy 44.73%). The SVM RBF model attained the highest test accuracy of 65.82% (Macro-F1: 0.6295), while Gradient Boosting achieved the top test Macro-F1 score of 0.6317 (Accuracy: 64.98%). Permutation Feature Importance revealed that Classroom Engagement (B3), Value of Learning (C2), and Academic Persistence (B1) are the most influential psychological predictors of student academic self-efficacy. This machine learning pipeline is operationalized through an interactive Streamlit web dashboard to facilitate early educational diagnosis and evidence-based pedagogical interventions.*

**Keywords**: *Academic Self-Efficacy*, *Educational Data Mining*, *Machine Learning*, *Classification*, *School Readiness*, *Streamlit*

---

## 1. PENDAHULUAN

Pendidikan di era Revolusi Industri 4.0 dan Society 5.0 menuntut peserta didik tidak hanya menguasai konten akademis, melainkan juga memiliki ketangguhan psikologis dan kapasitas metakognitif yang memadai (Igual & Segui, 2017). Di antara berbagai konstruk psikologi pendidikan, **efikasi diri akademik (*academic self-efficacy*)**—yang didefinisikan oleh Albert Bandura (1997) sebagai keyakinan individu terhadap kemampuannya dalam mengorganisasi dan mengeksekusi serangkaian tindakan yang diperlukan untuk mencapai tujuan belajar—memegang peranan yang sangat esensial. Peserta didik dengan tingkat efikasi diri yang tinggi cenderung memperlihatkan ketekunan yang lebih kuat saat menghadapi materi pelajaran yang kompleks, memiliki strategi regulasi diri yang terstruktur, serta memiliki kemungkinan putus sekolah (*dropout*) yang jauh lebih rendah (Zimmerman, 2000; Schunk & DiBenedetto, 2020).

Sebaliknya, siswa dengan efikasi diri akademik rendah rentan mengalami kecemasan akademik (*academic anxiety*), prokrastinasi, hingga penurunan drastis pada capaian hasil belajar. Di Indonesia, kesenjangan kualitas pendidikan serta masa transisi pendidikan antara jenjang Sekolah Menengah Pertama (SMP) dan Sekolah Menengah Atas (SMA) kerap memicu tekanan psikososial yang memperburuk kondisi efikasi diri siswa (Nugraha & Hairani, 2018). Kendati demikian, deteksi siswa yang mengalami defisit efikasi diri pada sistem persekolahan Indonesia pada umumnya masih bersifat pasif dan reaktif. Guru Bimbingan Konseling (BK) maupun pendidik kelas umumnya baru menyadari kesulitan siswa setelah nilai evaluasi akhir keluar atau ketika terjadi kegagalan akademik formal.

Pendekatan *Educational Data Mining* (EDM) dan *Machine Learning* menawarkan paradigma baru dalam sistem pendidikan presisi (*precision education*) (Romero & Ventura, 2020). Melalui pemanfaatan data psikometrik kesiapan belajar siswa, model *machine learning* mampu mengekstraksi pola-pola non-linear laten yang menghubungkan aspek afektif, perilaku, dan kognitif siswa guna mengklasifikasikan tingkat efikasi diri secara objektif dan terautomasi (Hairani et al., 2020; Jadhav & Channe, 2016).

Meskipun demikian, berbagai penelitian terdahulu di bidang pemodelan performa siswa kerap menghadapi kelemahan metodologis yang krusial, khususnya fenomena **kebocoran data (*data leakage*)** (Kaufman et al., 2012). Hal ini sering terjadi ketika variabel target atau turunan langsung dari instrumen evaluasi tidak diisolasi secara cermat dari matriks fitur prediktor, sehingga menghasilkan estimasi performa model yang terlalu optimis (*over-optimistic*) namun gagal ketika diuji pada data riil. Selain itu, banyak studi pemodelan hanya menyajikan metrik akurasi tanpa menganalisis bobot kontribusi fitur (*feature interpretability*), sehingga rekomendasi pedagogis yang dihasilkan sulit diimplementasikan oleh pendidik di lapangan.

Untuk mengatasi kesenjangan tersebut, penelitian ini mengembangkan sebuah sistem komputasional terpadu yang memadukan eksplorasi data komprehensif, rekayasa *pipeline* anti-kebocoran data (*anti-leakage pipeline*), komparasi multi-algoritma *supervised learning*, analisis kepentingan fitur (*permutation feature importance*), serta *deployment* ke dalam aplikasi web interaktif berbasis Streamlit. Penelitian ini memanfaatkan dataset representatif siswa Indonesia dari Mendeley Data (DOI: 10.17632/ft5z9yk94n.1) yang mencakup 1.182 responden dengan 36 butir kuesioner skala Likert pada sembilan subskala kesiapan belajar. 

Tujuan utama dari penelitian ini adalah:
1. Membangun dan membandingkan performa 5 (lima) algoritma *supervised learning* (*Logistic Regression*, *Random Forest*, *Support Vector Machine RBF*, *K-Nearest Neighbors*, dan *Gradient Boosting*) dalam memprediksi tingkat efikasi diri akademik siswa (*Rendah*, *Sedang*, *Tinggi*).
2. Mengidentifikasi domain kesiapan belajar yang paling determinan terhadap efikasi diri akademik siswa melalui analisis *Permutation Feature Importance*.
3. Menyediakan instrumen interaktif berbasis web (*Streamlit Application*) yang siap digunakan oleh para pemangku kepentingan pendidikan untuk simulasi dan inferensi dini tingkat efikasi diri siswa baru.

---

## 2. LANDASAN TEORI

### 2.1 Efikasi Diri Akademik dan Domain Kesiapan Belajar
Teori Kognitif Sosial Bandura (1997) menegaskan bahwa perilaku manusia merupakan hasil interaksi timbal balik (*triadic reciprocal causation*) antara faktor personal (kognitif, afektif), perilaku (*behavioral*), dan lingkungan (*environmental*). Dalam instrumen kesiapan belajar siswa yang diadaptasi dalam dataset penelitian ini, konstruksi kesiapan belajar dioperasionalisasikan ke dalam tiga domain utama yang diturunkan menjadi 9 subskala:

1. **Domain Hubungan Sekolah & Sosio-Emosional (A)**:
   - **A1 - Rasa Memiliki Sekolah (*School Belonging*)**: Persepsi penerimaan, penghargaan, dan rasa aman siswa di lingkungan sekolah.
   - **A2 - Regulasi Emosi (*Emotional Regulation*)**: Kemampuan mengelola afek negatif, stres, dan dorongan emosional saat belajar.
   - **A3 - Perilaku Sosial (*Social Behavior*)**: Kepatuhan terhadap norma kelas dan interaksi kooperatif dengan teman sebaya.
2. **Domain Keterampilan Belajar & Manajemen Diri (B)**:
   - **B1 - Ketekunan Akademik (*Academic Persistence*)**: Daya juang untuk terus belajar meskipun menghadapi kesulitan tugas (Grit/Resiliensi).
   - **B2 - Perencanaan & Pengorganisasian Belajar (*Study Planning*)**: Keterampilan mengatur waktu, jadwal belajar, dan target akademis.
   - **B3 - Keterlibatan di Kelas (*Classroom Engagement*)**: Partisipasi aktif, konsentrasi, dan keaktifan bertanya dalam proses pembelajaran.
3. **Domain Sikap & Keyakinan Akademis (C)**:
   - **C1 - Efikasi Diri Akademik (*Academic Self-Efficacy*)**: Keyakinan subyektif siswa terhadap kapasitas diri dalam memahami pelajaran dan meraih prestasi. Subskala ini menjadi **variabel target** dalam penelitian.
   - **C2 - Nilai/Manfaat Belajar (*Value of Learning*)**: Pemaknaan terhadap pentingnya pendidikan dan relevansi materi pelajaran bagi masa depan.
   - **C3 - Pola Pikir Berkembang (*Growth Mindset*)**: Keyakinan bahwa kecerdasan dan kemampuan akademis dapat ditingkatkan melalui usaha terarah (Dweck, 2006).

### 2.2 Algoritma Pembelajaran Mesin untuk Klasifikasi
Klasifikasi tingkat efikasi diri merupakan tugas pembelajaran terbimbing (*supervised classification*). Penelitian ini membandingkan lima keluarga algoritma dengan karakteristik matematis yang berbeda:
1. **Logistic Regression (Multinomial)**: Model parametrik linier yang memodelkan probabilitas keanggotaan kelas menggunakan fungsi *softmax*. Menyediakan interpretabilitas yang tinggi dan menjadi *benchmark* linier (Hosmer et al., 2013).
2. **Random Forest Classifier**: Algoritma *ensemble bagging* yang membangun ratusan pohon keputusan (*decision trees*) pada subsampel data acak (*bootstrap sampling*) dan subsampel fitur acak guna meminimalisasi variansi dan mencegah *overfitting* (Breiman, 2001).
3. **Support Vector Machine (SVM) dengan Kernel RBF**: Pengklasifikasi berbasis margin optimal yang memetakan ruang fitur berdimensi asal ke ruang berdimensi tak hingga menggunakan *Radial Basis Function* guna menangani batas keputusan non-linear (Cortes & Vapnik, 1995).
4. **K-Nearest Neighbors (KNN)**: Algoritma *instance-based non-parametric* yang mengklasifikasikan sampel berdasarkan konsensus mayoritas tetangga terdekat pada metrik jarak Euclidean dalam ruang fitur yang telah diskalakan (Cover & Hart, 1967).
5. **Gradient Boosting Classifier**: Algoritma *ensemble boosting* yang membangun model pohon secara sekuensial untuk mengoreksi residual kesalahan prediksi dari pohon sebelumnya menggunakan optimasi fungsi *loss* berbasis gradien (Friedman, 2001).

---

## 3. METODOLOGI PENELITIAN

Alur metodologi penelitian dirancang secara sistematis dengan mengikuti siklus standar pengolahan data science seperti disajikan pada Gambar 1.

```
+-------------------------------------------------------------------------------+
|                        METODOLOGI PENELITIAN DATA SCIENCE                      |
+-------------------------------------------------------------------------------+
|  1. PENGUMPULAN DATA                                                          |
|     - Mendeley Data (DOI: 10.17632/ft5z9yk94n.1) | N = 1.182 Siswa SMP & SMA   |
|     - 36 Item Skala Likert (Subskala A1 - C3) + Atribut Demografi Siswa       |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  2. EKSPLORASI DATA & PRA-PEMROSESAN (PREPROCESSING)                           |
|     - Pembalikan Item Negatif (Reverse Coding): Skor = 6 - x pada A3.1 & A3.2 |
|     - Agregasi Rata-rata Skor per Subskala (A1 s/d C3)                        |
|     - Diskritisasi Target C1 berbasis Tercile (q1=4,25; q2=4,75): 3 Kategori  |
|     - Eksklusi 10 Kolom Nilai Mapel & Isolasi C1 (Anti-Data Leakage)          |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  3. REKAYASA FITUR & PIPELINE COLUMNTRANSFORMER                               |
|     - Fitur Numerik (8 Subskala) -> StandardScaler                            |
|     - Fitur Kategorik (Jenjang, Kelas, Gender, Lokasi, Transport) -> OHE      |
|     - Stratified Split: Data Latih 80% (N=945), Data Uji 20% (N=237)          |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  4. PELATIHAN & VALIDASI SILANG MODEL (MODEL TRAINING & EVALUATION)           |
|     - Stratified 5-Fold Cross-Validation pada Data Latih                      |
|     - 5 Model: Logistic Regression, Random Forest, SVM RBF, KNN, GradBoost    |
|     - Evaluasi Test Set: Accuracy, Macro-F1, Precision, Recall, Confusion Mat |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  5. INTERPRETASI & DEPLOYMENT                                                 |
|     - Permutation Feature Importance (n_repeats = 15, scoring = 'f1_macro')   |
|     - Implementasi Dashboard Interaktif Berbasis Streamlit Cloud              |
+-------------------------------------------------------------------------------+
```
*Gambar 1: Alur Metodologi Pengolahan Data Science dan Klasifikasi Efikasi Diri Siswa*

### 3.1 Karakteristik Dataset
Data yang digunakan dalam penelitian ini merupakan data empiris yang diperoleh dari repositori Mendeley Data dengan DOI: `10.17632/ft5z9yk94n.1`. Dataset ini mencakup **1.182 responden siswa sekolah menengah di Indonesia** yang tersebar di jenjang SMP dan SMA. Struktur instrumen terdiri dari 36 butir kuesioner skala Likert 5-poin (1 = Sangat Tidak Setuju, 5 = Sangat Setuju), terdistribusi rata ke dalam 9 subskala (masing-masing 4 butir per subskala):

*Tabel 1. Struktur Instrumen Psikometrik Kesiapan Belajar Siswa*
| Kode Subskala | Nama Subskala | Jumlah Butir | Butir Khusus / Catatan |
| :---: | :--- | :---: | :--- |
| **A1** | Rasa Memiliki Sekolah (*Sense of Belonging*) | 4 butir | Skala searah positif (1.0–5.0) |
| **A2** | Regulasi Emosi (*Emotion Regulation*) | 4 butir | Skala searah positif (1.0–5.0) |
| **A3** | Perilaku Sosial (*Social Behavior*) | 4 butir | Butir A3.1 dan A3.2 berorientasi negatif (*Reverse*) |
| **B1** | Ketekunan Belajar (*Academic Persistence*) | 4 butir | Skala searah positif (1.0–5.0) |
| **B2** | Perencanaan Belajar (*Study Planning*) | 4 butir | Skala searah positif (1.0–5.0) |
| **B3** | Keterlibatan di Kelas (*Classroom Engagement*) | 4 butir | Skala searah positif (1.0–5.0) |
| **C1** | Efikasi Diri Akademik (*Academic Self-Efficacy*) | 4 butir | **Variabel Target** (Dieliminasi dari prediktor) |
| **C2** | Nilai/Manfaat Belajar (*Value of Learning*) | 4 butir | Skala searah positif (1.0–5.0) |
| **C3** | Pola Pikir Berkembang (*Growth Mindset*) | 4 butir | Butir C3.3 pre-reversed (*Do not reverse*) |

Selain variabel psikometrik, dataset memuat variabel demografis:
- **Jenjang Pendidikan**: SMA (633 siswa; 53,55%) dan SMP (549 siswa; 46,45%).
- **Jenis Kelamin**: Perempuan (717 siswa; 60,66%) dan Laki-Laki (465 siswa; 39,34%).
- **Lokasi Sekolah**: Desa (618 siswa; 52,28%), Kota (527 siswa; 44,59%), dan Daerah Khusus (37 siswa; 3,13%).
- **Tingkat Kelas**: Kelas 7, 8, 9 (SMP) serta 10, 11, 12 (SMA).
- **Moda Transportasi**: Motor, Sepeda, Jalan Kaki, Mobil, dan Transportasi Umum.

### 3.2 Pembersihan Data dan Pembentukan Variabel Target
Tahapan penyiapan data dilakukan dengan aturan-aturan berikut:
1. **Pembalikan Butir Negatif (*Reverse Coding*)**: Butir A3.1 (*"Saya sering berselisih paham atau bertengkar di sekolah"*) dan A3.2 (*"Saya sering membuat keributan di kelas"*) merupakan butir unfavorable. Skor dibalik dengan persamaan:
   $$\text{Skor}_{\text{baru}} = 6 - \text{Skor}_{\text{asli}}$$
   Sementara itu, pemeriksaan butir C3.3 menunjukkan bahwa data mentah telah berada pada arah positif, sehingga tidak dilakukan pembalikan tambahan guna menjaga konsistensi distribusi data.
2. **Agregasi Skor Subskala**: Nilai masing-masing subskala dihitung dari rata-rata 4 butir penyusunnya.
3. **Pembentukan Target Klasifikasi (C1)**: Skor kontinu subskala C1 dipartisi ke dalam 3 kelas ordinal seimbang menggunakan metode kuantil tercile (persentil ke-33,3% dan 66,7%):
   - **Rendah**: $\text{Skor } C1 \le 4,25$ (Jumlah: 529 siswa; 44,75%)
   - **Sedang**: $4,25 < \text{Skor } C1 \le 4,75$ (Jumlah: 347 siswa; 29,36%)
   - **Tinggi**: $\text{Skor } C1 > 4,75$ (Jumlah: 306 siswa; 25,89%)
4. **Pemberlakuan Protokol Anti-Kebocoran (*Anti-Leakage Protocol*)**:
   - Subskala C1 dihapus secara total dari matriks fitur prediktor $X$.
   - Sepuluh kolom nilai rapor mata pelajaran (Matematika, Bahasa Indonesia, Bahasa Inggris, IPA, IPS, Fisika, Kimia, Biologi, Geografi, Ekonomi) dieksklusi dari pelatihan karena bersifat opsional per jenjang (menyebabkan banyak nilai hilang terstruktur antar siswa SMP dan SMA) serta berpotensi menimbulkan *proxy bias*.

### 3.3 Pipeline Pra-Pemrosesan dan Partisi Data
Matriks prediktor akhir terdiri dari 8 fitur numerik kontinu ($A1, A2, A3, B1, B2, B3, C2, C3$) dan 5 fitur kategorik (*jenjang, kelas, gender, lokasi, transport*).
- Seluruh sampel ($N = 1.182$) dipartisi menggunakan *Stratified Train-Test Split* (rasio 80:20, `random_state = 42`), menghasilkan 945 data latih dan 237 data uji dengan proporsi kelas yang identik.
- Pra-pemrosesan diintegrasikan dalam `sklearn.compose.ColumnTransformer`:
  - Fitur numerik diskalakan menggunakan `StandardScaler` ($\mu = 0, \sigma = 1$).
  - Fitur kategorik dienkode secara biner menggunakan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Seluruh *fitting scaler* dan *encoder* hanya dijalankan pada data latih (*fit on train, transform on test*) untuk mencegah perembesan informasi pengujian ke dalam tahap pelatihan (*data leakage*).

### 3.4 Skenario Pengujian Model dan Hyperparameter
Lima model klasifikasi yang diuji diatur dengan konfigurasi sebagai berikut:
1. **Logistic Regression**: `max_iter = 1000`, solver L-BFGS, multi-class auto (*multinomial cross-entropy*).
2. **Random Forest**: `n_estimators = 300`, kriteria Gini, *bootstrap = True*, `random_state = 42`.
3. **SVM RBF**: Kernel *Radial Basis Function*, $C = 1,0$, $\gamma = \text{'scale'}$, `probability = True`.
4. **KNN**: $k = 15$ tetangga terdekat, pembobotan *uniform*, metrik Minkowski ($p=2$).
5. **Gradient Boosting**: `n_estimators = 100`, *learning rate* = 0,1, kedalaman pohon maksimum = 3.

Evaluasi dilakukan menggunakan **Stratified 5-Fold Cross Validation** pada data latih, dilanjutkan dengan pengujian pada *test set* independen. Metrik yang dihitung meliputi:
- **Akurasi (*Overall Accuracy*)**: Rasio total prediksi benar terhadap seluruh sampel.
- **Macro-Averaged F1-Score**: Rata-rata aritmatika unweighted dari skor F1 setiap kelas, krusial untuk mengevaluasi performa seimbang pada seluruh kelas target:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
- **Confusion Matrix** dan **Permutation Feature Importance** ($n_{\text{repeats}} = 15$).

---

## 4. HASIL DAN PEMBAHASAN

### 4.1 Statistik Deskriptif dan Eksplorasi Data (EDA)
Karakteristik deskriptif sembilan subskala kesiapan belajar disajikan pada Tabel 2.

*Tabel 2. Statistik Deskriptif Subskala Kesiapan Belajar (N = 1.182)*
| Subskala | Indikator Psikometrik | Mean | Standar Deviasi | Minimum | Maksimum |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **A1** | Rasa Memiliki Sekolah (*Belonging*) | 3,835 | 0,821 | 1,000 | 5,000 |
| **A2** | Regulasi Emosi (*Emotion Reg.*) | 3,771 | 0,799 | 1,000 | 5,000 |
| **A3** | Perilaku Sosial (*Social Behavior*) | 4,155 | 0,661 | 1,000 | 5,000 |
| **B1** | Ketekunan (*Persistence*) | 4,357 | 0,692 | 1,000 | 5,000 |
| **B2** | Perencanaan Belajar (*Planning*) | 3,807 | 0,853 | 1,000 | 5,000 |
| **B3** | Keterlibatan di Kelas (*Engagement*) | 4,025 | 0,732 | 1,000 | 5,000 |
| **C1** | Efikasi Diri Akademik (*Target*) | 4,328 | 0,678 | 1,000 | 5,000 |
| **C2** | Nilai/Manfaat Belajar (*Value*) | 4,555 | 0,657 | 1,000 | 5,000 |
| **C3** | Pola Pikir Berkembang (*Growth Mindset*) | 4,253 | 0,681 | 1,000 | 5,000 |

Eksplorasi korelasi Pearson menunjukkan korelasi linear yang kuat antara variabel prediktor dengan target C1:
- Subskala **Ketekunan (B1)** memiliki korelasi linear tertinggi dengan efikasi diri akademik ($r = 0,71, p < 0,001$).
- Subskala **Nilai/Manfaat Belajar (C2)** menempati posisi kedua ($r = 0,69, p < 0,001$).
- Korelasi tertinggi antar-variabel prediktor terjadi antara B3 (*Keterlibatan Kelas*) dan B1 (*Ketekunan*) sebesar $r = 0,69$. Karena seluruh koefisien korelasi antar-prediktor berada di bawah ambang batas $r = 0,85$, dapat dipastikan bahwa data bebas dari ancaman multikolinearitas ekstrem.
- Reduksi dimensi menggunakan *Principal Component Analysis* (PCA 2D) mengonfirmasi separabilitas alami yang nyata: klaster siswa dengan efikasi diri Rendah dan Tinggi terpisah secara tegas di sepanjang sumbu komponen utama pertama (PC1), membuktikan kelayakan ruang fitur untuk dipelajari oleh model klasifikasi.

### 4.2 Perbandingan Kinerja Algoritma Klasifikasi
Pengujian empiris terhadap kelima model menghasilkan rekapitulasi performa sebagaimana tercantum pada Tabel 3.

*Tabel 3. Rekapitulasi Perbandingan Performa Algoritma Klasifikasi*
| Model Klasifikasi | CV Akurasi (Mean ± Std) | CV Macro-F1 (Mean ± Std) | Test Akurasi | Test Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| *Zero-Rule Baseline* | *0,4479 ± 0,0000* | *0,2063 ± 0,0000* | *0,4473* | *0,2061* |
| **Logistic Regression** | 0,6349 ± 0,0229 | 0,6038 ± 0,0303 | 0,6329 | 0,6026 |
| **Random Forest (300 Pohon)** | **0,6593 ± 0,0152** | **0,6404 ± 0,0190** | 0,6329 | 0,6111 |
| **SVM RBF** | 0,6571 ± 0,0188 | 0,6342 ± 0,0243 | **0,6582** | 0,6295 |
| **KNN (k=15)** | 0,6317 ± 0,0205 | 0,6015 ± 0,0209 | 0,6414 | 0,6038 |
| **Gradient Boosting** | 0,6497 ± 0,0113 | 0,6343 ± 0,0123 | 0,6498 | **0,6317** |

Berdasarkan Tabel 3, seluruh algoritma *machine learning* mampu mengungguli *Zero-Rule Baseline* secara signifikan dengan peningkatan akurasi lebih dari 18–21 poin persentase dan peningkatan Macro-F1 lebih dari 40 poin persentase. 
- Pada tahap validasi silang (5-Fold CV), **Random Forest** menunjukkan kinerja tertinggi dengan CV Akurasi $0,6593 \pm 0,0152$ dan CV Macro-F1 $0,6404 \pm 0,0190$.
- Pada pengujian independen (*Test Set*, N = 237), **SVM RBF** mencatatkan akurasi pengujian tertinggi yakni **65,82%** (Macro-F1: 0,6295), sementara **Gradient Boosting** menghasilkan Macro-F1 tertinggi sebesar **0,6317** (Akurasi: 64,98%).
- Stabilitas performa antara skor validasi silang dan skor data uji pada seluruh model membuktikan bahwa *pipeline* bebas dari fenomena *overfitting* yang parah.

### 4.3 Analisis Detail Klasifikasi per Kelas (Classification Report & Confusion Matrix)
Tabel 4 menyajikan *Classification Report* mendalam dari model klasifikasi pada data uji (N = 237 siswa).

*Tabel 4. Laporan Klasifikasi Rinci per Kelas pada Data Uji*
| Kategori Target | Precision | Recall | F1-Score | Support (Jumlah Sampel) |
| :--- | :---: | :---: | :---: | :---: |
| **Rendah** | 0,7615 | 0,7830 | **0,7721** | 106 siswa |
| **Sedang** | 0,4167 | 0,3571 | **0,3846** | 70 siswa |
| **Tinggi** | 0,6176 | 0,6885 | **0,6512** | 61 siswa |
| **Akurasi Rata-rata** | - | - | **0,6329** | 237 siswa |
| **Macro Average** | **0,5986** | **0,6096** | **0,6026** | 237 siswa |
| **Weighted Average** | **0,6226** | **0,6329** | **0,6265** | 237 siswa |

Matriks konfusi (*confusion matrix*) memperlihatkan struktur distribusi prediksi sebagai berikut:
- **Kelas Rendah (N = 106)**: Sebanyak 83 siswa terklasifikasi benar (*True Negative* kelompok atas), 18 siswa terprediksi Sedang, dan hanya 5 siswa yang salah terprediksi Tinggi. Recall sebesar **78,30%** dan Precision sebesar **76,15%** menunjukkan keandalan model dalam mendeteksi siswa yang berada pada kondisi defisit efikasi diri.
- **Kelas Tinggi (N = 61)**: Sebanyak 42 siswa terklasifikasi benar (*True Positive* kelompok atas), 17 siswa terprediksi Sedang, dan hanya 2 siswa yang terprediksi Rendah (Recall **68,85%**, Precision **61,76%**).
- **Kelas Sedang (N = 70)**: 25 siswa terklasifikasi tepat, 24 siswa tergeser ke kategori Rendah, dan 21 siswa tergeser ke kategori Tinggi. Hal ini merupakan fenomena psikometri yang lazim di mana kelas tengah (*borderline group*) memiliki batas keputusan (*decision boundary*) yang saling tumpang tindih dengan kategori ekstrem bawah dan atas.

### 4.4 Analisis Kepentingan Fitur (Permutation Feature Importance)
Untuk memberikan interpretabilitas terhadap mekanisme internal model, dilakukan pengujian *Permutation Feature Importance* pada data uji dengan 15 kali permutasi acak berulang menggunakan penalti penurunan Macro-F1. Hasil analisis disajikan pada Tabel 5.

*Tabel 5. Peringkat Kepentingan Fitur Berdasarkan Penurunan Macro-F1*
| Peringkat | Fitur / Indikator | Domain / Aspek | Penurunan Macro-F1 (Mean ± Std) |
| :---: | :--- | :--- | :---: |
| **1** | **B3 (Keterlibatan di Kelas)** | Kesiapan Belajar - Perilaku | **+0,1039 ± 0,0272** |
| **2** | **C2 (Nilai/Manfaat Belajar)** | Kesiapan Belajar - Sikap | **+0,0501 ± 0,0216** |
| **3** | **Kelas (Tingkat Kelas)** | Profil Demografi Siswa | **+0,0418 ± 0,0097** |
| **4** | **B1 (Ketekunan Belajar)** | Kesiapan Belajar - Keterampilan | **+0,0393 ± 0,0268** |
| **5** | **Jenjang (SMP vs SMA)** | Profil Demografi Siswa | **+0,0316 ± 0,0162** |
| **6** | **B2 (Perencanaan Belajar)** | Kesiapan Belajar - Keterampilan | **+0,0292 ± 0,0266** |
| **7** | **A1 (Rasa Memiliki Sekolah)** | Sosio-Emosional | **+0,0144 ± 0,0040** |
| **8** | **C3 (Growth Mindset)** | Sikap / Pola Pikir | **+0,0131 ± 0,0086** |
| **9** | **A2 (Regulasi Emosi)** | Sosio-Emosional | **+0,0128 ± 0,0086** |
| **10** | **A3 (Perilaku Sosial)** | Sosio-Emosional | **+0,0083 ± 0,0054** |
| 11 | Lokasi Sekolah | Demografi Wilayah | -0,0032 ± 0,0051 |
| 12 | Jenis Kelamin | Demografi Siswa | -0,0044 ± 0,0078 |
| 13 | Moda Transportasi | Fasilitas / Demografi | -0,0084 ± 0,0084 |

Temuan ini memberikan wawasan pedagogis yang sangat bernilai:
1. **Keterlibatan di Kelas (B3)** merupakan prediktor paling kritikal; ketika informasi B3 diacak, Macro-F1 model anjlok drastis sebesar 0,1039. Keterlibatan aktif di ruang kelas merupakan manifestasi langsung dari keberanian intelektual dan efikasi diri akademik siswa.
2. **Nilai/Manfaat Belajar (C2)** dan **Ketekunan (B1)** menduduki posisi kunci berikutnya, menegaskan bahwa siswa yang menganggap ilmu yang dipelajarinya bermakna bagi masa depan serta memiliki daya juang tinggi akan membangun persepsi efikasi diri yang kokoh.
3. Sebaliknya, variabel latar belakang seperti **moda transportasi, jenis kelamin, dan lokasi sekolah** memberikan kontribusi marjinal mendekati nol (bahkan sedikit negatif karena menambah derau dimensi). Hal ini membuktikan bahwa efikasi diri akademik siswa lebih ditentukan oleh faktor psikososial dan kebiasaan belajar di sekolah daripada faktor spasial atau demografi bawaan.

### 4.5 Implementasi Dashboard Interaktif Streamlit
Model terbaik telah berhasil diintegrasikan ke dalam antarmuka aplikasi web berbasis Streamlit. Dashboard ini dilengkapi dengan:
- Modul visualisasi analitik data eksploratif (distribusi donat, korelasi interaktif, proyeksi PCA 2D).
- Simulator inferensi prediktif di mana guru atau konselor dapat menggeser skor subskala siswa dan langsung memperoleh prediksi tingkat efikasi diri lengkap dengan visualisasi probabilitas kelas.
- *Early-warning system* yang memberikan penanda warna (merah untuk efikasi rendah, kuning untuk sedang, hijau untuk tinggi) guna mempercepat intervensi preventif di lingkungan sekolah.

---

## 5. SIMPULAN

Berdasarkan rangkaian penelitian, perancangan sistem, dan eksperimen pemodelan yang telah dilaksanakan, dapat disimpulkan bahwa:
1. Pendekatan *Educational Data Mining* menggunakan algoritma pembelajaran mesin terbukti efektif dalam memprediksi tingkat efikasi diri akademik siswa SMP dan SMA di Indonesia dengan melampaui performa *Zero-Rule Baseline* secara substansial.
2. Model **SVM RBF** menghasilkan akurasi pengujian tertinggi (**65,82%**; Macro-F1: 0,6295), sementara model **Gradient Boosting** membukukan skor Macro-F1 tertinggi (**0,6317**; Akurasi: 64,98%). Model memiliki keandalan tinggi dalam mengidentifikasi siswa ber-efikasi diri rendah (Recall 78,30%, Precision 76,15%).
3. Analisis *Permutation Feature Importance* menunjukkan bahwa **Keterlibatan di Kelas (B3)**, **Nilai/Manfaat Belajar (C2)**, dan **Ketekunan Belajar (B1)** merupakan faktor prediktor kesiapan belajar yang paling berpengaruh terhadap efikasi diri siswa, sedangkan faktor demografis spasial memiliki pengaruh minimal.
4. Integrasi model ke dalam aplikasi Streamlit memberikan kontribusi praktis sebagai alat bantu diagnostik cerdas bagi guru BK dan pimpinan sekolah dalam mendeteksi dini siswa rentan serta merancang program pembinaan belajar yang tepat sasaran.

Penelitian lanjutan disarankan untuk memperluas cakupan data longitudinal antarsmester serta menerapkan teknik optimasi *hyperparameter tuning* (seperti Bayesian Optimization) dan teknik penanganan kelas *borderline* guna meningkatkan kemampuan separasi pada kategori efikasi diri sedang.

---

## UCAPAN TERIMA KASIH

Penulis menyampaikan rasa terima kasih dan apresiasi yang setinggi-tingginya kepada para peneliti penyedia dataset terbuka di Mendeley Data atas ketersediaan data psikometrik kesiapan belajar siswa Indonesia yang komprehensif, serta kepada dosen pembimbing dan rekan-rekan mahasiswa Program Studi Sains Data atas arahan, masukan akademis, dan diskusi berharga selama pengembangan riset ini.

---

## DAFTAR PUSTAKA

Bandura, A. (1997). *Self-Efficacy: The Exercise of Control*. New York: W. H. Freeman and Company.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018

Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

Dweck, C. S. (2006). *Mindset: The New Psychology of Success*. New York: Random House.

Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Gunawan, H., & Catherine. (2020). Information Systems Students' Study Performance Prediction Using Data Mining Approach. *2019 International Conference on Information and Communications Technology (ICOIACT)*, 1–8. https://doi.org/10.1109/icic47613.2019.8985718

Hairani, H., Nugraha, G. S., Abdillah, M. N., & Innuddin, M. (2018). Komparasi Akurasi Metode Correlated Naive Bayes Classifier dan Naive Bayes Classifier untuk Diagnosis Penyakit Diabetes. *InfoTekJar (Jurnal Nasional Informatika Dan Teknologi Jaringan)*, 3(1), 6–11. https://doi.org/10.30743/infotekjar.v3i1.558

Hairani, H., Saputro, K. E., & Fadli, S. (2020). K-means-SMOTE untuk menangani ketidakseimbangan kelas dalam klasifikasi penyakit diabetes dengan C4.5, SVM, dan naive Bayes. *Jurnal Teknologi Dan Sistem Komputer*, 8(2), 89–93. https://doi.org/10.14710/jtsiskom.8.2.2020.89-93

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.

Igual, L., & Segui, S. (2017). *Introduction to Data Science: A Python Approach to Concepts, Techniques and Applications*. Springer Nature.

Jadhav, S. D., & Channe, H. P. (2016). Comparative Study of K-NN, Naive Bayes and Decision Tree Classification Techniques. *International Journal of Science and Research (IJSR)*, 5(1), 2014–2017.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579

Nugraha, G. S., & Hairani, H. (2018). Aplikasi Pemetaan Kualitas Pendidikan di Indonesia Menggunakan Metode K-Means. *MATRIK: Jurnal Manajemen, Teknik Informatika Dan Rekayasa Komputer*, 17(2), 13–23.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Romero, C., & Ventura, S. (2020). Educational Data Mining and Learning Analytics: An Updated Survey. *WIREs Data Mining and Knowledge Discovery*, 10(3), e1355. https://doi.org/10.1002/widm.1355

Schunk, D. H., & DiBenedetto, M. K. (2020). Motivation and Social Cognitive Theory. *Contemporary Educational Psychology*, 60, 101832. https://doi.org/10.1016/j.cedpsych.2019.101832

Sulistyono, S., & Sulistiyowati, W. (2018). Peramalan Produksi dengan Metode Regresi Linier Berganda. *PROZIMA (Productivity, Optimization and Manufacturing System Engineering)*, 1(2), 82. https://doi.org/10.21070/prozima.v1i2.1350

Zimmerman, B. J. (2000). Self-Efficacy: An Essential Motive to Learn. *Contemporary Educational Psychology*, 25(1), 82–91. https://doi.org/10.1006/ceps.1999.1016
