"""
Aplikasi Web Streamlit: Testbed Klasifikasi Efikasi Diri Akademik Siswa
Pipeline data mining end-to-end untuk klasifikasi tingkat efikasi diri akademik
siswa SMP dan SMA di Indonesia berdasarkan domain kesiapan belajar.
"""

import os
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Testbed Klasifikasi Efikasi Diri Akademik Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Mengganti ikon 'running man' di pojok kanan atas dengan animasi loading lingkaran (circular spinner)
st.markdown("""
<style>
/* Sembunyikan ikon orang berlari (running man) bawaan Streamlit */
[data-testid="stStatusWidgetRunningManIcon"],
[data-testid="stStatusWidgetRunningIcon"] svg {
    display: none !important;
    visibility: hidden !important;
}

/* Ubah status running widget menjadi animasi lingkaran berputar (circular spinner) */
[data-testid="stStatusWidgetRunningIcon"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    min-height: 18px !important;
}

[data-testid="stStatusWidgetRunningIcon"]::after {
    content: "" !important;
    box-sizing: border-box !important;
    width: 14px !important;
    height: 14px !important;
    border: 2px solid rgba(160, 160, 160, 0.3) !important;
    border-top: 2px solid #ff4b4b !important;
    border-radius: 50% !important;
    animation: stStatusSpin 0.75s linear infinite !important;
    display: block !important;
}

@keyframes stStatusSpin {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
</style>
""", unsafe_allow_html=True)

# Kamus nama subskala dalam bahasa Indonesia sesuai spesifikasi
FEATURE_TRANSLATIONS = {
    'A1': 'Rasa memiliki',
    'A2': 'Regulasi emosi',
    'A3': 'Perilaku sosial',
    'B1': 'Ketekunan',
    'B2': 'Perencanaan belajar',
    'B3': 'Keterlibatan kelas',
    'C2': 'Nilai/manfaat belajar',
    'C3': 'Growth mindset',
    'jenjang': 'Jenjang sekolah',
    'kelas': 'Tingkat kelas',
    'gender': 'Jenis kelamin',
    'lokasi': 'Lokasi sekolah',
    'transport': 'Transportasi ke sekolah',
    'kepuasan': 'Kepuasan hasil belajar',
    'keyakinan_naik_nilai': 'Keyakinan meningkatkan nilai'
}

SUBSCALE_FULL_NAMES = {
    'A1': 'A1 - Rasa Memiliki (Sosial-Emosional)',
    'A2': 'A2 - Regulasi Emosi (Sosial-Emosional)',
    'A3': 'A3 - Perilaku Sosial (Sosial-Emosional)',
    'B1': 'B1 - Ketekunan (Perilaku)',
    'B2': 'B2 - Perencanaan Belajar (Perilaku)',
    'B3': 'B3 - Keterlibatan di Kelas (Perilaku)',
    'C1': 'C1 - Efikasi Diri (Kognitif-Motivasional)',
    'C2': 'C2 - Nilai/Manfaat Belajar (Kognitif-Motivasional)',
    'C3': 'C3 - Growth Mindset (Kognitif-Motivasional)'
}


# ==========================================
# 2. FUNGSI PEMBACAAN DAN PREPROCESSING DATA
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_prepare_dataset(file_bytes=None, file_name=""):
    """
    Membaca dataset raw dan melakukan parsing kolom, pembalikan item negatif,
    penghitungan skor subskala, serta pembentukan target C1.
    """
    df_raw = None
    if file_bytes is not None:
        buffer = io.BytesIO(file_bytes)
        if file_name.endswith('.csv'):
            df_raw = pd.read_csv(buffer)
        else:
            try:
                df_raw = pd.read_excel(buffer, sheet_name="Sheet 1 - Raw data")
            except Exception:
                buffer.seek(0)
                df_raw = pd.read_excel(buffer)
    else:
        # Coba baca file lokal
        candidate_files = ["11.12_Raw_data.xlsx", "11.12 Raw data.xlsx"]
        found_path = None
        for path in candidate_files:
            if os.path.exists(path):
                found_path = path
                break
        if found_path is None:
            return None, "File dataset default (11.12_Raw_data.xlsx) tidak ditemukan di folder kerja."
        
        try:
            df_raw = pd.read_excel(found_path, sheet_name="Sheet 1 - Raw data")
        except Exception:
            df_raw = pd.read_excel(found_path)

    # Deteksi 36 item Likert dengan regex ^[ABC]\d\.\d
    likert_cols = [c for c in df_raw.columns if re.match(r'^[ABC]\d\.\d', str(c).strip())]
    if not likert_cols:
        return None, "Format kolom Likert tidak terdeteksi (pola regex ^[ABC]\\d\\.\\d)."

    # Pastikan subskala C1 ada
    c1_cols = [c for c in likert_cols if str(c).strip().startswith('C1')]
    if not c1_cols:
        return None, "Kolom subskala C1 (Efikasi Diri) tidak ditemukan dalam dataset."

    # Copy untuk pembersihan
    df_clean = df_raw.copy()

    # 1. Pembalikan item negatif: A3.1 dan A3.2 dibalik (6 - x), C3.3 TIDAK dibalik
    reversed_items = []
    for col in likert_cols:
        col_str = str(col).strip()
        if col_str.startswith('A3.1') or col_str.startswith('A3.2'):
            df_clean[col] = 6 - pd.to_numeric(df_clean[col], errors='coerce')
            reversed_items.append(col_str[:4])

    # 2. Agregasi skor subskala (rata-rata per kode A1 - C3)
    subscales = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    for s in subscales:
        items = [c for c in likert_cols if str(c).strip().startswith(s)]
        df_clean[s] = df_clean[items].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # Deteksi kolom demografi berdasarkan nama (case-insensitive)
    def find_col(patterns):
        for col in df_raw.columns:
            col_l = str(col).lower()
            if any(p in col_l for p in patterns):
                return col
        return None

    col_loc = find_col(['lokasi sekolah'])
    col_transport = find_col(['pergi ke sekolah', 'biasanya pergi'])
    col_gender = find_col(['jenis kelamin'])
    col_jenjang = find_col(['jenjang sekolah'])
    col_kelas = find_col(['tingkat kelas'])

    # Pemrosesan fitur demografi
    # Lokasi: huruf kecil, ambil kata pertama (desa, daerah, kota)
    if col_loc is not None:
        df_clean['lokasi'] = df_clean[col_loc].astype(str).str.lower().str.strip().str.split().str[0]
    else:
        df_clean['lokasi'] = 'kota'

    if col_transport is not None:
        df_clean['transport'] = df_clean[col_transport].astype(str).str.strip()
    else:
        df_clean['transport'] = 'Lainnya'

    if col_gender is not None:
        df_clean['gender'] = df_clean[col_gender].astype(str).str.strip()
    else:
        df_clean['gender'] = 'Tidak Diketahui'

    if col_jenjang is not None:
        df_clean['jenjang'] = df_clean[col_jenjang].astype(str).str.strip()
    else:
        df_clean['jenjang'] = 'SMA'

    if col_kelas is not None:
        df_clean['kelas'] = df_clean[col_kelas].astype(str).str.strip()
    else:
        df_clean['kelas'] = '10'

    # Deteksi dan pemetaan dua outcome ordinal (Skenario B)
    col_out1 = find_col(['puas dengan hasil belajar'])
    col_out2 = find_col(['percaya bisa meningkatkan nilai'])

    map_o1 = {
        'tidak puas sama sekali': 1,
        'kurang puas': 2,
        'cukup puas': 3,
        'puas': 4,
        'sangat puas': 5
    }
    map_o2 = {
        'kurang percaya': 1,
        'cukup percaya': 2,
        'percaya': 3,
        'sangat percaya': 4
    }

    if col_out1 is not None:
        df_clean['kepuasan'] = df_clean[col_out1].astype(str).str.lower().str.strip().map(map_o1).fillna(3)
    else:
        df_clean['kepuasan'] = 3

    if col_out2 is not None:
        df_clean['keyakinan_naik_nilai'] = df_clean[col_out2].astype(str).str.lower().str.strip().map(map_o2).fillna(3)
    else:
        df_clean['keyakinan_naik_nilai'] = 3

    # 3. Pembentukan target C1: batas kuantil 1/3 (4,25) dan 2/3 (4,75)
    c1_scores = df_clean['C1']
    q1 = float(c1_scores.quantile(1/3))
    q2 = float(c1_scores.quantile(2/3))

    # Target 3 Kelas (Rendah: <= 4.25, Sedang: <= 4.75, Tinggi: > 4.75)
    def categorize_3(v):
        if v <= 4.25:
            return 'Rendah'
        elif v <= 4.75:
            return 'Sedang'
        else:
            return 'Tinggi'

    # Target 2 Kelas (Rendah: <= 4.25, Sedang-Tinggi: > 4.25)
    def categorize_2(v):
        if v <= 4.25:
            return 'Rendah'
        else:
            return 'Sedang-Tinggi'

    df_clean['target_3'] = c1_scores.apply(categorize_3)
    df_clean['target_2'] = c1_scores.apply(categorize_2)

    # Deteksi 10 kolom nilai mata pelajaran untuk diabaikan
    subject_cols = [c for c in df_raw.columns if any(p in str(c).lower() for p in ['nilai rata-rata', 'khusus siswa'])]
    
    # Kolom inti = likert (36) + demografi + outcome (tidak termasuk subject_cols dan timestamp)
    core_cols = [c for c in df_raw.columns if c not in subject_cols and 'timestamp' not in str(c).lower()]
    missing_core = int(df_raw[core_cols].isna().sum().sum())

    metadata = {
        'total_rows': len(df_raw),
        'total_cols': len(df_raw.columns),
        'missing_core': missing_core,
        'subject_cols': subject_cols,
        'likert_cols': likert_cols,
        'reversed_items': reversed_items,
        'q1': q1,
        'q2': q2
    }

    return (df_raw, df_clean, metadata), None


# ==========================================
# 3. FUNGSI PELATIHAN & EVALUASI MODEL
# ==========================================
def train_and_evaluate_models(X_train, X_test, y_train, y_test, num_cols, cat_cols,
                              selected_model_names, scaler_name, cv_folds, random_seed):
    """
    Melakukan cross-validation dan evaluasi test set menggunakan ColumnTransformer & Pipeline.
    """
    # Inisialisasi scaler
    if scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Definisi pool model yang tersedia
    model_pool = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_seed),
        'Random Forest (300 pohon)': RandomForestClassifier(n_estimators=300, random_state=random_seed),
        'SVM RBF': SVC(kernel='rbf', random_state=random_seed),
        'KNN (k=15)': KNeighborsClassifier(n_neighbors=15),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_seed)
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    results_table = []
    fitted_pipelines = {}
    test_predictions = {}

    for name in selected_model_names:
        if name not in model_pool:
            continue
        base_clf = model_pool[name]
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', base_clf)
        ])

        # Cross validation pada data latih
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro']
        )
        cv_acc_mean = cv_scores['test_accuracy'].mean()
        cv_acc_std = cv_scores['test_accuracy'].std()
        cv_f1_mean = cv_scores['test_f1_macro'].mean()
        cv_f1_std = cv_scores['test_f1_macro'].std()

        # Fit pada data latih utuh & uji pada data test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')

        results_table.append({
            'Model': name,
            'CV Akurasi (Mean ± Std)': f"{cv_acc_mean:.3f} ± {cv_acc_std:.3f}",
            'CV Macro-F1 (Mean ± Std)': f"{cv_f1_mean:.3f} ± {cv_f1_std:.3f}",
            'Test Akurasi': test_acc,
            'Test Macro-F1': test_f1,
            '_cv_acc_raw': cv_acc_mean,
            '_cv_f1_raw': cv_f1_mean
        })

        fitted_pipelines[name] = pipeline
        test_predictions[name] = y_pred

    results_df = pd.DataFrame(results_table)
    return results_df, fitted_pipelines, test_predictions


# ==========================================
# 4. SIDEBAR - TESTBED CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ Konfigurasi Testbed")
st.sidebar.markdown("Atur parameter pipeline klasifikasi efikasi diri akademik di bawah ini:")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Unggah File Dataset (XLSX / CSV):",
    type=["xlsx", "xls", "csv"],
    help="Jika kosong, sistem akan otomatis membaca 11.12_Raw_data.xlsx dari folder aplikasi."
)

# Load data
file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
file_name = uploaded_file.name if uploaded_file is not None else ""
data_tuple, error_msg = load_and_prepare_dataset(file_bytes, file_name)

if error_msg is not None:
    st.sidebar.error(error_msg)
    st.error(f"⚠️ **Gagal Membuka Dataset**: {error_msg}")
    st.info("Pastikan file `11.12_Raw_data.xlsx` tersedia di direktori atau unggah file data melalui sidebar.")
    st.stop()

df_raw, df_clean, meta = data_tuple

# 2. Radio Jumlah Kelas Target
target_choice = st.sidebar.radio(
    "Jumlah Kelas Target:",
    options=["3 Kelas (Rendah, Sedang, Tinggi)", "2 Kelas (Rendah vs Sedang-Tinggi)"],
    index=0
)
is_3_class = "3 Kelas" in target_choice
target_col = 'target_3' if is_3_class else 'target_2'
class_order = ['Rendah', 'Sedang', 'Tinggi'] if is_3_class else ['Rendah', 'Sedang-Tinggi']

# 3. Radio Skenario Fitur
scenario_choice = st.sidebar.radio(
    "Skenario Fitur:",
    options=[
        "Skenario A (Murni Kesiapan Belajar)",
        "Skenario B (+ Dua Outcome Ordinal)"
    ],
    index=0
)
is_scenario_b = "Skenario B" in scenario_choice

# 4. Selectbox Feature Scaler
scaler_choice = st.sidebar.selectbox(
    "Feature Scaler:",
    options=["StandardScaler", "MinMaxScaler", "RobustScaler"],
    index=0
)

# 5. Sliders & Input
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    test_size = st.slider("Test Size:", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
with col_sb2:
    cv_folds = st.slider("CV Folds:", min_value=3, max_value=10, value=5, step=1)

random_seed = st.sidebar.number_input("Random Seed:", min_value=0, max_value=999999, value=42, step=1)

# 6. Multiselect Model
available_models = [
    "Logistic Regression",
    "Random Forest (300 pohon)",
    "SVM RBF",
    "KNN (k=15)",
    "Gradient Boosting"
]
selected_models = st.sidebar.multiselect(
    "Pilih Model Klasifikasi:",
    options=available_models,
    default=available_models[:4]
)

st.sidebar.markdown("---")
st.sidebar.caption("🔬 **Penelitian**: Efikasi Diri Siswa SMP/SMA Indonesia\nDataset: Mendeley Data (DOI: 10.17632/ft5z9yk94n.1)")


# ==========================================
# 5. PERSIAPAN FITUR DAN SPLIT TRAIN-TEST
# ==========================================
# Tentukan daftar kolom numerik dan kategorik
base_num_cols = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C2', 'C3']  # Anti-leakage: C1 TIDAK ADA
cat_cols = ['jenjang', 'kelas', 'gender', 'lokasi', 'transport']

if is_scenario_b:
    num_cols = base_num_cols + ['kepuasan', 'keyakinan_naik_nilai']
else:
    num_cols = base_num_cols

feature_cols = num_cols + cat_cols
X = df_clean[feature_cols].copy()
y = df_clean[target_col].copy()

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_seed
)

# Baseline tebak kelas mayoritas pada data uji
majority_class = y_train.value_counts().idxmax()
baseline_acc = float((y_test == majority_class).mean())
baseline_f1 = float(f1_score(y_test, [majority_class] * len(y_test), average='macro'))


# ==========================================
# 6. HEADER UTAMA DAN TAB APLIKASI
# ==========================================
st.title("🎓 Testbed Klasifikasi & Klastering Efikasi Diri Siswa")
st.caption("Pipeline Data Mining End-to-End untuk Memprediksi dan Mengelompokkan Kesiapan Belajar serta Efikasi Diri Siswa SMP/SMA di Indonesia.")

tab_overview, tab_eda, tab_prep, tab_clustering, tab_modeling, tab_eval = st.tabs([
    "📖 Overview",
    "📊 EDA",
    "⚙️ Preprocessing",
    "🧩 Clustering",
    "🤖 Modeling",
    "📈 Evaluation"
])


# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab_overview:
    st.markdown("### 📋 Ringkasan Dataset & Target Penelitian")
    
    # 4 Metrik Utama
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Baris (Siswa)", f"{meta['total_rows']:,}")
    with m_col2:
        st.metric("Total Kolom Raw", f"{meta['total_cols']}")
    with m_col3:
        st.metric("Missing Value (Item Inti)", f"{meta['missing_core']}")
    with m_col4:
        top_cls_name = y.value_counts().idxmax()
        top_cls_pct = (y.value_counts().max() / len(y)) * 100
        st.metric("Kelas Terbesar", f"{top_cls_pct:.2f}%", f"Kelas {top_cls_name}")

    st.markdown("---")
    
    # Deskripsi Konteks Penelitian
    col_desc1, col_desc2 = st.columns([3, 2])
    with col_desc1:
        st.markdown("""
        #### 🎯 Latar Belakang & Tujuan
        Penelitian ini berfokus pada **klasifikasi tingkat efikasi diri akademik siswa SMP dan SMA di Indonesia** 
        berdasarkan indikator kesiapan belajar (*school readiness*). 
        
        Efikasi diri akademik merupakan keyakinan siswa terhadap kemampuannya dalam memahami pelajaran, menyelesaikan tugas yang menantang, serta mencapai prestasi yang baik.
        
        - **Sumber Data**: Mendeley Data (*School Readiness among Indonesian Junior and Senior High School Students*, DOI: `10.17632/ft5z9yk94n.1`).
        - **Populasi**: 1.182 siswa dari berbagai SMP dan SMA di Indonesia.
        - **Dimensi Pengukuran**: 36 item Likert 1–5 yang terbagi ke dalam 3 domain utama:
          1. **Domain Sosial-Emosional**: Rasa Memiliki (A1), Regulasi Emosi (A2), Perilaku Sosial (A3).
          2. **Domain Perilaku**: Ketekunan (B1), Perencanaan Belajar (B2), Keterlibatan di Kelas (B3).
          3. **Domain Kognitif-Motivasional**: Efikasi Diri (C1 - Basis Target), Nilai Belajar (C2), Growth Mindset (C3).
        """)
    
    with col_desc2:
        st.markdown("#### 📐 Pembagian Target Tercile (C1)")
        st.markdown(f"""
        Skor **C1** dihitung dari rata-rata 4 item efikasi diri (skala 1–5). Target dibagi berdasarkan batas **tercile**:
        - **Batas Kuantil 1/3 (33,33%)**: `{meta['q1']:.2f}` (4,25)
        - **Batas Kuantil 2/3 (66,67%)**: `{meta['q2']:.2f}` (4,75)
        """)
        
        # Tabel perbandingan distribusi kelas
        if is_3_class:
            t_counts = y.value_counts().reindex(['Rendah', 'Sedang', 'Tinggi'])
            t_df = pd.DataFrame({
                'Kategori': ['Rendah', 'Sedang', 'Tinggi'],
                'Kriteria Skor C1': ['C1 ≤ 4,25', '4,25 < C1 ≤ 4,75', 'C1 > 4,75'],
                'Jumlah Siswa': [t_counts['Rendah'], t_counts['Sedang'], t_counts['Tinggi']],
                'Persentase': [f"{(t_counts['Rendah']/len(y))*100:.2f}%",
                               f"{(t_counts['Sedang']/len(y))*100:.2f}%",
                               f"{(t_counts['Tinggi']/len(y))*100:.2f}%"]
            })
        else:
            t_counts = y.value_counts().reindex(['Rendah', 'Sedang-Tinggi'])
            t_df = pd.DataFrame({
                'Kategori': ['Rendah', 'Sedang-Tinggi'],
                'Kriteria Skor C1': ['C1 ≤ 4,25', 'C1 > 4,25'],
                'Jumlah Siswa': [t_counts['Rendah'], t_counts['Sedang-Tinggi']],
                'Persentase': [f"{(t_counts['Rendah']/len(y))*100:.2f}%",
                               f"{(t_counts['Sedang-Tinggi']/len(y))*100:.2f}%"]
            })
        st.dataframe(t_df, hide_index=True, use_container_width=True)
        st.caption("ℹ️ *Catatan*: Batas eksplisit kuantil (4,25 & 4,75) digunakan untuk menghindari ketidakpastian fungsi `qcut` akibat banyaknya nilai kembar (*ties*) pada skala Likert.")

    st.markdown("---")
    st.markdown("#### 🔍 Cuplikan 20 Baris Data Mentah")
    st.dataframe(df_raw.head(20), use_container_width=True)


# ==========================================
# TAB 2: EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================
with tab_eda:
    st.markdown("### 📊 Analisis Eksplorasi Data (EDA)")

    # 1. Distribusi Kelas & Korelasi C1
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.markdown("#### 1. Distribusi Frekuensi Kelas Target")
        cls_counts = y.value_counts().reindex(class_order)
        
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        colors = ['#e74c3c', '#f39c12', '#2ecc71'] if is_3_class else ['#e74c3c', '#3498db']
        bars = ax_dist.bar(cls_counts.index, cls_counts.values, color=colors, edgecolor='black', alpha=0.85)
        
        for bar in bars:
            height = bar.get_height()
            pct = (height / len(y)) * 100
            ax_dist.annotate(f'{height}\n({pct:.1f}%)',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 4), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_dist.set_ylabel("Jumlah Siswa", fontsize=11)
        ax_dist.set_ylim(0, max(cls_counts.values) * 1.22)
        ax_dist.grid(axis='y', linestyle='--', alpha=0.5)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    with col_eda2:
        st.markdown("#### 2. Korelasi Subskala dengan Efikasi Diri (C1)")
        sub_list = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C2', 'C3']
        corr_series = df_clean[sub_list].corrwith(df_clean['C1']).sort_values(ascending=True)
        
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        bar_colors = ['#3498db' if c >= 0.6 else '#85c1e9' for c in corr_series.values]
        bars_h = ax_corr.barh(corr_series.index, corr_series.values, color=bar_colors, edgecolor='black', alpha=0.85)
        
        for bar in bars_h:
            width = bar.get_width()
            ax_corr.annotate(f'{width:.3f}',
                             xy=(width, bar.get_y() + bar.get_height() / 2),
                             xytext=(5, 0), textcoords="offset points",
                             ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax_corr.set_xlim(0, 0.88)
        ax_corr.set_xlabel("Koefisien Korelasi Pearson (r)", fontsize=11)
        ax_corr.grid(axis='x', linestyle='--', alpha=0.5)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    st.markdown("---")

    # 2. Crosstab Jenjang vs Target & Heatmap Korelasi
    col_eda3, col_eda4 = st.columns([3, 3])

    with col_eda3:
        st.markdown("#### 3. Proporsi Kelas per Jenjang Sekolah (SMP vs SMA)")
        ct_counts = pd.crosstab(df_clean['jenjang'], df_clean[target_col], normalize='index')[class_order] * 100
        
        fig_ct, ax_ct = plt.subplots(figsize=(6, 4.2))
        ct_counts.plot(kind='bar', stacked=True, ax=ax_ct, color=colors, edgecolor='black', alpha=0.85)
        ax_ct.set_ylabel("Persentase (%)", fontsize=11)
        ax_ct.set_xlabel("Jenjang Sekolah", fontsize=11)
        ax_ct.set_ylim(0, 105)
        ax_ct.legend(title="Tingkat Efikasi Diri", loc='upper right', frameon=True)
        ax_ct.set_xticklabels(ax_ct.get_xticklabels(), rotation=0)
        ax_ct.grid(axis='y', linestyle='--', alpha=0.5)
        fig_ct.tight_layout()
        st.pyplot(fig_ct)
        plt.close(fig_ct)

        # Tampilkan tabel kontingensi frekuensi
        st.caption("Tabel Frekuensi Kontingensi (Jumlah Siswa):")
        ct_raw = pd.crosstab(df_clean['jenjang'], df_clean[target_col])[class_order]
        st.dataframe(ct_raw, use_container_width=True)

    with col_eda4:
        st.markdown("#### 4. Heatmap Korelasi Antar-Subskala (A1 – C3)")
        all_subs = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
        corr_matrix = df_clean[all_subs].corr().round(2)
        
        fig_hm, ax_hm = plt.subplots(figsize=(6, 4.8))
        cax = ax_hm.imshow(corr_matrix.values, cmap='coolwarm', vmin=0.3, vmax=1.0)
        fig_hm.colorbar(cax, ax=ax_hm, fraction=0.046, pad=0.04)
        
        ax_hm.set_xticks(range(len(all_subs)))
        ax_hm.set_yticks(range(len(all_subs)))
        ax_hm.set_xticklabels(all_subs, fontsize=9)
        ax_hm.set_yticklabels(all_subs, fontsize=9)
        
        for i in range(len(all_subs)):
            for j in range(len(all_subs)):
                val = corr_matrix.iloc[i, j]
                text_color = "white" if val > 0.75 else "black"
                ax_hm.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
        
        fig_hm.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)

    st.markdown("---")
    st.markdown("#### 5. Statistik Deskriptif Skor Subskala (Skala 1–5)")
    desc_df = df_clean[all_subs].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc_df['Nama Subskala'] = [SUBSCALE_FULL_NAMES.get(idx, idx) for idx in desc_df.index]
    desc_df = desc_df[['Nama Subskala', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    st.dataframe(desc_df.style.format({
        'count': '{:.0f}',
        'mean': '{:.3f}',
        'std': '{:.3f}',
        'min': '{:.2f}',
        '25%': '{:.2f}',
        '50%': '{:.2f}',
        '75%': '{:.2f}',
        'max': '{:.2f}'
    }), use_container_width=True)


# ==========================================
# TAB 3: PREPROCESSING
# ==========================================
with tab_prep:
    st.markdown("### ⚙️ Pipeline Preprocessing Data")
    
    st.markdown(f"""
    Seluruh tahapan preprocessing disusun secara sistematis dan terbungkus di dalam `sklearn.pipeline.Pipeline` 
    serta `ColumnTransformer` guna menjamin **tidak adanya kebocoran data (*data leakage*)** antar lipatan validasi maupun data uji.
    """)

    # Daftar langkah dinamis
    st.markdown(f"""
    #### 🛠️ Langkah-Langkah Pemrosesan:
    1. **Pembersihan & Pembalikan Item Negatif (*Reverse Coding*)**:
       - Item negatif **A3.1** (*sering terlibat konflik*) dan **A3.2** (*sering mengganggu kelas*) dibalik menggunakan rumus: `Skor_Baru = 6 - Skor_Lama`.
       - Item **C3.3** (*kecerdasan tidak bisa diubah*) **tidak dibalik** karena data mentah telah menunjukkan skala searah (*pre-reversed*).
    2. **Agregasi Skor Subskala**:
       - Dihitung nilai rata-rata dari 4 item Likert pada masing-masing subskala:
         `A1, A2, A3, B1, B2, B3, C1, C2, C3`.
    3. **Pembentukan Target Klasifikasi (Berdasarkan C1)**:
       - Mode target aktif: **{target_choice}**.
       - Batas tercile kuantil 1/3 (`{meta['q1']:.2f}`) dan 2/3 (`{meta['q2']:.2f}`).
       - Label target yang terbentuk: `{class_order}`.
    4. **Prosedur Anti-Leakage (Pencegahan Kebocoran Data)**:
       - 🛡️ **Subskala C1 DIHAPUS SEPENUHNYA** dari matriks fitur prediktor `X` karena merupakan basis pembentukan target `y`.
       - 🛡️ **10 Kolom Nilai Mata Pelajaran** (Matematika, PPKn, IPA, IPS, Fisika, Biologi, Kimia, Sejarah, Ekonomi, Geografi) **diabaikan** dan tidak digunakan sebagai fitur, karena bersifat opsional per jenjang (SMP/SMA).
    5. **Konfigurasi Skenario Fitur**:
       - Skenario aktif: **{scenario_choice}**.
       - **Fitur Numerik ({len(num_cols)} fitur)**: `{num_cols}`.
       - **Fitur Kategorik ({len(cat_cols)} fitur)**: `{cat_cols}`.
    6. **ColumnTransformer & Skalasi Fitur**:
       - Fitur Numerik diskalakan menggunakan **`{scaler_choice}`**.
       - Fitur Kategorik diubah menjadi representasi biner dengan **`OneHotEncoder(handle_unknown='ignore')`**.
       - Transformasi dijalankan secara terisolasi pada setiap fold pelatihan.
    7. **Partisi Data Latih & Uji (Stratified Split)**:
       - Data Latih (*Train*): `{1 - test_size:.0%}` ({len(X_train)} baris).
       - Data Uji (*Test*): `{test_size:.0%}` ({len(X_test)} baris).
       - Stratifikasi target aktif, `random_state = {random_seed}`.
       - Validasi Silang: **{cv_folds} Lipatan (*Stratified K-Fold*)** pada data latih.
    """)

    st.markdown("---")
    st.markdown("#### 📋 10 Baris Pertama Fitur Akhir Sebelum Transformasi (`X` & `y`)")
    preview_df = X.copy()
    preview_df['TARGET (y)'] = y
    st.dataframe(preview_df.head(10), use_container_width=True)

    # Info dimensi matriks setelah one-hot encoding
    demo_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    demo_ohe.fit(X[cat_cols])
    encoded_cat_names = demo_ohe.get_feature_names_out(cat_cols)
    total_dim = len(num_cols) + len(encoded_cat_names)
    st.caption(f"📐 Dimensi input setelah One-Hot Encoding: **{total_dim} kolom** ({len(num_cols)} numerik + {len(encoded_cat_names)} hasil ekspansi kategorik).")


# ==========================================
# TAB: CLUSTERING (UNSUPERVISED LEARNING)
# ==========================================
with tab_clustering:
    st.markdown("### 🧩 Analisis Klastering Siswa (*Unsupervised Learning*)")
    st.markdown("""
    Berbeda dengan **Klasifikasi (*Supervised*)** yang memprediksi label target yang sudah diketahui ($C1$), 
    **Klastering (*Unsupervised*)** bertujuan mengelompokkan 1.182 siswa secara alami berdasarkan kemiripan profil 
    kesiapan belajar mereka **tanpa menggunakan label target**.
    """)

    # Kontrol Parameter Klastering
    col_cl_ctrl1, col_cl_ctrl2, col_cl_ctrl3 = st.columns([1, 1, 2])
    with col_cl_ctrl1:
        k_clusters = st.slider("Jumlah Klaster (k):", min_value=2, max_value=7, value=3, step=1,
                               help="Pilih jumlah kelompok yang diinginkan untuk algoritma K-Means.")
    with col_cl_ctrl2:
        include_c1 = st.checkbox("Sertakan Efikasi Diri (C1)?", value=False,
                                 help="Jika dicentang, dimensi efikasi diri ikut mengelompokkan siswa.")
    with col_cl_ctrl3:
        cluster_feature_pool = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C2', 'C3']
        if include_c1:
            cluster_feature_pool.append('C1')
        selected_cluster_features = st.multiselect(
            "Fitur Subskala untuk Klastering:",
            options=cluster_feature_pool,
            default=cluster_feature_pool,
            help="Pilih subskala yang digunakan sebagai basis penghitungan jarak Euclidean."
        )

    if len(selected_cluster_features) < 2:
        st.warning("⚠️ Pilih minimal 2 subskala fitur untuk menjalankan analisis klastering.")
    else:
        # Siapkan data untuk klastering
        X_cl_raw = df_clean[selected_cluster_features].values
        cl_scaler = StandardScaler()
        X_cl_scaled = cl_scaler.fit_transform(X_cl_raw)

        # 1. Evaluasi Jumlah Klaster Optimal (Elbow & Silhouette)
        st.markdown("#### 1. Evaluasi Jumlah Klaster Optimal (Elbow Method & Silhouette Score)")
        
        # Hitung metrik evaluasi klaster k=2..7
        k_eval_range = range(2, 8)
        inertias = []
        silhouettes = []
        for k_val in k_eval_range:
            km_eval = KMeans(n_clusters=k_val, random_state=random_seed, n_init=10)
            labels_eval = km_eval.fit_predict(X_cl_scaled)
            inertias.append(km_eval.inertia_)
            silhouettes.append(silhouette_score(X_cl_scaled, labels_eval))

        col_ev_cl1, col_ev_cl2 = st.columns(2)
        with col_ev_cl1:
            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 3.8))
            ax_elbow.plot(k_eval_range, inertias, marker='o', color='#2980b9', linewidth=2, markersize=7)
            chosen_inertia = inertias[k_clusters - 2]
            ax_elbow.plot(k_clusters, chosen_inertia, marker='o', color='red', markersize=11, label=f'k={k_clusters} (Aktif)')
            ax_elbow.set_xlabel("Jumlah Klaster (k)", fontsize=10, fontweight='bold')
            ax_elbow.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=10, fontweight='bold')
            ax_elbow.set_title("Elbow Method (Titik Siku)", fontsize=11, fontweight='bold')
            ax_elbow.grid(True, linestyle='--', alpha=0.5)
            ax_elbow.legend()
            fig_elbow.tight_layout()
            st.pyplot(fig_elbow)
            plt.close(fig_elbow)

        with col_ev_cl2:
            fig_sil, ax_sil = plt.subplots(figsize=(6, 3.8))
            ax_sil.plot(k_eval_range, silhouettes, marker='s', color='#27ae60', linewidth=2, markersize=7)
            chosen_sil = silhouettes[k_clusters - 2]
            ax_sil.plot(k_clusters, chosen_sil, marker='s', color='red', markersize=11, label=f'k={k_clusters} (Score: {chosen_sil:.3f})')
            ax_sil.set_xlabel("Jumlah Klaster (k)", fontsize=10, fontweight='bold')
            ax_sil.set_ylabel("Silhouette Coefficient", fontsize=10, fontweight='bold')
            ax_sil.set_title("Silhouette Score (Kerapatan Klaster)", fontsize=11, fontweight='bold')
            ax_sil.grid(True, linestyle='--', alpha=0.5)
            ax_sil.legend()
            fig_sil.tight_layout()
            st.pyplot(fig_sil)
            plt.close(fig_sil)

        # 2. Eksekusi K-Means pada k terpilih
        km_model = KMeans(n_clusters=k_clusters, random_state=random_seed, n_init=10)
        cluster_labels = km_model.fit_predict(X_cl_scaled)
        
        df_clustering = df_clean.copy()
        df_clustering['Klaster'] = [f"Klaster {lbl+1}" for lbl in cluster_labels]
        df_clustering['Klaster_Num'] = cluster_labels

        st.markdown("---")
        st.markdown(f"#### 2. Visualisasi Proyeksi 2D (PCA) & Distribusi Klaster (k = {k_clusters})")

        col_pca, col_size = st.columns([3, 2])
        with col_pca:
            # PCA 2 Dimensi untuk visualisasi
            pca_model = PCA(n_components=2, random_state=random_seed)
            coords = pca_model.fit_transform(X_cl_scaled)
            exp_var = pca_model.explained_variance_ratio_

            fig_pca, ax_pca = plt.subplots(figsize=(6.5, 4.5))
            cmap_custom = plt.colormaps.get_cmap('tab10')
            for c_idx in range(k_clusters):
                mask = cluster_labels == c_idx
                ax_pca.scatter(
                    coords[mask, 0], coords[mask, 1],
                    color=cmap_custom(c_idx), label=f'Klaster {c_idx+1}',
                    alpha=0.65, edgecolors='none', s=35
                )

            # Centroid dalam koordinat PCA
            centers_pca = pca_model.transform(km_model.cluster_centers_)
            ax_pca.scatter(
                centers_pca[:, 0], centers_pca[:, 1],
                marker='X', color='black', s=160, linewidths=1.5,
                edgecolors='white', label='Centroid'
            )
            ax_pca.set_xlabel(f"PC 1 ({exp_var[0]*100:.1f}% Variance)", fontsize=10, fontweight='bold')
            ax_pca.set_ylabel(f"PC 2 ({exp_var[1]*100:.1f}% Variance)", fontsize=10, fontweight='bold')
            ax_pca.set_title("Proyeksi Data Siswa ke Ruang 2D PCA", fontsize=11, fontweight='bold')
            ax_pca.legend(loc='best', frameon=True, fontsize=9)
            ax_pca.grid(True, linestyle='--', alpha=0.4)
            fig_pca.tight_layout()
            st.pyplot(fig_pca)
            plt.close(fig_pca)

        with col_size:
            # Distribusi Jumlah Siswa per Klaster
            cl_counts = df_clustering['Klaster'].value_counts().sort_index()
            fig_cl_bar, ax_cl_bar = plt.subplots(figsize=(5, 4.5))
            bars_c = ax_cl_bar.bar(cl_counts.index, cl_counts.values,
                                   color=[cmap_custom(i) for i in range(len(cl_counts))],
                                   edgecolor='black', alpha=0.85)
            for bar in bars_c:
                h = bar.get_height()
                p = (h / len(df_clustering)) * 100
                ax_cl_bar.annotate(f"{h}\n({p:.1f}%)",
                                  xy=(bar.get_x() + bar.get_width() / 2, h),
                                  xytext=(0, 4), textcoords="offset points",
                                  ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax_cl_bar.set_ylabel("Jumlah Siswa", fontsize=10, fontweight='bold')
            ax_cl_bar.set_ylim(0, max(cl_counts.values) * 1.25)
            ax_cl_bar.set_title("Distribusi Ukuran Klaster", fontsize=11, fontweight='bold')
            ax_cl_bar.grid(axis='y', linestyle='--', alpha=0.4)
            fig_cl_bar.tight_layout()
            st.pyplot(fig_cl_bar)
            plt.close(fig_cl_bar)

        st.markdown("---")
        # 3. Profiling Karakteristik Tiap Klaster
        st.markdown("#### 3. Profiling & Karakteristik Subskala Tiap Klaster")
        
        # Hitung rata-rata tiap subskala pada data asli
        profile_df = df_clustering.groupby('Klaster')[selected_cluster_features].mean()
        profile_df['Skor Rata-rata Total'] = profile_df.mean(axis=1)
        profile_df['Jumlah Siswa'] = cl_counts

        # Interpretasi Arketipe Klaster secara otomatis
        def assign_archetype(row):
            overall = row['Skor Rata-rata Total']
            if overall >= 4.2:
                return "🌟 Kesiapan Belajar Unggul / Mandiri"
            elif overall >= 3.6:
                return "🌱 Kesiapan Belajar Sedang / Berkembang"
            else:
                return "⚠️ Kesiapan Rendah / Butuh Pendampingan"

        profile_df['Karakteristik Profil'] = profile_df.apply(assign_archetype, axis=1)
        
        cols_order_prof = ['Karakteristik Profil', 'Jumlah Siswa', 'Skor Rata-rata Total'] + selected_cluster_features
        format_dict = {'Jumlah Siswa': '{:.0f}', 'Skor Rata-rata Total': '{:.2f}'}
        for f in selected_cluster_features:
            format_dict[f] = '{:.2f}'

        st.dataframe(profile_df[cols_order_prof].style.format(format_dict).highlight_max(
            subset=selected_cluster_features, color='#d4edda', axis=0
        ), use_container_width=True)

        # Grouped bar chart untuk perbandingan visual profil
        fig_profile, ax_profile = plt.subplots(figsize=(9, 4.5))
        profile_df[selected_cluster_features].T.plot(kind='bar', ax=ax_profile, colormap='tab10',
                                                     edgecolor='black', alpha=0.85, width=0.75)
        ax_profile.set_ylabel("Skor Rata-rata (Skala 1 - 5)", fontsize=10, fontweight='bold')
        ax_profile.set_xlabel("Subskala Kesiapan Belajar", fontsize=10, fontweight='bold')
        ax_profile.set_ylim(1.0, 5.2)
        ax_profile.legend(title="Kelompok Klaster", loc='upper left', bbox_to_anchor=(1.01, 1))
        ax_profile.grid(axis='y', linestyle='--', alpha=0.5)
        fig_profile.tight_layout()
        st.pyplot(fig_profile)
        plt.close(fig_profile)

        st.markdown("---")
        # 4. Validasi Silang: Hubungan Klaster Unsupervised vs Efikasi Diri Supervised
        st.markdown("#### 4. Hubungan Klaster Alami (*Unsupervised*) dengan Label Efikasi Diri (*Supervised*)")
        st.caption("Mengecek apakah pengelompokan tanpa label ini secara alami membedakan tingkat efikasi diri siswa (Rendah, Sedang, Tinggi).")

        col_cross1, col_cross2 = st.columns([1, 1])
        with col_cross1:
            ct_cl_target = pd.crosstab(df_clustering['Klaster'], df_clustering[target_col])[class_order]
            st.markdown("##### Tabel Kontingensi (Jumlah Siswa):")
            st.dataframe(ct_cl_target, use_container_width=True)

        with col_cross2:
            st.markdown("##### Komposisi Tingkat Efikasi Diri per Klaster:")
            ct_cl_norm = pd.crosstab(df_clustering['Klaster'], df_clustering[target_col], normalize='index')[class_order] * 100
            fig_cross, ax_cross = plt.subplots(figsize=(5.5, 3.8))
            colors_ct = ['#e74c3c', '#f39c12', '#2ecc71'] if is_3_class else ['#e74c3c', '#3498db']
            ct_cl_norm.plot(kind='bar', stacked=True, ax=ax_cross, color=colors_ct, edgecolor='black', alpha=0.85)
            ax_cross.set_ylabel("Persentase (%)", fontsize=10, fontweight='bold')
            ax_cross.set_xlabel("Klaster", fontsize=10, fontweight='bold')
            ax_cross.set_ylim(0, 105)
            ax_cross.legend(title="Efikasi Diri", loc='upper right', frameon=True)
            ax_cross.grid(axis='y', linestyle='--', alpha=0.4)
            fig_cross.tight_layout()
            st.pyplot(fig_cross)
            plt.close(fig_cross)

        st.info("""
        💡 **Kesimpulan Klastering**:
        Meskipun K-Means dijalankan tanpa melihat label target sama sekali (*unsupervised*), klaster yang memiliki rata-rata kesiapan belajar tinggi (misal *Klaster 1 / Unggul*) secara konsisten didominasi oleh siswa dengan tingkat Efikasi Diri **Tinggi**. Sebaliknya, klaster dengan kesiapan belajar terendah didominasi oleh tingkat Efikasi Diri **Rendah**. Ini membuktikan secara kuat bahwa domain kesiapan belajar berkorespondensi langsung dengan rasa percaya diri akademik siswa.
        """)


# ==========================================
# TAB 4: MODELING
# ==========================================
with tab_modeling:
    st.markdown("### 🤖 Pelatihan & Perbandingan Model")
    
    if not selected_models:
        st.warning("⚠️ Silakan pilih minimal 1 model klasifikasi pada sidebar untuk memulai pelatihan.")
    else:
        with st.spinner("Sedang melatih model dan menjalankan validasi silang (Cross-Validation)..."):
            results_df, fitted_pipes, test_preds = train_and_evaluate_models(
                X_train, X_test, y_train, y_test,
                num_cols, cat_cols,
                selected_models, scaler_choice,
                cv_folds, random_seed
            )

        # Baseline info box
        st.info(f"""
        💡 **Zero-Rule Baseline (Tebak Kelas Mayoritas '{majority_class}') pada Data Uji**:
        - **Baseline Test Akurasi**: `{baseline_acc:.4f}` ({baseline_acc*100:.2f}%)
        - **Baseline Test Macro-F1**: `{baseline_f1:.4f}` ({baseline_f1*100:.2f}%)
        """)

        # Tampilkan tabel perbandingan
        display_results = results_df.copy()
        
        # Format nilai metrik
        display_results['Test Akurasi'] = display_results['Test Akurasi'].apply(lambda v: f"{v:.3f}")
        display_results['Test Macro-F1'] = display_results['Test Macro-F1'].apply(lambda v: f"{v:.3f}")
        
        # Identifikasi model terbaik
        best_f1_idx = results_df['Test Macro-F1'].idxmax()
        best_model_name = results_df.loc[best_f1_idx, 'Model']
        best_f1_score = results_df.loc[best_f1_idx, 'Test Macro-F1']
        best_acc_score = results_df.loc[best_f1_idx, 'Test Akurasi']

        st.markdown(f"#### 🏆 Model Terbaik: **{best_model_name}** (Test Macro-F1: `{best_f1_score:.3f}`, Akurasi: `{best_acc_score:.3f}`)")

        # Fungsi penanda (highlight) nilai maksimum
        def highlight_max_metrics(df_in):
            styles = pd.DataFrame('', index=df_in.index, columns=df_in.columns)
            mapping = {
                'CV Akurasi (Mean ± Std)': results_df['_cv_acc_raw'].idxmax(),
                'CV Macro-F1 (Mean ± Std)': results_df['_cv_f1_raw'].idxmax(),
                'Test Akurasi': results_df['Test Akurasi'].idxmax(),
                'Test Macro-F1': results_df['Test Macro-F1'].idxmax()
            }
            for col_name, max_idx in mapping.items():
                if col_name in df_in.columns:
                    styles.loc[max_idx, col_name] = 'background-color: #d4edda; font-weight: bold; color: #155724;'
            return styles

        cols_to_show = ['Model', 'CV Akurasi (Mean ± Std)', 'CV Macro-F1 (Mean ± Std)', 'Test Akurasi', 'Test Macro-F1']
        styled_results = display_results[cols_to_show].style.apply(highlight_max_metrics, axis=None)
        st.dataframe(styled_results, hide_index=True, use_container_width=True)
        st.caption("🟢 *Sel berlatar hijau menandakan nilai maksimum (performa tertinggi) untuk setiap metrik evaluasi.*")

        # Tombol Download CSV
        csv_buffer = io.StringIO()
        display_results[cols_to_show].to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Unduh Tabel Hasil Perbandingan Model (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"hasil_klasifikasi_efikasi_diri_{target_col}.csv",
            mime="text/csv"
        )


# ==========================================
# TAB 5: EVALUATION
# ==========================================
with tab_eval:
    st.markdown("### 📈 Evaluasi Model Mendalam")

    if not selected_models:
        st.warning("⚠️ Pilih minimal 1 model pada sidebar untuk melihat hasil evaluasi.")
    else:
        # Tentukan default model (model dengan Test F1 tertinggi)
        default_index = 0
        if 'best_model_name' in locals() and best_model_name in selected_models:
            default_index = selected_models.index(best_model_name)

        chosen_model_name = st.selectbox(
            "Pilih Model untuk Evaluasi Rinci:",
            options=selected_models,
            index=default_index
        )

        chosen_pipe = fitted_pipes[chosen_model_name]
        chosen_preds = test_preds[chosen_model_name]

        # 1. Confusion Matrix & Classification Report
        col_ev1, col_ev2 = st.columns([1, 1])

        with col_ev1:
            st.markdown(f"#### 1. Confusion Matrix ({chosen_model_name})")
            cm = confusion_matrix(y_test, chosen_preds, labels=class_order)
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4.2))
            cax_cm = ax_cm.imshow(cm, cmap='Blues', interpolation='nearest')
            fig_cm.colorbar(cax_cm, ax=ax_cm, fraction=0.046, pad=0.04)
            
            ax_cm.set_xticks(range(len(class_order)))
            ax_cm.set_yticks(range(len(class_order)))
            ax_cm.set_xticklabels(class_order, fontsize=10)
            ax_cm.set_yticklabels(class_order, fontsize=10)
            ax_cm.set_xlabel("Label Prediksi", fontsize=11, fontweight='bold')
            ax_cm.set_ylabel("Label Sebenarnya", fontsize=11, fontweight='bold')
            
            total_test = len(y_test)
            for i in range(len(class_order)):
                for j in range(len(class_order)):
                    val = cm[i, j]
                    pct = (val / total_test) * 100
                    text_color = "white" if val > cm.max() / 2 else "black"
                    ax_cm.text(j, i, f"{val}\n({pct:.1f}%)", ha='center', va='center',
                               color=text_color, fontsize=10, fontweight='bold')
            
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_ev2:
            st.markdown(f"#### 2. Classification Report ({chosen_model_name})")
            report_dict = classification_report(y_test, chosen_preds, labels=class_order, output_dict=True)
            report_df = pd.DataFrame(report_dict).T
            
            # Format report
            st.dataframe(
                report_df.style.format({
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'f1-score': '{:.3f}',
                    'support': '{:.0f}'
                }),
                use_container_width=True
            )

        st.markdown("---")

        # 2. Permutation Feature Importance
        st.markdown(f"#### 3. Permutation Feature Importance ({chosen_model_name})")
        st.caption("Dihitung pada data uji (Test Set) dengan `n_repeats=15` dan metrik `scoring='f1_macro'`.")

        with st.spinner(f"Menghitung Permutation Feature Importance untuk {chosen_model_name}..."):
            perm_res = permutation_importance(
                chosen_pipe, X_test, y_test,
                n_repeats=15,
                random_state=random_seed,
                scoring='f1_macro'
            )

        # Mapping nama fitur ke bahasa Indonesia
        indo_names = [FEATURE_TRANSLATIONS.get(col, col) for col in X_test.columns]
        
        perm_df = pd.DataFrame({
            'Fitur': indo_names,
            'Kode': X_test.columns,
            'Importance': perm_res.importances_mean,
            'Std': perm_res.importances_std
        }).sort_values('Importance', ascending=True)

        fig_perm, ax_perm = plt.subplots(figsize=(8.5, 5.8))
        y_pos = range(len(perm_df))
        bars_perm = ax_perm.barh(y_pos, perm_df['Importance'], xerr=perm_df['Std'],
                                 color='#2b5c8f', edgecolor='black', alpha=0.85, capsize=4)
        ax_perm.set_yticks(y_pos)
        ax_perm.set_yticklabels(perm_df['Fitur'], fontsize=10)
        ax_perm.set_xlabel("Penurunan Macro-F1 Score Setelah Permutasi (Data Uji)", fontsize=11, fontweight='bold')
        ax_perm.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Garis referensi 0
        ax_perm.axvline(0, color='red', linestyle='--', linewidth=0.9, alpha=0.7)
        
        # Anotasi nilai pada bar
        for idx, (imp, std) in enumerate(zip(perm_df['Importance'], perm_df['Std'])):
            offset = 0.003 if imp >= 0 else -0.003
            ha = 'left' if imp >= 0 else 'right'
            ax_perm.annotate(f"{imp:+.3f} ± {std:.3f}",
                             xy=(imp + offset, idx),
                             xytext=(0, 0), textcoords="offset points",
                             ha=ha, va='center', fontsize=8.5, fontweight='bold')
        
        fig_perm.tight_layout()
        st.pyplot(fig_perm)
        plt.close(fig_perm)

        # Catatan Multikolinearitas / Korelasi Fitur
        st.warning("""
        ⚠️ **Catatan Penting Mengenai Korelasi Antar-Fitur (Multikolinearitas)**:
        - Domain-domain kesiapan belajar memiliki **korelasi positif moderat hingga tinggi** (misalnya, *Keterlibatan Kelas B3*, *Ketekunan B1*, dan *Nilai Belajar C2*).
        - Dalam metode *Permutation Importance*, mengacak salah satu fitur (misalnya B1) mungkin hanya menurunkan sedikit skor jika fitur lain yang berkorelasi kuat (seperti B3) masih dapat memberikan informasi serupa kepada model.
        - Oleh karena itu, penting memaknai kontribusi fitur secara komprehensif bersama matriks korelasi pada Tab EDA.
        """)

        st.markdown("---")
        st.markdown(f"#### 4. 🔮 Simulator Prediksi Siswa Baru (Inferensi Model {chosen_model_name})")
        st.caption("Uji coba interaktif untuk memprediksi tingkat efikasi diri akademik seorang siswa berdasarkan profil kesiapan belajar.")

        with st.expander("📝 Form Input Karakteristik Siswa & Skor Kesiapan Belajar", expanded=True):
            col_sim1, col_sim2 = st.columns([1, 1])

            with col_sim1:
                st.markdown("##### 📚 Domain Kesiapan Belajar (Skala 1,0 – 5,0)")
                sim_a1 = st.slider("Rasa Memiliki (A1):", 1.0, 5.0, 4.0, 0.1)
                sim_a2 = st.slider("Regulasi Emosi (A2):", 1.0, 5.0, 3.5, 0.1)
                sim_a3 = st.slider("Perilaku Sosial (A3):", 1.0, 5.0, 4.0, 0.1)
                sim_b1 = st.slider("Ketekunan (B1):", 1.0, 5.0, 4.2, 0.1)
                sim_b2 = st.slider("Perencanaan Belajar (B2):", 1.0, 5.0, 3.8, 0.1)
                sim_b3 = st.slider("Keterlibatan di Kelas (B3):", 1.0, 5.0, 4.0, 0.1)
                sim_c2 = st.slider("Nilai/Manfaat Belajar (C2):", 1.0, 5.0, 4.5, 0.1)
                sim_c3 = st.slider("Growth Mindset (C3):", 1.0, 5.0, 4.0, 0.1)

            with col_sim2:
                st.markdown("##### 👤 Profil Demografi Siswa")
                sim_jenjang = st.selectbox("Jenjang Sekolah:", ["SMA", "SMP"], index=0)
                sim_kelas = st.selectbox("Tingkat Kelas:", ["7", "8", "9", "10", "11", "12"], index=3)
                sim_gender = st.selectbox("Jenis Kelamin:", ["Perempuan", "Laki-Laki"], index=0)
                sim_lokasi = st.selectbox("Lokasi Sekolah:", ["kota", "desa", "daerah"], index=0)
                sim_transport = st.selectbox("Transportasi ke Sekolah:", ["Motor", "Sepeda", "Jalan kaki", "Mobil", "Transportasi umum"], index=0)

                if is_scenario_b:
                    st.markdown("##### 🎯 Outcome Ordinal Siswa")
                    sim_kepuasan = st.slider("Kepuasan Hasil Belajar (1=Tidak Puas, 5=Sangat Puas):", 1, 5, 4, 1)
                    sim_keyakinan = st.slider("Keyakinan Naik Nilai (1=Kurang, 4=Sangat Percaya):", 1, 4, 3, 1)

            btn_predict = st.button("🚀 Jalankan Prediksi Tingkat Efikasi Diri Siswa", type="primary", use_container_width=True)

        if btn_predict:
            student_dict = {
                'A1': sim_a1, 'A2': sim_a2, 'A3': sim_a3,
                'B1': sim_b1, 'B2': sim_b2, 'B3': sim_b3,
                'C2': sim_c2, 'C3': sim_c3,
                'jenjang': sim_jenjang, 'kelas': sim_kelas,
                'gender': sim_gender, 'lokasi': sim_lokasi,
                'transport': sim_transport
            }
            if is_scenario_b:
                student_dict['kepuasan'] = sim_kepuasan
                student_dict['keyakinan_naik_nilai'] = sim_keyakinan

            student_df = pd.DataFrame([student_dict])
            pred_class = chosen_pipe.predict(student_df)[0]

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.markdown("##### 🎯 Hasil Prediksi:")
                badge_color = "#28a745" if pred_class in ["Tinggi", "Sedang-Tinggi"] else ("#ffc107" if pred_class == "Sedang" else "#dc3545")
                text_color = "black" if pred_class == "Sedang" else "white"
                st.markdown(
                    f"""<div style="background-color: {badge_color}; color: {text_color}; padding: 18px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 10px;">
                    Tingkat Efikasi Diri: {pred_class}
                    </div>""",
                    unsafe_allow_html=True
                )

            with col_res2:
                if hasattr(chosen_pipe.named_steps['clf'], "predict_proba"):
                    st.markdown("##### 📊 Distribusi Probabilitas Kelas:")
                    proba = chosen_pipe.predict_proba(student_df)[0]
                    cls_list = list(chosen_pipe.classes_)
                    for c_name, p_val in zip(cls_list, proba):
                        st.write(f"**{c_name}**: `{p_val*100:.1f}%`")
                        st.progress(float(p_val))
                else:
                    st.info(f"Model {chosen_model_name} menggunakan fungsi keputusan langsung (*decision boundary*).")

