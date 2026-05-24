import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from collections import Counter
import warnings

# Mengabaikan pesan peringatan agar tampilan tetap bersih
warnings.filterwarnings("ignore")

# =====================================================================
# LANGKAH 1: PENGATURAN AWAL & TEMA
# =====================================================================
# Mengatur judul tab browser dan layout menjadi lebar (wide)
st.set_page_config(page_title="E-Commerce Analytics", page_icon="🛍️", layout="wide")

# Mengatur palet warna agar seragam dan indah dipandang
PALETTE    = ["#38BDF8", "#818CF8", "#34D399", "#FB923C", "#F472B6", "#FACC15", "#A78BFA"]
GRID_COLOR = "#334155"
ACCENT     = "#38BDF8"
WARN       = "#FACC15"
POS        = "#34D399"

# Fungsi untuk menerapkan tema pada grafik Plotly
def terapkan_tema(fig, height=380):
    fig.update_layout(
        font=dict(family="sans-serif", size=12),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bordercolor=GRID_COLOR, borderwidth=1),
        hoverlabel=dict(bordercolor=ACCENT),
        height=height
    )
    return fig

# =====================================================================
# LANGKAH 2: FUNGSI PERAMALAN (TRIPLE EXPONENTIAL SMOOTHING)
# =====================================================================
def hitung_double_exponential(data_series, alpha, beta, langkah_kedepan):
    n = len(data_series)
    levels, trends = np.zeros(n), np.zeros(n)
    levels[0] = data_series[0]
    trends[0] = 0 if n <= 1 else (data_series[1] - data_series[0])
    for t in range(1, n):
        levels[t] = alpha * data_series[t] + (1 - alpha) * (levels[t-1] + trends[t-1])
        trends[t] = beta * (levels[t] - levels[t-1]) + (1 - beta) * trends[t-1]
    prediksi = [levels[-1] + m * trends[-1] for m in range(1, langkah_kedepan + 1)]
    return levels + trends, prediksi

def hitung_holt_winters(data_series, alpha, beta, gamma, L, langkah_kedepan):
    n = len(data_series)
    if n < 2 * L or L <= 1:
        return hitung_double_exponential(data_series, alpha, beta, langkah_kedepan)

    level = np.mean(data_series[:L])
    trend = np.mean(data_series[L:2*L] - data_series[:L]) / L
    seasonals = list(data_series[:L] - level)

    levels, trends, smoothed = np.zeros(n), np.zeros(n), np.zeros(n)
    levels[0], trends[0], smoothed[0] = level, trend, data_series[0]

    for t in range(1, n):
        val = data_series[t]
        s_prev = seasonals[t - L] if t >= L else seasonals[t % L]
        levels[t] = alpha * (val - s_prev) + (1 - alpha) * (levels[t-1] + trends[t-1])
        trends[t] = beta * (levels[t] - levels[t-1]) + (1 - beta) * trends[t-1]

        if t >= L: seasonals.append(gamma * (val - levels[t]) + (1 - gamma) * s_prev)
        else: seasonals[t] = gamma * (val - levels[t]) + (1 - gamma) * s_prev
        smoothed[t] = levels[t] + s_prev

    prediksi = [levels[-1] + m * trends[-1] + seasonals[n - L + (m - 1) % L] for m in range(1, langkah_kedepan + 1)]
    return smoothed, prediksi

def optimasi_holt_winters(data_series, L, langkah_kedepan):
    best_mse, best_params = float("inf"), (0.2, 0.1, 0.2)
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for b in [0.05, 0.1, 0.2]:
            for g in [0.1, 0.3, 0.5]:
                smoothed, _ = hitung_holt_winters(data_series, a, b, g, L, langkah_kedepan)
                mse = np.mean((data_series - smoothed) ** 2)
                if mse < best_mse:
                    best_mse, best_params = mse, (a, b, g)
    return best_params

# =====================================================================
# LANGKAH 3: MEMBACA DATA DARI GOOGLE SHEETS & PREPROCESSING
# =====================================================================
st.title("🛍️ E-Commerce Customer Behavior Analytics")
st.caption("Workshop Data Analitik | Live Customer Performance Monitor")
st.divider()

@st.cache_data(ttl=60)
def ambil_data():
    sheet_id = "1RFF-5XhqHbtHqXmzTVjgTmGekd763wyZsAEVhMJ-PAc"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    df = pd.read_csv(url)

    # ---------------------------------------------------------
    # A. Imputasi Missing Value
    # ---------------------------------------------------------
    for col in ["Customer_Rating", "Age", "Session_Duration_Minutes"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in ["City", "Device_Type", "Payment_Method", "Gender", "Product_Category"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ---------------------------------------------------------
    # B. Penghapusan Outlier (Metode IQR)
    # ---------------------------------------------------------
    kolom_outlier = ["Total_Amount", "Quantity"]
    for col in kolom_outlier:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        df = df[(df[col] >= batas_bawah) & (df[col] <= batas_atas)]

    # ---------------------------------------------------------
    # Persiapan Data Dasar (Wajib untuk grafik Streamlit)
    # ---------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Bulan"] = df["Date"].dt.to_period("M").astype(str)
    df["Tahun"] = df["Date"].dt.year
    df["Kelompok_Umur"] = pd.cut(
        df["Age"],
        bins=[0, 18, 25, 35, 45, 55, 100],
        labels=["≤18", "19-25", "26-35", "36-45", "46-55", "56+"],
    )
    df["Ada_Diskon"] = df["Discount_Amount"] > 0

    return df

try:
    data_awal = ambil_data()

    # =====================================================================
    # LANGKAH 4: MEMBUAT MENU FILTER DI SIDEBAR (SAMPING)
    # =====================================================================


    # =====================================================================
    # LANGKAH 5: MENAMPILKAN ANGKA RINGKASAN (KPI METRICS)
    # =====================================================================


    # =====================================================================
    # LANGKAH 6: MEMBUAT HALAMAN TABS
    # =====================================================================


    # ---------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # TAB 3: STRATEGI PRODUK & HARGA (DENGAN FORECASTING)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # TAB 4: DATA CENTER & UNDUH
    # ---------------------------------------------------------

except Exception as e:
    st.error("Gagal memuat data. Pastikan koneksi internet aktif untuk mengakses Google Sheets.")
    st.exception(e)

st.divider()
st.caption("📊 E-Commerce Dashboard | Diperbarui untuk Pemula dengan Sumber Data Google Sheets")
