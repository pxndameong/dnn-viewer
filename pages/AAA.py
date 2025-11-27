# pages/Analytical Table.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tabel Data Komparatif",
    page_icon="📊",
    layout="wide"
)
os.environ["STREAMLIT_WATCHDOG"] = "false"

# URL dasar untuk data prediksi
base_url_pred = "data/10k_epoch/pred"

# URL dasar untuk data padanan
base_url_padanan = "data/10k_epoch/padanan"

# Info dataset yang akan dibandingkan
dataset_info = {
    "0 Variabel": {"folder": "0_var", "prefix": "DNN_all_data_0var"},
    #"10 Variabel": {"folder": "10_var", "prefix": "all_data_10var"},
    #"51 Variabel": {"folder": "51_var", "prefix": "all_data_51var"},
}

bulan_dict = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Data stasiun yang baru
station_data = [
    {"name": "Stasiun 218 (Lat: -8, Lon: 113.5)", "lat": -8, "lon": 113.5, "index": 218},
    {"name": "Stasiun 294 (Lat: -7.5, Lon: 110)", "lat": -7.5, "lon": 110, "index": 294},
    {"name": "Stasiun 329 (Lat: -7.25, Lon: 107.5)", "lat": -7.25, "lon": 107.5, "index": 329},
    {"name": "Stasiun 333 (Lat: -7.25, Lon: 108.5)", "lat": -7.25, "lon": 108.5, "index": 333},
    {"name": "Stasiun 384 (Lat: -7, Lon: 110)", "lat": -7, "lon": 110, "index": 384},
    {"name": "Stasiun 393 (Lat: -7, Lon: 112.25)", "lat": -7, "lon": 112.25, "index": 393},
    {"name": "Stasiun 505 (Lat: -6.25, Lon: 106.5)", "lat": -6.25, "lon": 106.5, "index": 505},
]

# Modifikasi: Tambahkan opsi "Rata-Rata Seluruh Stasiun"
station_names = [s["name"] for s in station_data]
station_names.insert(0, "Rata-Rata Seluruh Stasiun")

# Precompute station coords set for fast filtering (tuples of floats)
station_coords = {(float(s["lat"]), float(s["lon"])) for s in station_data}

@st.cache_data
def load_data(dataset_name: str, tahun: int):
    folder = dataset_info[dataset_name]["folder"]
    prefix = dataset_info[dataset_name]["prefix"]
    url = f"{base_url_pred}/{folder}/{prefix}_{tahun}.parquet"
    try:
        df = pd.read_parquet(url, engine="pyarrow")
    except Exception as e:
        # st.error(f"❌ Gagal baca file: {url}\nError: {e}") # Nonaktifkan Error agar tidak terlalu banyak pesan
        return pd.DataFrame()
    df = df.convert_dtypes()
    # Normalize latitude/longitude column names if needed
    if 'lat' in df.columns and 'latitude' not in df.columns:
        df = df.rename(columns={'lat': 'latitude'})
    if 'lon' in df.columns and 'longitude' not in df.columns:
        df = df.rename(columns={'lon': 'longitude'})
    # Ensure numeric types for lat/lon if present
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_padanan_data(tahun: int):
    """
    Fungsi untuk memuat data padanan.
    """
    url = f"{base_url_padanan}/CLEANED_PADANAN_{tahun}.parquet"
    
    try:
        df = pd.read_parquet(url, engine="pyarrow")
        if 'lon' in df.columns:
            df = df.rename(columns={'lon': 'longitude'})
        if 'lat' in df.columns:
            df = df.rename(columns={'lat': 'latitude'})
        if 'idx_new' in df.columns:
            df = df.rename(columns={'idx_new': 'idx'})
    except Exception as e:
        # st.warning(f"⚠️ Gagal membaca file padanan: {url}\nError: {e}") # Nonaktifkan Warning agar tidak terlalu banyak pesan
        return pd.DataFrame()
        
    required_cols = ['month', 'year', 'latitude', 'longitude', 'rainfall', 'idx']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Ensure latitude/longitude numeric if present
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
    return df[required_cols]

# --- FUNGSI BARU UNTUK MENGHITUNG METRIK ---
def calculate_metrics(df: pd.DataFrame, actual_col: str, pred_col: str):
    """
    Menghitung MAE, RMSE, dan R^2.
    Menggunakan kolom yang tersedia dan hanya baris tanpa NaN di kedua kolom.
    """
    # Pastikan hanya baris dengan nilai valid yang digunakan
    df_clean = df.dropna(subset=[actual_col, pred_col])
    
    if df_clean.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    actual = df_clean[actual_col].astype(float)
    pred = df_clean[pred_col].astype(float)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred - actual))

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((pred - actual)**2))

    # R^2 (Coefficient of Determination)
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum((actual - pred)**2)
    
    # Handle zero division for ss_total in case of constant actual values
    if ss_total == 0:
        r2 = 1.0 # Perfect score if actual is constant and prediction matches
    else:
        r2 = 1 - (ss_residual / ss_total)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_comparative_charts_monthly(tahun_start: int, bulan_start: int, tahun_end: int, bulan_end: int, selected_station_name: str):
    """
    Fungsi untuk menampilkan bar chart perbandingan curah hujan bulanan (prediksi vs ground truth) 
    dan di bawahnya, Scatter Plot MAE, RMSE, dan R^2 dalam rentang waktu kustom.

    Implementasi ALL(rata): ketika selected_station_name == "Rata-Rata Seluruh Stasiun",
    prediksi dan padanan difilter ke 7 koordinat stasiun sebelum dirata-ratakan.
    """
    st.markdown("---")
    
    # Format judul untuk rentang kustom
    start_date_str = f"{bulan_dict[bulan_start]} {tahun_start}"
    end_date_str = f"{bulan_dict[bulan_end]} {tahun_end}"
    date_range_str = f"{start_date_str} - {end_date_str}"
    
    st.subheader(f"Perbandingan Curah Hujan Bulanan dan Metrik ({date_range_str}) untuk {selected_station_name}")

    is_all_stations = selected_station_name == "Rata-Rata Seluruh Stasiun"
    
    all_data_for_plot = []
    all_combined_data = {} # Untuk menyimpan data mentah untuk metrik

    # Daftar tahun yang perlu dimuat
    years_to_load = list(range(tahun_start, tahun_end + 1))
    
    # Kumpulkan data Padanan (Ground Truth) dari semua tahun
    df_padanan_all = []
    for th in years_to_load:
        df_padanan_all.append(load_padanan_data(th))
    if len(df_padanan_all) == 0:
        st.error("❌ Tidak ada file padanan ditemukan untuk tahun yang dipilih.")
        return

    df_padanan_full = pd.concat(df_padanan_all, ignore_index=True)
    
    # Buat objek datetime untuk batas filter (dipakai nanti)
    start_date = pd.to_datetime(f"{tahun_start}-{bulan_start}-01")
    end_date = pd.to_datetime(f"{tahun_end}-{bulan_end}-01")
    
    # 1. Filter/Rata-rata Ground Truth
    if is_all_stations:
        # Filter padanan hanya untuk 7 stasiun lalu hitung rata-rata per bulan
        # Pastikan kolom latitude/longitude numeric
        df_padanan_stations = df_padanan_full.copy()
        if {'latitude', 'longitude'}.issubset(df_padanan_stations.columns):
            # Use tuple matching with float to avoid dtype mismatch
            df_padanan_stations['coord_tuple'] = list(zip(df_padanan_stations['latitude'].astype(float), df_padanan_stations['longitude'].astype(float)))
            df_padanan_stations = df_padanan_stations[df_padanan_stations['coord_tuple'].isin(station_coords)].copy()
            df_padanan_stations.drop(columns=['coord_tuple'], inplace=True)
        else:
            # fallback: if latitude/longitude not present, keep as empty
            df_padanan_stations = pd.DataFrame(columns=df_padanan_full.columns)
        
        if df_padanan_stations.empty:
            st.warning("⚠️ Ground Truth (padanan) untuk 7 stasiun tidak ditemukan dalam data padanan.")
            df_padanan_filtered = pd.DataFrame()
        else:
            # create date col, filter range, then groupby year-month mean
            df_padanan_stations['date'] = pd.to_datetime(df_padanan_stations[['year', 'month']].assign(day=1))
            df_padanan_stations = df_padanan_stations[(df_padanan_stations['date'] >= start_date) & (df_padanan_stations['date'] <= end_date)].copy()
            df_padanan_stations.drop(columns=['date'], inplace=True)
            df_padanan_filtered = df_padanan_stations.groupby(['year', 'month']).agg(rainfall=('rainfall', 'mean')).reset_index()
    else:
        station_info = next((s for s in station_data if s["name"] == selected_station_name), None)
        if not station_info:
            st.error("❌ Informasi stasiun tidak ditemukan.")
            return
        # Filter Ground Truth untuk stasiun yang dipilih
        df_padanan_filtered = df_padanan_full[
            (df_padanan_full['latitude'] == station_info['lat']) &
            (df_padanan_full['longitude'] == station_info['lon'])
        ].copy()
        if not df_padanan_filtered.empty:
            df_padanan_filtered['date'] = pd.to_datetime(df_padanan_filtered[['year', 'month']].assign(day=1))
            df_padanan_filtered = df_padanan_filtered[(df_padanan_filtered['date'] >= start_date) & (df_padanan_filtered['date'] <= end_date)].copy()
            df_padanan_filtered.drop(columns=['date'], inplace=True)

    df_padanan_station = df_padanan_filtered.copy() if isinstance(df_padanan_filtered, pd.DataFrame) else pd.DataFrame()

    # Rename kolom for plotting if available
    if not df_padanan_station.empty and 'rainfall' in df_padanan_station.columns:
        df_padanan_plot = df_padanan_station.rename(columns={'rainfall': 'Curah Hujan (mm)'})
        df_padanan_plot['Tipe Data'] = 'Ground Truth (Rainfall)'
        all_data_for_plot.append(df_padanan_plot[['year', 'month', 'Curah Hujan (mm)', 'Tipe Data']])
    else:
        # If padanan empty, warn (but continue to try pred)
        if df_padanan_station.empty:
            st.warning("⚠️ Ground Truth (Rainfall) tidak tersedia untuk stasiun/rata-rata ini di tahun yang dipilih.")
    
    # 2. Ambil data Prediksi (0, 10, 51 Variabel)
    for dataset_name in dataset_info.keys():
        df_pred_all = []
        for th in years_to_load:
            df_pred_all.append(load_data(dataset_name, th))
        if len(df_pred_all) == 0:
            df_pred_full = pd.DataFrame()
        else:
            df_pred_full = pd.concat(df_pred_all, ignore_index=True)
        
        # Normalize lat/lon names if necessary
        if 'lat' in df_pred_full.columns and 'latitude' not in df_pred_full.columns:
            df_pred_full = df_pred_full.rename(columns={'lat': 'latitude'})
        if 'lon' in df_pred_full.columns and 'longitude' not in df_pred_full.columns:
            df_pred_full = df_pred_full.rename(columns={'lon': 'longitude'})
        # Ensure numeric for lat/lon
        if 'latitude' in df_pred_full.columns:
            df_pred_full['latitude'] = pd.to_numeric(df_pred_full['latitude'], errors='coerce')
        if 'longitude' in df_pred_full.columns:
            df_pred_full['longitude'] = pd.to_numeric(df_pred_full['longitude'], errors='coerce')

        # Filter/Rata-rata Prediksi
        if is_all_stations:
            # Filter prediksi hanya untuk 7 stasiun (matching lat-lon) lalu rata-rata per bulan
            if {'latitude', 'longitude'}.issubset(df_pred_full.columns):
                df_pred_full['coord_tuple'] = list(zip(df_pred_full['latitude'].astype(float), df_pred_full['longitude'].astype(float)))
                df_pred_stations = df_pred_full[df_pred_full['coord_tuple'].isin(station_coords)].copy()
                df_pred_stations.drop(columns=['coord_tuple'], inplace=True)
            else:
                df_pred_stations = pd.DataFrame(columns=df_pred_full.columns)
            
            if df_pred_stations.empty:
                st.warning(f"⚠️ Prediksi ({dataset_name}) tidak memiliki grid yang cocok dengan 7 stasiun.")
                df_pred_filtered = pd.DataFrame()
            else:
                df_pred_stations['date'] = pd.to_datetime(df_pred_stations[['year', 'month']].assign(day=1))
                df_pred_stations = df_pred_stations[(df_pred_stations['date'] >= start_date) & (df_pred_stations['date'] <= end_date)].copy()
                df_pred_stations.drop(columns=['date'], inplace=True)
                df_pred_filtered = df_pred_stations.groupby(['year', 'month']).agg(ch_pred=('ch_pred', 'mean')).reset_index()
        else:
            # Filter Prediksi untuk stasiun yang dipilih (single station)
            station_info = next((s for s in station_data if s["name"] == selected_station_name), None)
            if station_info is None:
                df_pred_filtered = pd.DataFrame()
            else:
                df_pred_filtered = df_pred_full[
                    (df_pred_full['latitude'] == station_info['lat']) &
                    (df_pred_full['longitude'] == station_info['lon'])
                ].copy()
                if not df_pred_filtered.empty:
                    df_pred_filtered['date'] = pd.to_datetime(df_pred_filtered[['year', 'month']].assign(day=1))
                    df_pred_filtered = df_pred_filtered[(df_pred_filtered['date'] >= start_date) & (df_pred_filtered['date'] <= end_date)].copy()
                    df_pred_filtered.drop(columns=['date'], inplace=True)

        df_pred_station = df_pred_filtered.copy() if isinstance(df_pred_filtered, pd.DataFrame) else pd.DataFrame()

        # Gabungkan data Prediksi dan Aktual (Padanan) untuk perhitungan Metrik dan Scatter Plot
        df_merged_custom_range = pd.merge(
            df_pred_station[['year', 'month', 'ch_pred']],
            df_padanan_station[['year', 'month', 'rainfall']],
            on=['year', 'month'],
            how='inner'
        ).drop_duplicates(subset=['year', 'month'])

        all_combined_data[dataset_name] = df_merged_custom_range

        # Data untuk Bar Chart
        if not df_pred_station.empty and 'ch_pred' in df_pred_station.columns:
            df_pred_plot = df_pred_station.rename(columns={'ch_pred': 'Curah Hujan (mm)'})
            df_pred_plot['Tipe Data'] = f'Prediksi ({dataset_name})'
            all_data_for_plot.append(df_pred_plot[['year', 'month', 'Curah Hujan (mm)', 'Tipe Data']])
        else:
            st.warning(f"⚠️ Prediksi ({dataset_name}) tidak tersedia dalam rentang waktu yang dipilih atau tidak cocok dengan stasiun.")

    # --- Plot Bar Chart (Plotly Express) ---
    if not all_data_for_plot:
        st.error("❌ Tidak ada data (prediksi maupun ground truth) yang ditemukan untuk periode ini.")
        return

    df_plot = pd.concat(all_data_for_plot, ignore_index=True)
    df_plot['Bulan-Tahun'] = df_plot['month'].map(bulan_dict) + ' ' + df_plot['year'].astype(str)

    # Buat kolom urutan untuk sorting di plot
    df_plot['date_sort'] = pd.to_datetime(df_plot[['year', 'month']].assign(day=1))
    df_plot = df_plot.sort_values(by=['date_sort', 'Tipe Data'])

    # Warna untuk Bar Chart
    bar_color_map = {
        'Ground Truth (Rainfall)': 'saddlebrown',
        'Prediksi (0 Variabel)': 'royalblue',
        'Prediksi (10 Variabel)': 'deeppink',
        'Prediksi (51 Variabel)': 'forestgreen'
    }

    # Urutan Kategori di sumbu X
    x_order = df_plot['Bulan-Tahun'].unique().tolist()

    fig_bar = px.bar(
        df_plot,
        x='Bulan-Tahun',
        y='Curah Hujan (mm)',
        color='Tipe Data',
        barmode='group',
        color_discrete_map=bar_color_map,
        title=f'Curah Hujan Bulanan Komparatif ({date_range_str}) di {selected_station_name}',
        labels={'Curah Hujan (mm)': 'Curah Hujan (mm)', 'Bulan-Tahun': 'Bulan-Tahun'},
    )

    fig_bar.update_layout(
        xaxis_title="Bulan-Tahun",
        yaxis_title="Curah Hujan (mm)",
        legend_title="Tipe Data",
        bargap=0.15,
        xaxis={'categoryorder': 'array', 'categoryarray': x_order}
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    # --- Akhir Plot Bar Chart ---

    st.markdown("---")
    st.subheader(f"Scatter Plot Curah Hujan Aktual vs Prediksi Bulanan ({date_range_str})")

    # --- Plot Scatter Plot (Matplotlib) ---
    # layout : 1 baris 3 kolom (3 model)
    fig_scatter, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.style.use('ggplot')

    scatter_color_map = {
        '0 Variabel': 'royalblue',
        '10 Variabel': 'deeppink',
        '51 Variabel': 'forestgreen'
    }

    i = 0
    max_val = 0
    
    # Fill up to 3 plots, even if some models are commented out in dataset_info
    all_dataset_names = ['0 Variabel'] # Only current active one
    # if '10 Variabel' in dataset_info: all_dataset_names.append('10 Variabel')
    # if '51 Variabel' in dataset_info: all_dataset_names.append('51 Variabel')
    
    # Ensure there are 3 axes to iterate over for the 1x3 subplot setup
    for idx, dataset_name in enumerate(all_dataset_names):
        if dataset_name in all_combined_data:
            df_combined = all_combined_data[dataset_name]
            ax = axes[idx]

            # Hitung Metrik Rentang Kustom
            metrics = calculate_metrics(df_combined, 'rainfall', 'ch_pred')

            # Data untuk Scatter Plot (1 titik = 1 bulan)
            actual = df_combined['rainfall'].astype(float)
            pred = df_combined['ch_pred'].astype(float)

            # Update max_val
            if not actual.empty and not pred.empty:
                current_max = max(actual.max(), pred.max())
                if current_max > max_val:
                    max_val = current_max

            # Scatter Plot
            ax.scatter(actual, pred, color=scatter_color_map[dataset_name], label=dataset_name, alpha=0.7)

            # Teks Metrik di Pojok Kanan Bawah
            textstr = '\n'.join((
                r'MAE = %.2f' % (metrics['MAE'], ),
                r'RMSE = %.2f' % (metrics['RMSE'], ),
                r'$R^2$ = %.2f' % (metrics['R2'], )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
            ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

            # Label dan Judul
            ax.set_title(f'Model {dataset_name}', fontsize=14)
            ax.set_xlabel('Curah Hujan Aktual Bulanan (mm)', fontsize=12)
            ax.set_ylabel('Curah Hujan Prediksi Bulanan (mm)', fontsize=12)
        else:
            # Kosongkan plot jika data tidak tersedia
            axes[idx].set_title(f'Model {dataset_name} (Data Tidak Tersedia)', fontsize=14, color='gray')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    # Atur batas sumbu X dan Y agar sama, dan tambahkan garis 1:1
    plot_limit = max_val * 1.05 if max_val > 0 else 100
    for ax in axes:
        ax.set_xlim(0, plot_limit)
        ax.set_ylim(0, plot_limit)
        ax.plot([0, plot_limit], [0, plot_limit], color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    st.pyplot(fig_scatter)
    # --- Akhir Plot Scatter Plot ---

# Main Streamlit app logic 
st.title("📊 DK Viewer - Tabel Analisis Komparatif") 

if 'comparative_data' not in st.session_state: 
    st.session_state.comparative_data = None 
if 'combinations' not in st.session_state: 
    st.session_state.combinations = [] 
if 'selected_station_name' not in st.session_state: 
    st.session_state.selected_station_name = station_names[0] 
# Tambahkan untuk menyimpan data mentah untuk plot baru
if 'all_raw_data' not in st.session_state:
    st.session_state.all_raw_data = None

with st.sidebar.form("config_form"): 
    st.header("⚙️ Konfigurasi") 
    
    year_options = list(range(1985, 2015)) 
    bulan_options = list(range(1, 13)) 
    
    display_type = st.radio( 
        "Pilih Tampilan Data:", 
        ["Time Series & Summary", "Bar Chart dan Scatter Plot Tahunan"]
    ) 
    
    # --- Konfigurasi Rentang Waktu --- 
    st.subheader("Dari") 
    col1, col2 = st.columns(2) 
    with col1: 
        bulan_from = st.selectbox( 
            "Bulan Awal:", 
            bulan_options, 
            index=0, 
            format_func=lambda x: bulan_dict[x], 
            key="bulan_from_ts" if display_type == "Time Series & Summary" else "bulan_from_bar"
        ) 
    with col2: 
        tahun_from = st.selectbox("Tahun Awal:", year_options, index=0, key="tahun_from_ts" if display_type == "Time Series & Summary" else "tahun_from_bar") 
    
    st.subheader("Sampai") 
    col3, col4 = st.columns(2) 
    with col3: 
        bulan_until = st.selectbox( 
            "Bulan Akhir:", 
            bulan_options, 
            index=len(bulan_options) - 1, 
            format_func=lambda x: bulan_dict[x],
            key="bulan_until_ts" if display_type == "Time Series & Summary" else "bulan_until_bar" 
        ) 
    with col4: 
        tahun_until = st.selectbox("Tahun Akhir:", year_options, index=len(year_options) - 1, key="tahun_until_ts" if display_type == "Time Series & Summary" else "tahun_until_bar") 

    selected_station_name = st.selectbox("Pilih stasiun:", station_names) 
    st.session_state.selected_station_name = selected_station_name 
    
    submit = st.form_submit_button("🚀 Submit konfigurasi dan bandingkan") 

if submit: 
    from_date_tuple = (tahun_from, bulan_from) 
    until_date_tuple = (tahun_until, bulan_until) 

    if from_date_tuple > until_date_tuple: 
        st.error("❌ Tanggal 'Dari' tidak boleh lebih baru dari tanggal 'Sampai'.") 
    elif display_type == "Bar Chart dan Scatter Plot Tahunan": 
        # Panggil fungsi rentang waktu custom (menggantikan mode tahunan original)
        st.session_state.comparative_data = {} # Reset data Time Series 
        st.session_state.all_raw_data = None # Reset data mentah
        plot_comparative_charts_monthly(tahun_from, bulan_from, tahun_until, bulan_until, selected_station_name) 
        st.success(f"✅ Data berhasil dimuat dan siap untuk Bar Chart dan Scatter Plot untuk rentang **{bulan_dict[bulan_from]} {tahun_from}** hingga **{bulan_dict[bulan_until]} {tahun_until}**.")
        
    else: # Time Series & Summary 
        tahun_final = list(range(tahun_from, tahun_until + 1)) 
        is_all_stations = selected_station_name == "Rata-Rata Seluruh Stasiun"
        
        filtered_data_dict = {} 
        raw_data_all_models = [] # Untuk data mentah dari semua stasiun (Prediksi)
        raw_data_padanan = []    # Untuk data mentah dari semua stasiun (Padanan)

        start_date = pd.to_datetime(f"{tahun_from}-{bulan_from}-01")
        end_date = pd.to_datetime(f"{tahun_until}-{bulan_until}-01")
        
        for dataset_name in dataset_info.keys(): 
            all_filtered = [] 
            
            # 1. Muat dan Gabungkan data prediksi dan padanan untuk semua tahun
            for th in tahun_final: 
                df_main = load_data(dataset_name, th) 
                df_padanan = load_padanan_data(th) 
                
                # --- Logika Penambahan Data Set Baru: Semua Data (Mentah) ---
                if not df_main.empty:
                    # Filter prediksi untuk 7 stasiun yang diinginkan
                    df_pred_stations_raw = df_main.copy()
                    if {'latitude', 'longitude'}.issubset(df_pred_stations_raw.columns):
                        df_pred_stations_raw['coord_tuple'] = list(zip(df_pred_stations_raw['latitude'].astype(float), df_pred_stations_raw['longitude'].astype(float)))
                        df_pred_stations_raw = df_pred_stations_raw[df_pred_stations_raw['coord_tuple'].isin(station_coords)].copy()
                        df_pred_stations_raw.drop(columns=['coord_tuple'], inplace=True)
                    
                    if not df_pred_stations_raw.empty:
                        # Tambahkan kolom model_name dan merge dengan padanan untuk data mentah
                        df_pred_stations_raw['model_name'] = dataset_name
                        
                        df_merged_raw = pd.merge(df_pred_stations_raw, df_padanan[['year', 'month', 'latitude', 'longitude', 'rainfall']], 
                                                 on=['month', 'year', 'latitude', 'longitude'], how='left')
                        raw_data_all_models.append(df_merged_raw)

                # --- Lanjutkan Logika Time Series & Summary (Rata-rata/Stasiun Tunggal) ---
                
                if not df_main.empty and not df_padanan.empty: 
                    # Merge data prediksi dan padanan
                    df_merged_year = pd.merge(df_main, df_padanan, on=['month', 'year', 'latitude', 'longitude'], how='left') 
                    df_merged_year = df_merged_year.drop_duplicates(subset=['latitude', 'longitude', 'month', 'year']) 
                    all_filtered.append(df_merged_year) 
                elif not df_main.empty: 
                    all_filtered.append(df_main) 
            
            if all_filtered: 
                df_filtered_all = pd.concat(all_filtered, ignore_index=True) 
                
                # 2. Filter/Rata-rata Stasiun
                if is_all_stations:
                    # Hitung rata-rata per bulan dari SEMUA stasiun (pakai hanya grid yang match ke stasiun jika tersedia)
                    df_temp = df_filtered_all.copy()
                    if {'latitude', 'longitude'}.issubset(df_temp.columns):
                        df_temp['coord_tuple'] = list(zip(df_temp['latitude'].astype(float), df_temp['longitude'].astype(float)))
                        if df_temp['coord_tuple'].isin(station_coords).any():
                            df_temp = df_temp[df_temp['coord_tuple'].isin(station_coords)].copy()
                        if 'coord_tuple' in df_temp.columns:
                            df_temp.drop(columns=['coord_tuple'], inplace=True)
                    
                    aggregation_cols = ['ch_pred']
                    if 'rainfall' in df_temp.columns:
                        aggregation_cols.append('rainfall')
                    
                    # Use grouping on df_temp
                    if not df_temp.empty:
                        df_filtered_station = df_temp.groupby(['year', 'month']).agg(
                            **{col: ('ch_pred' if col == 'ch_pred' else 'rainfall', 'mean') for col in aggregation_cols}
                        ).reset_index()
                    else:
                        df_filtered_station = pd.DataFrame()
                else:
                    station_info = next(s for s in station_data if s["name"] == selected_station_name) 
                    df_filtered_station = df_filtered_all[ 
                        (df_filtered_all['latitude'] == station_info['lat']) & 
                        (df_filtered_all['longitude'] == station_info['lon']) 
                    ].copy() 

                # 3. Hitung Metrik Error (Hanya jika ada kolom 'rainfall')
                if 'rainfall' in df_filtered_station.columns and not df_filtered_station.empty: 
                    df_filtered_station['error_bias'] = df_filtered_station['ch_pred'] - df_filtered_station['rainfall'] 
                    df_filtered_station['absolute_error'] = abs(df_filtered_station['ch_pred'] - df_filtered_station['rainfall']) 
                    df_filtered_station['squared_error'] = (df_filtered_station['ch_pred'] - df_filtered_station['rainfall'])**2 
                else: 
                    # If df empty, create empty columns to keep format
                    df_filtered_station['error_bias'] = None 
                    df_filtered_station['absolute_error'] = None 
                    df_filtered_station['squared_error'] = None 

                # 4. Filter Rentang Waktu
                if not df_filtered_station.empty:
                    df_filtered_station['date'] = pd.to_datetime(df_filtered_station[['year', 'month']].assign(day=1))
                    df_filtered_station = df_filtered_station[(df_filtered_station['date'] >= start_date) & (df_filtered_station['date'] <= end_date)].copy()
                    df_filtered_station.drop(columns=['date'], inplace=True, errors='ignore')
                else:
                    df_filtered_station = pd.DataFrame()

                filtered_data_dict[dataset_name] = df_filtered_station 
            else: 
                filtered_data_dict[dataset_name] = pd.DataFrame() 
        
        # --- Pengolahan Data Mentah Final ---
        if raw_data_all_models:
            df_raw_combined = pd.concat(raw_data_all_models, ignore_index=True)
            # Filter rentang waktu untuk data mentah
            if not df_raw_combined.empty:
                df_raw_combined['date'] = pd.to_datetime(df_raw_combined[['year', 'month']].assign(day=1))
                df_raw_combined = df_raw_combined[(df_raw_combined['date'] >= start_date) & (df_raw_combined['date'] <= end_date)].copy()
                df_raw_combined.drop(columns=['date'], inplace=True, errors='ignore')
                
                # Hanya simpan kolom yang relevan untuk plot mentah
                st.session_state.all_raw_data = df_raw_combined[['year', 'month', 'latitude', 'longitude', 'ch_pred', 'rainfall', 'model_name']]
            else:
                st.session_state.all_raw_data = pd.DataFrame()
        else:
            st.session_state.all_raw_data = pd.DataFrame()


        st.session_state.comparative_data = filtered_data_dict 
        st.success(f"✅ Data berhasil dimuat dan siap untuk perbandingan Time Series dari **{bulan_dict[bulan_from]} {tahun_from}** hingga **{bulan_dict[bulan_until]} {tahun_until}** untuk {selected_station_name}.") 

# --- Tampilan Time Series & Summary --- 
if st.session_state.comparative_data and st.session_state.comparative_data.keys() and display_type == "Time Series & Summary": 
    
    # Ringkasan Statistik 
    st.markdown("---") 
    st.subheader(f"Ringkasan Statistik Komparatif (Rata-Rata Bulanan) untuk {st.session_state.selected_station_name}") 
    
    summary_cols = ['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error'] 
    comparison_summary = [] 
    
    for dataset_name, df in st.session_state.comparative_data.items(): 
        if not df.empty: 
            # Re-calculate overall metrics for Time Series display (just in case)
            metrics = calculate_metrics(df, 'rainfall', 'ch_pred')
            
            summary_row = {"Metrik": dataset_name} 
            for col in summary_cols: 
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): 
                    summary_row[f"Mean ({col})"] = df[col].mean() 
                    summary_row[f"Sum ({col})"] = df[col].sum() 
                else: 
                    summary_row[f"Mean ({col})"] = None 
                    summary_row[f"Sum ({col})"] = None 
            
            # Add MAE, RMSE, R2
            summary_row['MAE'] = metrics['MAE']
            summary_row['RMSE'] = metrics['RMSE']
            summary_row['R2'] = metrics['R2']
            
            comparison_summary.append(summary_row) 

    if comparison_summary: 
        # Tampilkan metrik error di bagian atas
        metrics_df = pd.DataFrame(comparison_summary).set_index("Metrik")[['MAE', 'RMSE', 'R2']].T
        for col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else None)
        st.markdown("##### Metrik Error Keseluruhan")
        st.dataframe(metrics_df)
        
        # Tampilkan ringkasan mean/sum (Ringkasan Statistik)
        summary_df_full = pd.DataFrame(comparison_summary).set_index("Metrik")
        summary_df_cols = [col for col in summary_df_full.columns if col not in ['MAE', 'RMSE', 'R2']]
        comparison_df = summary_df_full[summary_df_cols].T
        for col in comparison_df.columns: 
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else None) 
        
        st.markdown("##### Ringkasan Data Bulanan (Rata-Rata)")
        st.dataframe(comparison_df) 
    else: 
        st.warning("⚠️ Tidak ada data rata-rata bulanan untuk ditampilkan. Pastikan rentang waktu valid.") 
        
    # Plot Time Series 
    st.markdown("---") 
    st.subheader("Plot Perbandingan Time Series") 
    
    # --- MODIFIKASI: Tambahkan opsi 'Semua Data (Raw)' ---
    plot_options = list(dataset_info.keys())
    
    # Check if raw data is available before adding the option
    if not st.session_state.all_raw_data.empty:
        plot_options.append('Semua Data (Raw)')
        
    selected_models = st.multiselect( 
        "Pilih model yang akan di-plot:", 
        options=plot_options, 
        default=['0 Variabel'], 
        key='ts_models' 
    ) 

    metrics_to_plot_options = ['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error']
    if 'Semua Data (Raw)' in selected_models:
        metrics_to_plot_options = ['ch_pred', 'rainfall'] # Error metrics don't make sense for raw data across all stations in a single monthly line plot
    
    metrics_to_plot = st.multiselect( 
        "Pilih metrik untuk di-plot:", 
        options=metrics_to_plot_options, 
        default=['ch_pred', 'rainfall'], 
        key='ts_metrics' 
    ) 

    if not selected_models or not metrics_to_plot: 
        st.info("💡 Pilih setidaknya satu model dan satu metrik untuk menampilkan plot.") 
    else: 
        is_rainfall_selected = 'rainfall' in metrics_to_plot 
        other_metrics = [m for m in metrics_to_plot if m != 'rainfall'] 

        dfs_to_plot = [] 
        
        # Logika penggabungan data untuk Time Series 
        
        # 1. Tambahkan data rata-rata bulanan (jika tidak memilih 'Semua Data (Raw)')
        if 'Semua Data (Raw)' not in selected_models:
            if is_rainfall_selected: 
                # Gunakan Ground Truth dari model pertama yang dipilih (semua Ground Truth rata-rata harus sama)
                first_model = selected_models[0] 
                df_rainfall = st.session_state.comparative_data.get(first_model) 
                if not df_rainfall.empty and 'rainfall' in df_rainfall.columns: 
                    rainfall_df = df_rainfall[['year', 'month', 'rainfall']].copy() 
                    rainfall_df['model_name'] = 'Ground Truth' 
                    rainfall_df = rainfall_df.rename(columns={'rainfall': 'Value'}) 
                    rainfall_df['Metric'] = 'rainfall' 
                    dfs_to_plot.append(rainfall_df) 

            for model_name in selected_models: 
                df = st.session_state.comparative_data.get(model_name, pd.DataFrame()) 
                if not df.empty and other_metrics: 
                    existing_other_metrics = [m for m in other_metrics if m in df.columns] 
                    
                    if existing_other_metrics: 
                        df_other_metrics = df[['year', 'month'] + existing_other_metrics].copy() 
                        df_other_metrics['model_name'] = model_name 
                        
                        melted_df = df_other_metrics.melt( 
                            id_vars=['year', 'month', 'model_name'], 
                            value_vars=existing_other_metrics, 
                            var_name='Metric', 
                            value_name='Value' 
                        ) 
                        dfs_to_plot.append(melted_df) 

        # 2. Tambahkan data mentah (jika 'Semua Data (Raw)' dipilih)
        if 'Semua Data (Raw)' in selected_models and not st.session_state.all_raw_data.empty:
            df_raw = st.session_state.all_raw_data.copy()
            
            # Melt data prediksi
            df_raw_pred = df_raw.rename(columns={'ch_pred': 'Value'})[['year', 'month', 'model_name', 'Value']]
            df_raw_pred['Metric'] = 'ch_pred'
            df_raw_pred['model_name'] = 'Prediksi (Raw)'
            
            # Melt data aktual/padanan
            df_raw_rain = df_raw.rename(columns={'rainfall': 'Value'})[['year', 'month', 'Value']].drop_duplicates()
            df_raw_rain['model_name'] = 'Ground Truth (Raw)'
            df_raw_rain['Metric'] = 'rainfall'
            
            # Gabungkan dan filter metrik yang dipilih
            df_raw_melted = pd.concat([df_raw_pred, df_raw_rain], ignore_index=True)
            df_raw_melted = df_raw_melted[df_raw_melted['Metric'].isin(metrics_to_plot)].copy()

            # Hanya untuk plot 'Semua Data (Raw)', label digabungkan
            df_raw_melted['combined_label'] = df_raw_melted['model_name']
            
            dfs_to_plot.append(df_raw_melted)
        
        # --- Plotting ---

        if not dfs_to_plot: 
            st.warning("⚠️ Data tidak tersedia untuk model atau metrik yang dipilih.") 
        else: 
            combined_df = pd.concat(dfs_to_plot, ignore_index=True) 
            combined_df['date'] = pd.to_datetime(combined_df[['year', 'month']].assign(day=1)) 
            combined_df.sort_values(by='date', inplace=True) 
            
            # Kolom label untuk data Rata-rata (hanya jika data raw tidak ada atau model rata-rata yang dipilih)
            if 'combined_label' not in combined_df.columns:
                combined_df['combined_label'] = combined_df['Metric'] + ' (' + combined_df['model_name'] + ')' 
                combined_df.loc[combined_df['Metric'] == 'rainfall', 'combined_label'] = 'Rainfall (Ground Truth)' 
            
            color_map = { 
                'Rainfall (Ground Truth)': 'saddlebrown', 'ch_pred (0 Variabel)': 'royalblue', 
                'error_bias (0 Variabel)': 'darkblue', 'absolute_error (0 Variabel)': 'midnightblue', 
                'squared_error (0 Variabel)': 'navy', 
                
                # Tambahkan untuk data mentah
                'Prediksi (Raw)': 'orange',
                'Ground Truth (Raw)': 'red',
            } 

            fig = px.scatter(
                combined_df, 
                x='date', y='Value', color='combined_label', 
                title=f'Perbandingan Time Series untuk {st.session_state.selected_station_name}', 
                labels={'Value': 'Nilai', 'date': 'Tanggal', 'combined_label': 'Metrik'}, 
                color_discrete_map=color_map 
            ) 
            
            # Ubah tipe trace menjadi line+scatter hanya untuk data rata-rata (jika ada)
            if 'Semua Data (Raw)' not in selected_models:
                fig = px.line(
                    combined_df, 
                    x='date', y='Value', color='combined_label', 
                    title=f'Perbandingan Time Series Bulanan (Rata-rata) untuk {st.session_state.selected_station_name}', 
                    labels={'Value': 'Nilai', 'date': 'Tanggal', 'combined_label': 'Metrik'}, 
                    markers=True, color_discrete_map=color_map 
                )
                
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True) 

# --- Pesan Akhir (Tidak Berubah) --- 
elif not submit: 
    st.info("💡 Pilih Tampilan Data, rentang waktu/tahun, dan stasiun di sidebar, lalu tekan 'Submit konfigurasi dan bandingkan' untuk melihat data.") 

st.markdown( 
    """ 
    <style> 
    @keyframes fadeIn { 
        from {opacity: 0;} 
        to {opacity: 1;} 
    } 
    .fade-in-text { 
        animation: fadeIn 2s ease-in-out; 
        text-align: center; 
        margin-top: 20px; 
    } 
    </style> 

    <div class="fade-in-text"> 
        <h4>BRIN Research Team</h4> 
        <p><em>Data Visualization by Tsaqib</em></p> 
    </div> 
    """, 
    unsafe_allow_html=True 
)