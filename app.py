# EPİAŞ PTF TAHMİN SİSTEMİ - v2.1 (FIXED)
# Last Update: 2026-05-03 17:01
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import json
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

from snapping import apply_price_snapping

BASE_DIR = Path(__file__).resolve().parent
LOCAL_TZ = "Europe/Istanbul"

def file_mtime(filename):
    path = BASE_DIR / filename
    return path.stat().st_mtime if path.exists() else None

def to_local_index(index):
    if index.tz is None:
        return index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    return index.tz_convert(LOCAL_TZ)

def read_json_file(filename, default=None):
    path = BASE_DIR / filename
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}

def format_dt(value):
    if not value:
        return "Henüz yok"
    try:
        dt = pd.to_datetime(value, utc=True).tz_convert(LOCAL_TZ)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)

def run_update_pipeline():
    env = os.environ.copy()
    try:
        epias = st.secrets.get("epias", {})
        if epias.get("username") and epias.get("password"):
            env["EPIAS_USERNAME"] = epias["username"]
            env["EPIAS_PASSWORD"] = epias["password"]
    except Exception:
        pass

    return subprocess.run(
        [sys.executable, str(BASE_DIR / "update_pipeline.py")],
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

def render_model_management(container=None, button_key="update_model"):
    container = container or st
    status = read_json_file("model_status.json")
    metrics = read_json_file("model_metrics.json")
    active_metrics = status.get("metrics") or metrics

    with container.expander("🧠 Model Yönetimi", expanded=True):
        st.caption("Verileri çekip modeli yeniden eğitir. İşlem bitene kadar sayfayı kapatmayın.")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Son Eğitim", format_dt(active_metrics.get('trained_at')))
        with m2:
            st.metric("Son Veri", format_dt((status or {}).get('data_end') or active_metrics.get('data_end')))
        with m3:
            st.metric("MAPE", f"%{active_metrics.get('mape', 0):,.2f}" if active_metrics else "Henüz yok")

        if active_metrics:
            st.write(
                f"**MAE:** {active_metrics.get('mae', 0):,.0f} TL/MWh  "
                f"**Satır:** {active_metrics.get('rows', 0):,.0f}"
            )
        if status:
            state = "Başarılı" if status.get("ok") else "Son işlem başarısız/yarım"
            st.caption(f"Son pipeline: {state} · {format_dt(status.get('finished_at'))}")

        if st.button("🔄 Verileri Çek ve Modeli Eğit", use_container_width=True, key=button_key):
            with st.spinner("Veriler çekiliyor, model yeniden eğitiliyor..."):
                result = run_update_pipeline()

            if result.returncode == 0:
                load_data.clear()
                load_model.clear()
                st.success("Model güncellendi. Sayfa yeni modeli kullanacak.")
                st.rerun()

            st.error("Güncelleme tamamlanamadı. Eski model kullanılmaya devam ediyor.")
            with st.expander("Hata detayı"):
                st.code((result.stderr or result.stdout)[-4000:])

def render_sidebar_model_management():
    status = read_json_file("model_status.json")
    metrics = read_json_file("model_metrics.json")
    active_metrics = status.get("metrics") or metrics

    with st.sidebar.expander("🧠 Model Yönetimi", expanded=False):
        st.write(f"**Son eğitim:** {format_dt(active_metrics.get('trained_at'))}")
        st.write(f"**Son veri:** {format_dt((status or {}).get('data_end') or active_metrics.get('data_end'))}")
        if st.button("🔄 Verileri Çek ve Modeli Eğit", use_container_width=True, key="update_model_sidebar"):
            with st.spinner("Veriler çekiliyor, model yeniden eğitiliyor..."):
                result = run_update_pipeline()
            if result.returncode == 0:
                load_data.clear()
                load_model.clear()
                st.success("Model güncellendi.")
                st.rerun()
            st.error("Güncelleme tamamlanamadı.")

st.set_page_config(page_title="EPİAŞ PTF Kahini", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────────
# PREMIUM CSS TASARIMI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Ana Tema */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Başlık Alanı */
    .hero-container {
        background: linear-gradient(135deg, #0F0C29 0%, #302B63 50%, #24243E 100%);
        border-radius: 20px;
        padding: 35px 40px;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(78,205,196,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ECDC4, #44B09E, #e8d44d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.55);
        font-size: 0.9rem;
        margin-top: 8px;
        font-weight: 300;
        letter-spacing: 0.3px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(78,205,196,0.15);
        color: #4ECDC4;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 12px;
        border: 1px solid rgba(78,205,196,0.25);
        letter-spacing: 1px;
    }

    /* Metrik Kartlar */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        border-color: rgba(78,205,196,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(78,205,196,0.1);
    }
    .glass-card .card-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .glass-card .card-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.45);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .glass-card .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .glass-card .card-value.teal { color: #4ECDC4; }
    .glass-card .card-value.coral { color: #FF6B6B; }
    .glass-card .card-value.gold { color: #FFD93D; }
    .glass-card .card-value.purple { color: #A78BFA; }
    
    /* Alt çizgi efekti */
    .card-accent-teal { border-bottom: 3px solid #4ECDC4; }
    .card-accent-coral { border-bottom: 3px solid #FF6B6B; }
    .card-accent-gold { border-bottom: 3px solid #FFD93D; }
    .card-accent-purple { border-bottom: 3px solid #A78BFA; }

    /* Sekme Stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(78,205,196,0.2), rgba(68,176,158,0.1));
    }

    /* Bölüm Başlıkları */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 25px 0 15px 0;
    }
    .section-header .icon {
        font-size: 1.3rem;
    }
    .section-header .text {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(255,255,255,0.85);
    }
    .section-header .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(78,205,196,0.3), transparent);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0C29 0%, #1a1a2e 100%);
    }
    
    /* Butonlar */
    .stButton > button {
        background: linear-gradient(135deg, #4ECDC4, #44B09E);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(78,205,196,0.35);
        transform: translateY(-1px);
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: rgba(255,255,255,0.2);
        font-size: 0.75rem;
        margin-top: 60px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# VERİ VE MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_mtime, features_mtime):
    model = xgb.XGBRegressor()
    model.load_model(str(BASE_DIR / "ptf_xgboost_model.json"))
    with open(BASE_DIR / "model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_data(data_mtime):
    data_path = BASE_DIR / "model_ready_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, index_col='date', parse_dates=True)
        return df
    return None

def make_chart_layout(title=None):
    """Tüm grafikler için ortak premium layout"""
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.7)', family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="TL/MWh", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=0, r=10, t=40 if title else 10, b=0),
        title=dict(text=title, font=dict(size=14, color='rgba(255,255,255,0.6)')) if title else None,
        hovermode='x unified',
    )

# ─────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────
def main():
    # Hero Başlık
    st.markdown("""
    <div class="hero-container">
        <p class="hero-title">⚡ EPİAŞ Yapay Zeka PTF Tahmin Sistemi</p>
        <p class="hero-subtitle">XGBoost ML modeli ile 16 aylık piyasa verisi üzerinden eğitilmiş gerçek zamanlı fiyat tahminleme platformu</p>
        <span class="hero-badge">XGBoost + SNAPPING ENGINE</span>
    </div>
    """, unsafe_allow_html=True)

    render_model_management(button_key="update_model_main")

    df = load_data(file_mtime("model_ready_data.csv"))
    if df is None:
        st.error("Veri seti bulunamadı! Lütfen arkaplanda veri çekme ve işleme adımlarını tamamlayın.")
        return
    df = df.copy()

    try:
        model, features = load_model(
            file_mtime("ptf_xgboost_model.json"),
            file_mtime("model_features.pkl")
        )
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error("Modelin beklediği bazı geçmiş veri kolonları bulunamadı: " + ", ".join(missing_features))
        return

    X = df[features]
    df['predicted_price'] = model.predict(X)
    
    # --- EKSİK VERİ TAMAMLAMA (Interpolation) ---
    # Eğer veride saatlik boşluklar varsa (Mayıs-Haziran 2025 gibi), buraları doldurur.
    df = df.resample('h').mean(numeric_only=True)
    df['price'] = df['price'].interpolate(method='linear')
    df['predicted_price'] = df['predicted_price'].interpolate(method='linear')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    # --------------------------------------------

    df['predicted_price'] = apply_price_snapping(df)

    render_sidebar_model_management()

    # Sekmeler
    tab1, tab2, tab3 = st.tabs(["📊  Geçmiş Veri Analizi", "🔮  Gelecek Tahmini (GÖP)", "📅  Gelecek Ay Beklentisi"])
    
    # ═══════════════════════════════════════════
    # SEKME 1: GEÇMİŞ VERİ ANALİZİ
    # ═══════════════════════════════════════════
    with tab1:
        st.sidebar.markdown("### ⚙️ Kontrol Paneli")
        st.sidebar.markdown("---")
        
        local_index = to_local_index(df.index)
        min_date = local_index.min().date()
        max_date = local_index.max().date()
        
        selected_dates = st.sidebar.date_input(
            "📆 Tarih Aralığı",
            value=(max_date - pd.Timedelta(days=7), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = end_date = selected_dates[0]

        mask = (local_index.date >= start_date) & (local_index.date <= end_date)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            st.warning("Seçilen tarih aralığında veri bulunamadı.")
        else:
            actual_mean = filtered_df['price'].mean()
            predicted_mean = filtered_df['predicted_price'].mean()
            mae = abs(filtered_df['price'] - filtered_df['predicted_price']).mean()
            # Model Doğruluğu: MAPE yerine daha stabil bir skor
            # MAE'yi ortalama fiyata oranlayıp % bazlı bir skor üretiyoruz
            if actual_mean > 0:
                error_ratio = mae / actual_mean
                accuracy = max(0, (1 - error_ratio) * 100)
            else:
                accuracy = 0

            # Metrik Kartları
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="glass-card card-accent-teal">
                    <div class="card-icon">💰</div>
                    <div class="card-label">Gerçekleşen Ort. PTF</div>
                    <div class="card-value teal">{actual_mean:,.0f} ₺</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="glass-card card-accent-coral">
                    <div class="card-icon">🤖</div>
                    <div class="card-label">Model Tahmini Ort.</div>
                    <div class="card-value coral">{predicted_mean:,.0f} ₺</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="glass-card card-accent-gold">
                    <div class="card-icon">📏</div>
                    <div class="card-label">Ortalama Sapma (MAE)</div>
                    <div class="card-value gold">{mae:,.0f} ₺</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="glass-card card-accent-purple">
                    <div class="card-icon">🎯</div>
                    <div class="card-label">Model Doğruluğu</div>
                    <div class="card-value purple">%{accuracy:,.1f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Grafik
            st.markdown("""<div class="section-header"><span class="icon">📈</span><span class="text">Gerçekleşen vs Tahmin</span><span class="line"></span></div>""", unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df.index, y=filtered_df['price'],
                mode='lines', name='Gerçek PTF',
                line=dict(color='#4ECDC4', width=2),
                fill='tozeroy', fillcolor='rgba(78,205,196,0.06)'
            ))
            fig.add_trace(go.Scatter(
                x=filtered_df.index, y=filtered_df['predicted_price'],
                mode='lines', name='Model Tahmini',
                line=dict(color='#FF6B6B', width=2, dash='dot')
            ))
            fig.update_layout(**make_chart_layout())
            st.plotly_chart(fig, use_container_width=True)

            # Tablo
            st.markdown("""<div class="section-header"><span class="icon">📋</span><span class="text">Saatlik Detay Tablosu</span><span class="line"></span></div>""", unsafe_allow_html=True)
            
            display_df = filtered_df[['price', 'predicted_price']].copy()
            # Düzeltilen format:
            display_df.index = to_local_index(display_df.index).strftime('%Y-%m-%d %H:00')
            display_df.columns = ['Gerçek PTF', 'Tahmin']
            st.dataframe(display_df.style.format("{:,.2f}"), use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=True, sheet_name="Tahminler")
            st.download_button("📥 Excel İndir", data=buffer.getvalue(), file_name="epias_gecmis_tahminler.xlsx", use_container_width=True)

    # ═══════════════════════════════════════════
    # SEKME 2: GELECEK TAHMİNİ (GÖP)
    # ═══════════════════════════════════════════
    with tab2:
        st.markdown("""<div class="section-header"><span class="icon">🔮</span><span class="text">Gün Öncesi Piyasası (GÖP) Tahmini</span><span class="line"></span></div>""", unsafe_allow_html=True)
        st.caption("EPİAŞ'ın yayınladığı plan verilerini kullanarak, PTF'si henüz belli olmayan günlerin saatlik fiyat tahminini üretir.")
        
        col_date, col_btn = st.columns([2, 1])
        with col_date:
            target_date = st.date_input("Hedef Tarih", key="gop_date")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_forecast = st.button("⚡ Tahmin Üret", use_container_width=True)
        
        if run_forecast:
            with st.spinner("EPİAŞ'a bağlanılıyor ve model çalıştırılıyor..."):
                try:
                    import predict_future
                    import importlib
                    importlib.reload(predict_future)
                    from predict_future import predict_future_day
                    res, is_sim, msg = predict_future_day(target_date.strftime("%Y-%m-%d"))
                    
                    if res is not None:
                        if is_sim:
                            st.warning(f"⚠️ {msg}")
                        else:
                            st.success(f"✅ {target_date} planları EPİAŞ'tan başarıyla çekildi!")
                        
                        avg_price = res['predicted_price'].mean()
                        max_price = res['predicted_price'].max()
                        min_price = res['predicted_price'].min()
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""
                            <div class="glass-card card-accent-coral">
                                <div class="card-icon">📊</div>
                                <div class="card-label">Günlük Ortalama</div>
                                <div class="card-value coral">{avg_price:,.0f} ₺</div>
                            </div>""", unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="glass-card card-accent-gold">
                                <div class="card-icon">🔺</div>
                                <div class="card-label">En Yüksek Saat</div>
                                <div class="card-value gold">{max_price:,.0f} ₺</div>
                            </div>""", unsafe_allow_html=True)
                        with m3:
                            st.markdown(f"""
                            <div class="glass-card card-accent-teal">
                                <div class="card-icon">🔻</div>
                                <div class="card-label">En Düşük Saat</div>
                                <div class="card-value teal">{min_price:,.0f} ₺</div>
                            </div>""", unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Grafik
                        fig2 = go.Figure()
                        colors = ['#4ECDC4' if v < avg_price else '#FF6B6B' for v in res['predicted_price']]
                        fig2.add_trace(go.Bar(
                            x=res.index.strftime('%H:00'), y=res['predicted_price'],
                            marker_color=colors, name='Beklenen PTF',
                            marker_line_width=0, opacity=0.85
                        ))
                        fig2.add_hline(y=avg_price, line_dash="dash", line_color="#FFD93D",
                                       annotation_text=f"Ort: {avg_price:,.0f} ₺", annotation_font_color="#FFD93D")
                        fig2.update_layout(**make_chart_layout())
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        st.markdown("""<div class="section-header"><span class="icon">📋</span><span class="text">Saatlik Detay</span><span class="line"></span></div>""", unsafe_allow_html=True)
                        res_display = res.copy()
                        # Düzeltilen format (Gelecek tahmini için):
                        res_display.index = to_local_index(res_display.index).strftime('%Y-%m-%d %H:00')
                        
                        # Sütunları Türkçeleştir ve sırala
                        col_map = {
                            'predicted_price': 'Tahmini PTF',
                            'lep': 'Yük Planı',
                            'planned_total_gen': 'Toplam Üretim',
                            'planned_gas_gen': 'Doğalgaz Üretim',
                            'ruzgar': 'Rüzgar Plan',
                            'gunes': 'Güneş Plan',
                            'temperature': 'Sıcaklık (C)',
                            'wind_speed': 'Rüzgar Hızı (km/h)',
                            'precipitation': 'Yağış (mm)',
                            'cloud_cover': 'Bulutluluk (%)',
                            'price_independent_sales': 'Bağımsız Satış'
                        }
                        
                        display_cols = [c for c in col_map.keys() if c in res_display.columns]
                        res_display = res_display[display_cols].rename(columns=col_map)
                        st.dataframe(res_display.style.format("{:,.2f}"), use_container_width=True)
                        
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='openpyxl') as w:
                            res_display.to_excel(w, index=True, sheet_name="Gelecek Tahmin")
                        st.download_button("📥 Excel İndir", data=buf.getvalue(), file_name=f"tahmin_{target_date}.xlsx", use_container_width=True)
                    else:
                        st.error(f"Hata: {msg}")
                except Exception as e:
                    st.error(f"Tahmin motoru hatası: {str(e)}")

    # ═══════════════════════════════════════════
    # SEKME 3: AYLIK TREND TAHMİNİ
    # ═══════════════════════════════════════════
    with tab3:
        st.markdown("""<div class="section-header"><span class="icon">📊</span><span class="text">Geçmiş Aylık Performans (Gerçekleşen vs Tahmin)</span><span class="line"></span></div>""", unsafe_allow_html=True)
        st.caption("Modelin Ocak 2025'ten bugüne kadarki aylık ortalama tahmin başarısını gösterir.")

        # Aylık verileri hazırla
        try:
            # Geçmiş ham verileri projeksiyon için de yükleyelim
            df_ptf_25 = pd.read_csv("ptf_2025_saatlik_ham.csv", parse_dates=['date'])
            df_ptf_26 = pd.read_csv("ptf_2026_saatlik_ham.csv", parse_dates=['date'])
            df_ptf_all = pd.concat([df_ptf_25, df_ptf_26], ignore_index=True)
            df_ptf_all['date'] = pd.to_datetime(df_ptf_all['date'], utc=True)
            df_ptf_all.set_index('date', inplace=True)
            
            # Model performansını model_ready_data'dan (df) alalım
            m_df = df[['price', 'predicted_price']].resample('ME').mean()
            m_df['Ay'] = m_df.index.strftime('%Y-%m')
            m_df['Sapma (%)'] = ((m_df['predicted_price'] - m_df['price']) / m_df['price']) * 100
            
            # Projeksiyon için gerekli değişkeni tanımlayalım
            monthly_actual = df_ptf_all['price'].resample('ME').mean()
            df_ptf = df_ptf_all # Projeksiyon butonu için
            
            # Tabloyu göster
            st.subheader("📋 Aylık Ortalama Kıyaslama Tablosu")
            
            # Ay formatını Türkçeleştir (Örn: 2026-03 -> Mart 2026)
            tr_months = {
                "01": "Ocak", "02": "Şubat", "03": "Mart", "04": "Nisan",
                "05": "Mayıs", "06": "Haziran", "07": "Temmuz", "08": "Ağustos",
                "09": "Eylül", "10": "Ekim", "11": "Kasım", "12": "Aralık"
            }
            
            def format_month(ay_str):
                y, m = ay_str.split("-")
                return f"{tr_months[m]} {y}"

            disp_m_df = m_df[['Ay', 'price', 'predicted_price', 'Sapma (%)']].copy()
            disp_m_df['Ay'] = disp_m_df['Ay'].apply(format_month)
            disp_m_df.set_index('Ay', inplace=True) # Tarih damgasını (index) bununla değiştir
            
            disp_m_df.columns = ['Gerçekleşen Ort.', 'Model Tahmin Ort.', 'Hata (%)']
            st.dataframe(disp_m_df.style.format({
                'Gerçekleşen Ort.': "{:,.2f} ₺",
                'Model Tahmin Ort.': "{:,.2f} ₺",
                'Hata (%)': "%{:,.1f}"
            }), use_container_width=True)

            # Grafik (Line Chart olarak güncellendi)
            st.markdown("<br>", unsafe_allow_html=True)
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=m_df['Ay'], y=m_df['price'], name='Gerçekleşen Ort.', 
                                     line=dict(color='#4ECDC4', width=3, shape='spline'),
                                     mode='lines+markers'))
            fig_m.add_trace(go.Scatter(x=m_df['Ay'], y=m_df['predicted_price'], name='Model Tahmin Ort.', 
                                     line=dict(color='#FF6B6B', width=3, dash='dot', shape='spline'),
                                     mode='lines+markers'))
            
            fig_m.update_layout(**make_chart_layout(title="Aylık Ortalama PTF Trend Karşılaştırması"))
            st.plotly_chart(fig_m, use_container_width=True)
            
        except Exception as e:
            st.error(f"Aylık veriler hesaplanırken hata oluştu: {e}")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-header"><span class="icon">🔮</span><span class="text">Gelecek 6 Ay Projeksiyonu & Senaryo Analizi</span><span class="line"></span></div>""", unsafe_allow_html=True)
        
        # SENARYO SEÇİMLERİ
        with st.expander("🛠️ Senaryo Ayarları (Tahmini Özelleştir)", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                hydro_scenario = st.select_slider(
                    "💧 Hidrolojik Durum (Barajlar)",
                    options=["Yağışlı", "Normal", "Kurak"],
                    value="Normal",
                    help="Yağışlı havalarda HES üretimi artar ve fiyatlar düşer."
                )
            with col_s2:
                temp_scenario = st.select_slider(
                    "🌡️ Yaz Mevsimi Sıcaklığı",
                    options=["Normal", "Sıcak", "Aşırı Sıcak"],
                    value="Normal",
                    help="Aşırı sıcaklarda klima kullanımı artar ve fiyatlar yükselir."
                )
        
        # Senaryo Katsayıları
        hydro_coeff = {"Yağışlı": 0.85, "Normal": 1.0, "Kurak": 1.15}[hydro_scenario]
        temp_coeff = {"Normal": 1.0, "Sıcak": 1.05, "Aşırı Sıcak": 1.12}[temp_scenario]
        total_scenario_coeff = hydro_coeff * temp_coeff

        # 6 Aylık Projeksiyonu Hesapla
        try:
            future_months = []
            today = datetime.now()
            for i in range(1, 7):
                f_date = (today + pd.DateOffset(months=i))
                future_months.append(f_date)
            
            # Trend Hesaplama (2026 vs 2025 ilk 4 ay)
            recent_2026 = df_ptf[(df_ptf.index.year == 2026) & (df_ptf.index.month.isin([1,2,3,4]))]['price'].mean()
            past_2025 = df_ptf[(df_ptf.index.year == 2025) & (df_ptf.index.month.isin([1,2,3,4]))]['price'].mean()
            base_trend = recent_2026 / past_2025 if past_2025 > 0 else 1.0
            
            forecasts = []
            for m in future_months:
                # Geçen yılın aynı ayı
                ly_month = df_ptf[(df_ptf.index.year == 2025) & (df_ptf.index.month == m.month)]['price'].mean()
                if pd.isna(ly_month): ly_month = df_ptf['price'].mean() # Fallback
                
                # Projeksiyon = Geçen Yıl * Trend * Senaryo
                val = ly_month * base_trend * total_scenario_coeff
                forecasts.append({
                    'date': m.replace(day=1, hour=0, minute=0, second=0),
                    'Ay': m.strftime('%Y-%m'),
                    'price': val,
                    'Tip': 'Projeksiyon'
                })
            
            df_forecast = pd.DataFrame(forecasts)
            
            # Grafik İçin Birleştir
            past_plot = m_df.reset_index()[['date', 'Ay', 'price']].copy()
            past_plot['Tip'] = 'Gerçekleşen'
            
            full_plot_df = pd.concat([past_plot, df_forecast], ignore_index=True)
            full_plot_df['Ay_Etiket'] = full_plot_df['date'].apply(lambda x: f"{tr_months[x.strftime('%m')]} {x.year}")
            
            # Grafik
            fig_proj = px.line(full_plot_df, x='Ay_Etiket', y='price', color='Tip',
                               color_discrete_map={'Gerçekleşen': '#4ECDC4', 'Projeksiyon': '#FFD93D'},
                               markers=True, title="6 Aylık PTF Projeksiyonu")
            
            fig_proj.update_layout(**make_chart_layout())
            st.plotly_chart(fig_proj, use_container_width=True)
            
            # Bilgi Kartları
            st.markdown("<div style='display:flex; gap:15px; overflow-x:auto;'>", unsafe_allow_html=True)
            cols = st.columns(6)
            for i, row in df_forecast.iterrows():
                with cols[i]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding:10px; text-align:center; min-width:140px;">
                        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6)">{tr_months[row['Ay'].split('-')[1]]}</div>
                        <div style="font-size:1.1rem; font-weight:bold; color:#FFD93D">{row['price']:,.0f} ₺</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Projeksiyon hesaplanırken hata oluştu: {e}")

    # Footer
    st.markdown("""
    <div class="footer-text">
        Powered by XGBoost ML · EPİAŞ EPTR2 API · 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
