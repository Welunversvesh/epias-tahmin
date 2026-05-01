import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import io

st.set_page_config(page_title="EPİAŞ PTF Kahini", page_icon="⚡", layout="wide")

# CSS ile özel tasarım (Dark mode uyumlu, modern)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #1E1E2F;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border-left: 4px solid #4ECDC4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0B0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("ptf_xgboost_model.json")
    with open("model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_data():
    if os.path.exists("model_ready_data.csv"):
        df = pd.read_csv("model_ready_data.csv", index_col='date', parse_dates=True)
        return df
    return None

def main():
    st.markdown('<p class="main-header">⚡ EPİAŞ Yapay Zeka Tabanlı PTF Tahmin Sistemi</p>', unsafe_allow_html=True)
    st.markdown("XGBoost ile 16 aylık (2025-2026) fiyat, talep, SMF ve dengesizlik verileri üzerinden eğitilmiş gelişmiş tahminleme aracı.")
    st.divider()

    df = load_data()
    
    if df is None:
        st.error("Veri seti bulunamadı! Lütfen arkaplanda veri çekme ve işleme adımlarını tamamlayın.")
        return

    model, features = load_model()
    
    # Model predict all at once
    X = df[features]
    df['predicted_price'] = model.predict(X)
    
    # Kural Tabanlı Keskinleştirme (Snapping)
    import snapping
    import importlib
    importlib.reload(snapping)
    from snapping import apply_price_snapping
    df['predicted_price'] = apply_price_snapping(df)

    # Sekmeler Oluştur
    tab1, tab2, tab3 = st.tabs(["📊 Geçmiş Veri Analizi", "🔮 Gelecek Tahmini (GÖP)", "📅 Gelecek Ay Beklentisi"])
    
    with tab1:
        # Eski Sidebar ve Tarih Kodları (Sekme 1)
        st.sidebar.header("⚙️ Kontrol Paneli (Geçmiş)")
        
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        
        selected_dates = st.sidebar.date_input(
            "Geçmiş Tahminleri Göster",
            value=(max_date - pd.Timedelta(days=7), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = selected_dates[0]
            end_date = selected_dates[0]

        # Veriyi Filtrele
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            st.warning("Seçilen tarih aralığında veri bulunamadı.")
        else:
            # Metrikleri Hesapla
            actual_mean = filtered_df['price'].mean()
            predicted_mean = filtered_df['predicted_price'].mean()
            mae = abs(filtered_df['price'] - filtered_df['predicted_price']).mean()

            # Üst Göstergeler (Metrics)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Gerçekleşen Ortalama PTF</div>
                    <div class="metric-value">{actual_mean:,.2f} ₺</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #FF6B6B;">
                    <div class="metric-label">Yapay Zeka Tahmini Ortalama</div>
                    <div class="metric-value">{predicted_mean:,.2f} ₺</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #FFE66D;">
                    <div class="metric-label">Ortalama Sapma (MAE)</div>
                    <div class="metric-value">{mae:,.2f} ₺</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Etkileşimli Grafik
            st.subheader("📈 Gerçekleşen vs Tahmin Edilen Fiyatlar")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['price'], mode='lines', name='Gerçek PTF', line=dict(color='#4ECDC4', width=2)))
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['predicted_price'], mode='lines', name='Model Tahmini', line=dict(color='#FF6B6B', width=2, dash='dot')))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF'),
                xaxis=dict(showgrid=True, gridcolor='#333344'),
                yaxis=dict(showgrid=True, gridcolor='#333344', title="TL/MWh"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Veri Tablosu
            st.subheader("📋 Detaylı Saatlik Veriler")
            display_df = filtered_df[['price', 'predicted_price']].copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d %H:%00')
            display_df.columns = ['Gerçek PTF', 'Tahmin']
            
            st.dataframe(display_df.style.format("{:,.2f}"), use_container_width=True)
            
            # Excel İndirme Butonu
            st.markdown("<br>", unsafe_allow_html=True)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=True, sheet_name="Tahminler")
            
            st.download_button(
                label="📥 Tabloyu Excel Olarak İndir (.xlsx)",
                data=buffer.getvalue(),
                file_name="epias_gecmis_tahminler.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with tab2:
        st.subheader("🔮 Henüz Gerçekleşmemiş (Gelecek) Saatlerin Tahmini")
        st.info("EPİAŞ'ın yayınladığı Gün Öncesi Piyasası (GÖP) planlarını kullanarak, PTF'si henüz belli olmayan günleri tahmin eder.")
        
        target_date = st.date_input("Tahmin Edilecek Hedef Tarih (Örn: Bugün veya Yarın)")
        
        if st.button("EPİAŞ'tan Güncel Planı Çek ve Tahmin Üret ⚡"):
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
                            st.success(f"✅ {target_date} tarihi için planlar EPİAŞ'tan başarıyla çekildi ve PTF tahminleri üretildi!")
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #FF6B6B; margin-bottom: 20px;">
                            <div class="metric-label">Günlük Ortalama PTF Beklentisi</div>
                            <div class="metric-value">{res['predicted_price'].mean():,.2f} ₺</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=res.index, y=res['predicted_price'], mode='lines+markers', name='Beklenen PTF', line=dict(color='#FF6B6B', width=3)))
                        fig2.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(showgrid=True, gridcolor='#333344'),
                            yaxis=dict(showgrid=True, gridcolor='#333344', title="TL/MWh"),
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        st.subheader("📋 Saatlik Beklentiler ve Planlar")
                        res_display = res.copy()
                        res_display.index = res_display.index.strftime('%H:%00')
                        res_display.columns = ['Tahmini PTF', 'Yük (Tüketim) Planı', 'Üretim Planı', 'Rüzgar', 'Güneş', 'Bağımsız Satış']
                        st.dataframe(res_display.style.format("{:,.2f}"), use_container_width=True)
                        
                        # Gelecek Tahminini İndirme
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='openpyxl') as w:
                            res_display.to_excel(w, index=True, sheet_name="Gelecek Tahmin")
                        st.download_button("📥 Gelecek Tahminlerini Excel İndir", data=buf.getvalue(), file_name=f"tahmin_{target_date}.xlsx")
                        
                    else:
                        st.error(f"Hata: {msg}")
                except Exception as e:
                    st.error(f"Tahmin motoru çalıştırılırken bir hata oluştu: {str(e)}")
                    
    with tab3:
        st.subheader("📅 Uzun Vadeli (Aylık) Ortalama PTF Tahmini")
        st.info("Bu sekme, geçmişteki 16 aylık verinin genel trendini ve geçen yılın mevsimselliğini baz alarak gelecek ayların ortalama PTF değerini istatistiksel olarak tahmin eder.")
        
        # Aylık hesaplama için eksiksiz ham PTF verisini kullanıyoruz
        try:
            df_ptf_25 = pd.read_csv("ptf_2025_saatlik_ham.csv", parse_dates=['date'])
            df_ptf_26 = pd.read_csv("ptf_2026_saatlik_ham.csv", parse_dates=['date'])
            df_ptf = pd.concat([df_ptf_25, df_ptf_26])
            df_ptf.set_index('date', inplace=True)
            df_ptf.index = pd.to_datetime(df_ptf.index, utc=True)
            monthly_actual = df_ptf['price'].resample('ME').mean()
        except:
            monthly_actual = df['price'].resample('ME').mean()
            df_ptf = df
        
        col_month1, col_month2 = st.columns([1, 3])
        with col_month1:
            target_month = st.selectbox("Hedef Ay Seçin", ["Mayıs 2026", "Haziran 2026", "Temmuz 2026", "Ağustos 2026", "Eylül 2026", "Ekim 2026", "Kasım 2026", "Aralık 2026"])
            month_map = {"Mayıs": 5, "Haziran": 6, "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12}
            selected_month_num = month_map[target_month.split(" ")[0]]
            selected_year_num = int(target_month.split(" ")[1])
        
        if st.button("Aylık Trendi Hesapla 📈"):
            last_year_data = df_ptf[(df_ptf.index.year == selected_year_num-1) & (df_ptf.index.month == selected_month_num)]
            
            if last_year_data.empty:
                st.error("Modelin bu ayı tahmin edebilmesi için geçen yılın aynı ayına ait veriye ihtiyacı var (Örn: 2025 verisi bulunamadı).")
            else:
                last_year_mean = last_year_data['price'].mean()
                
                recent_2026 = df_ptf[(df_ptf.index.year == 2026) & (df_ptf.index.month.isin([1,2,3,4]))]['price'].mean()
                past_2025 = df_ptf[(df_ptf.index.year == 2025) & (df_ptf.index.month.isin([1,2,3,4]))]['price'].mean()
                
                trend_ratio = recent_2026 / past_2025 if past_2025 > 0 else 1.0
                
                forecast_mean = last_year_mean * trend_ratio
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Geçen Yıl Aynı Ay (2025)</div>
                        <div class="metric-value">{last_year_mean:,.2f} ₺</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    trend_dir = "Düşüş" if trend_ratio < 1 else "Artış"
                    trend_pct = abs((1 - trend_ratio) * 100)
                    trend_color = "#FF6B6B" if trend_ratio < 1 else "#4ECDC4"
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {trend_color};">
                        <div class="metric-label">Yıllık Trend Momentum</div>
                        <div class="metric-value">% {trend_pct:,.1f} {trend_dir}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #FFE66D;">
                        <div class="metric-label">Beklenen {target_month} Ortalaması</div>
                        <div class="metric-value">{forecast_mean:,.2f} ₺</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"📊 Aylık Ortalama PTF Gelişimi ve {target_month} Beklentisi")
                
                monthly_actual = monthly_actual.reset_index()
                monthly_actual['Ay'] = monthly_actual['date'].dt.strftime('%Y-%m')
                monthly_actual['Tip'] = 'Gerçekleşen'
                
                forecast_row = pd.DataFrame({
                    'date': [pd.to_datetime(f"{selected_year_num}-{selected_month_num:02d}-01").tz_localize('UTC')],
                    'Ay': [f"{selected_year_num}-{selected_month_num:02d}"],
                    'price': [forecast_mean],
                    'Tip': ['Beklenti']
                })
                
                plot_df = pd.concat([monthly_actual, forecast_row], ignore_index=True)
                plot_df.sort_values('date', inplace=True)
                
                fig3 = px.bar(plot_df, x='Ay', y='price', color='Tip', color_discrete_map={'Gerçekleşen': '#4ECDC4', 'Beklenti': '#FFE66D'})
                fig3.update_traces(texttemplate='%{y:.3s}', textposition='outside')
                fig3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#333344', title="TL/MWh"),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
