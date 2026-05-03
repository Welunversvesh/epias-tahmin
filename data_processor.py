import pandas as pd
import numpy as np

def load_and_merge_data():
    print("[*] Temel Veriler yükleniyor (2025 ve 2026)...")
    
    def load_combined(prefix, cols=None, rename_dict=None):
        df_2025 = pd.read_csv(f"{prefix}_2025_saatlik_ham.csv" if prefix == 'ptf' else f"{prefix}_2025.csv")
        df_2026 = pd.read_csv(f"{prefix}_2026_saatlik_ham.csv" if prefix == 'ptf' else f"{prefix}_2026.csv")
        df = pd.concat([df_2025, df_2026], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        if cols:
            df = df[cols]
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        # Drop duplicates just in case
        df = df[~df.index.duplicated(keep='first')]
        return df
        
    df_ptf = load_combined('ptf', cols=['price'])
    df_load = load_combined('load_plan', cols=['lep'])
    df_kgup = load_combined('kgup', cols=['toplam', 'ruzgar', 'gunes', 'dogalgaz'], rename_dict={'toplam': 'planned_total_gen', 'dogalgaz': 'planned_gas_gen'})
    
    print("[*] Yeni Faz 2 Verileri yükleniyor (2025 ve 2026)...")
    df_smp = load_combined('smp', cols=['systemMarginalPrice'])
    df_rtg = load_combined('rtg', cols=['total'], rename_dict={'total': 'actual_total_gen'})
    df_pio = load_combined('pi_offer', cols=['offerVolume'], rename_dict={'offerVolume': 'price_independent_sales'})
    
    # Dogalgaz GRF Fiyati (Gunluk -> Saatlige cevirilir)
    print("[*] Dogalgaz referans fiyatlari yukleniyor...")
    try:
        df_gas_25 = pd.read_csv('gas_prices_2025.csv')
        df_gas_26 = pd.read_csv('gas_prices_2026.csv')
        df_gas = pd.concat([df_gas_25, df_gas_26], ignore_index=True)
        df_gas['date'] = pd.to_datetime(df_gas['date'], utc=True)
        df_gas.set_index('date', inplace=True)
        df_gas = df_gas[~df_gas.index.duplicated(keep='first')]
        # Gunluk veriyi saatlige yay (her saate ayni gun fiyati)
        df_gas = df_gas.resample('h').ffill()
        print(f"    [+] {len(df_gas)} satir dogalgaz verisi yuklendi.")
    except Exception as e:
        print(f"    [-] Dogalgaz verisi yuklenemedi: {e}")
        df_gas = pd.DataFrame()
    
    # Hava Durumu Verileri (Sicaklik, Ruzgar, Gunes)
    print("[*] Hava durumu verileri yukleniyor...")
    try:
        df_w_25 = pd.read_csv('weather_2025.csv')
        df_w_26 = pd.read_csv('weather_2026.csv')
        df_w = pd.concat([df_w_25, df_w_26], ignore_index=True)
        df_w['date'] = pd.to_datetime(df_w['date'], utc=True)
        df_w.set_index('date', inplace=True)
        df_w = df_w[~df_w.index.duplicated(keep='first')]
        print(f"    [+] {len(df_w)} satir hava durumu verisi yuklendi.")
    except Exception as e:
        print(f"    [-] Hava durumu verisi yuklenemedi: {e}")
        df_w = pd.DataFrame()

    # Yan Hizmetler (SFK/PFK) Verileri
    print("[*] Yan hizmetler verileri yukleniyor...")
    try:
        df_anc_25 = pd.read_csv('ancillary_2025.csv')
        df_anc_26 = pd.read_csv('ancillary_2026.csv')
        df_anc = pd.concat([df_anc_25, df_anc_26], ignore_index=True)
        df_anc['date'] = pd.to_datetime(df_anc['date'], utc=True)
        df_anc.set_index('date', inplace=True)
        df_anc = df_anc[~df_anc.index.duplicated(keep='first')]
        print(f"    [+] {len(df_anc)} satir yan hizmetler verisi yuklendi.")
    except Exception as e:
        print(f"    [-] Yan hizmetler verisi yuklenemedi: {e}")
        df_anc = pd.DataFrame()

    print("[*] Tum Tablolar birlestiriliyor...")
    df = df_ptf[['price']].join(df_load, how='inner') \
                          .join(df_kgup, how='inner') \
                          .join(df_smp, how='left') \
                          .join(df_rtg, how='left') \
                          .join(df_pio, how='left')
    
    if not df_gas.empty:
        df = df.join(df_gas, how='left')

    if not df_w.empty:
        df = df.join(df_w, how='left')

    if not df_anc.empty:
        df = df.join(df_anc, how='left')
                          
    # NaN olanlari ileriye/geriye donuk doldurma
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

def get_turkish_holidays(years=[2024, 2025, 2026]):
    holidays = []
    
    holidays_2026 = ['2026-01-01', '2026-03-20', '2026-03-21', '2026-03-22', '2026-04-23', '2026-05-01', 
                     '2026-05-19', '2026-05-27', '2026-05-28', '2026-05-29', '2026-05-30', '2026-07-15', 
                     '2026-08-30', '2026-10-29']
                     
    holidays_2025 = ['2025-01-01', '2025-03-30', '2025-03-31', '2025-04-01', '2025-04-23', '2025-05-01', 
                     '2025-05-19', '2025-06-06', '2025-06-07', '2025-06-08', '2025-06-09', '2025-07-15', 
                     '2025-08-30', '2025-10-29']
    
    holidays_2024 = ['2024-01-01', '2024-04-10', '2024-04-11', '2024-04-12', '2024-04-23', '2024-05-01',
                     '2024-05-19', '2024-06-16', '2024-06-17', '2024-06-18', '2024-06-19', '2024-07-15',
                     '2024-08-30', '2024-10-29']
                     
    holidays.extend(holidays_2026)
    holidays.extend(holidays_2025)
    holidays.extend(holidays_2024)
    return pd.to_datetime(holidays).date

def create_features(df):
    print("[*] Zaman ve Tatil özellikleri (Time & Holiday features) üretiliyor...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    tr_holidays = get_turkish_holidays()
    df['is_holiday'] = df.index.date
    df['is_holiday'] = df['is_holiday'].apply(lambda x: 1 if x in tr_holidays else 0)
    
    print("[*] Geçmiş Dengesizlik, Gecikme (Lag) ve Momentum özellikleri üretiliyor...")
    
    # 1. Gecikme (Lag) özellikleri
    df['price_lag_24'] = df['price'].shift(24)
    df['price_lag_48'] = df['price'].shift(48)
    df['price_lag_168'] = df['price'].shift(168)
    df['smp_lag_24'] = df['systemMarginalPrice'].shift(24)
    df['sfk_lag_24'] = df['sfk_price'].shift(24)
    df['pfk_lag_24'] = df['pfk_price'].shift(24)
    df['sfk_amount_lag_24'] = df['sfk_amount'].shift(24)
    df['pfk_amount_lag_24'] = df['pfk_amount'].shift(24)
    
    # 2. Plan vs Gerçekleşme Sapması (Dünden gelen sapma)
    # Gerçekleşen üretim - Planlanan üretim (Ne kadar saptık?)
    df['gen_deviation'] = df['actual_total_gen'] - df['planned_total_gen']
    df['gen_deviation_lag_24'] = df['gen_deviation'].shift(24)
    
    # 3. İstatistiksel Özellikler (Volatility ve Momentum)
    # Son 24 saatin fiyat oynaklığı (Standart Sapma)
    df['price_volatility_24h'] = df['price'].shift(24).rolling(window=24).std()
    
    # Momentum (Fiyat artış hızında mı? Dün vs Evvelsi gün)
    df['price_momentum_24h'] = df['price'].shift(24) - df['price'].shift(48)
    
    # Hareketli Ortalamalar (Rolling Means)
    df['price_rolling_24h_mean'] = df['price'].shift(24).rolling(window=24).mean()
    df['price_rolling_168h_mean'] = df['price'].shift(24).rolling(window=168).mean()
    
    # 4. Arz-Talep Dengesi ve Fiyattan Bağımsız Satış
    # lep (talep) / toplam (üretim planı)
    df['demand_supply_ratio'] = df['lep'] / (df['planned_total_gen'] + 1)
    
    # Fiyattan bağımsız satış oranı
    df['pi_offer_ratio'] = df['price_independent_sales'] / (df['planned_total_gen'] + 1)
    
    # 5. Dogalgaz Ozellikleri
    if 'gas_price' in df.columns:
        print("[*] Dogalgaz ozellikleri uretiliyor...")
        df['gas_price_lag_24'] = df['gas_price'].shift(24)
        # Elektrik/Gaz orani (Spark Spread benzeri)
        df['elec_gas_ratio'] = df['price'].shift(24) / (df['gas_price'].shift(24) + 1)
    
    # Geleceği sızdırmaması gereken (Leakage) sütunları silelim
    # Actual Generation (RTG) ve SMP o saat geldiğinde belli olur, bu yüzden güncel halleri modelde OLAMAZ!
    df.drop(['systemMarginalPrice', 'actual_total_gen', 'gen_deviation'], axis=1, inplace=True)
    
    # Eksik verileri (NaN) olan ilk satırları sil
    df.dropna(inplace=True)
    
    return df

def main():
    print("=== EPİAŞ Faz 2: Gelişmiş Özellik Mühendisliği ===")
    df = load_and_merge_data()
    df_features = create_features(df)
    
    print(f"\n[+] Veri başarıyla işlendi! Toplam {len(df_features)} saatlik kayıt eğitime hazır.")
    print("Örnek Veri (İlk 3 Satır):")
    # Çok fazla sütun olduğu için sadece yenileri gösterelim
    new_cols = ['price', 'price_independent_sales', 'is_holiday', 'gen_deviation_lag_24', 'price_volatility_24h', 'price_momentum_24h']
    print(df_features[new_cols].head(3))
    
    df_features.to_csv("model_ready_data.csv")
    print("\n[+] Gelişmiş model için hazır veri 'model_ready_data.csv' olarak kaydedildi.")

if __name__ == "__main__":
    main()
