import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from eptr2 import EPTR2
from datetime import datetime, timedelta
import pytz
from functools import lru_cache
from pathlib import Path

from snapping import apply_price_snapping

BASE_DIR = Path(__file__).resolve().parent
LOCAL_TZ = "Europe/Istanbul"

def local_date_mask(index, date_str):
    if index.tz is None:
        local_index = index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    else:
        local_index = index.tz_convert(LOCAL_TZ)
    return local_index.strftime('%Y-%m-%d') == date_str

def file_mtime(filename):
    path = BASE_DIR / filename
    return path.stat().st_mtime if path.exists() else None

def same_local_day(df, date_str):
    return df[local_date_mask(df.index, date_str)].copy()

def smart_fallback_day(df_past, target_date_str, columns):
    target_local_index = pd.date_range(
        target_date_str,
        periods=24,
        freq='h',
        tz=LOCAL_TZ
    ).tz_convert('UTC')
    result = pd.DataFrame(index=target_local_index)

    target_dt = pd.Timestamp(target_date_str)
    t_minus_7_str = (target_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    t7 = same_local_day(df_past, t_minus_7_str)
    if t7.empty:
        return result

    t7 = t7.reindex(columns=columns).astype(float)
    t7.index = target_local_index[:len(t7)]
    result = result.join(t7, how='left').astype(float)

    past = df_past.copy()
    local_index = past.index.tz_convert(LOCAL_TZ) if past.index.tz is not None else past.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    past["_local_hour"] = local_index.hour
    past["_local_dow"] = local_index.dayofweek

    target_dow = pd.Timestamp(target_date_str).dayofweek
    recent_3d = past[past.index >= past.index.max() - pd.Timedelta(days=3)]
    recent_14d = past[past.index >= past.index.max() - pd.Timedelta(days=14)]
    same_dow_28d = recent_14d[recent_14d["_local_dow"] == target_dow]

    for hour, idx in enumerate(target_local_index):
        for col in columns:
            values = []

            if col in result.columns and pd.notna(result.at[idx, col]):
                values.append((0.40, result.at[idx, col]))

            for frame, weight in [
                (recent_3d, 0.25),
                (same_dow_28d, 0.25),
                (recent_14d, 0.10),
            ]:
                if col not in frame.columns or frame.empty:
                    continue
                hourly_values = frame.loc[frame["_local_hour"] == hour, col].dropna()
                if not hourly_values.empty:
                    values.append((weight, hourly_values.mean()))

            if values:
                weight_sum = sum(weight for weight, _ in values)
                result.at[idx, col] = sum(weight * value for weight, value in values) / weight_sum

    result.ffill(inplace=True)
    result.bfill(inplace=True)
    return result[columns]

def hourly_weather_fallback(target_date_str):
    idx = pd.date_range(
        target_date_str,
        periods=24,
        freq='h',
        tz=LOCAL_TZ
    ).tz_convert('UTC')
    return pd.DataFrame({
        "temperature": [15] * 24,
        "wind_speed": [10] * 24,
        "solar_radiation": [0] * 24,
        "precipitation": [0] * 24,
        "cloud_cover": [50] * 24,
    }, index=idx)

@lru_cache(maxsize=1)
def load_model_and_features(model_mtime=None, features_mtime=None):
    model = xgb.XGBRegressor()
    model.load_model(str(BASE_DIR / "ptf_xgboost_model.json"))
    with open(BASE_DIR / "model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

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

def load_recent_raw_data(days=15):
    # En güncel ham verileri yükle (Sadece son N günü alarak bellek tasarrufu)
    def load_combined(prefix, cols=None, rename_dict=None):
        file_2025 = f"{prefix}_2025_saatlik_ham.csv" if prefix == 'ptf' else f"{prefix}_2025.csv"
        file_2026 = f"{prefix}_2026_saatlik_ham.csv" if prefix == 'ptf' else f"{prefix}_2026.csv"
        df_2025 = pd.read_csv(BASE_DIR / file_2025)
        df_2026 = pd.read_csv(BASE_DIR / file_2026)
        df = pd.concat([df_2025, df_2026], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        if cols:
            df = df[cols]
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        return df.tail(days * 24)
        
    df_ptf = load_combined('ptf', cols=['price'])
    df_load = load_combined('load_plan', cols=['lep'])
    df_kgup = load_combined('kgup', cols=['toplam', 'ruzgar', 'gunes', 'dogalgaz'], rename_dict={'toplam': 'planned_total_gen', 'dogalgaz': 'planned_gas_gen'})
    df_smp = load_combined('smp', cols=['systemMarginalPrice'])
    df_rtg = load_combined('rtg', cols=['total'], rename_dict={'total': 'actual_total_gen'})
    df_pio = load_combined('pi_offer', cols=['offerVolume'], rename_dict={'offerVolume': 'price_independent_sales'})
    
    # Dogalgaz fiyati
    try:
        df_gas_25 = pd.read_csv(BASE_DIR / 'gas_prices_2025.csv')
        df_gas_26 = pd.read_csv(BASE_DIR / 'gas_prices_2026.csv')
        df_gas = pd.concat([df_gas_25, df_gas_26], ignore_index=True)
        df_gas['date'] = pd.to_datetime(df_gas['date'], utc=True)
        df_gas.set_index('date', inplace=True)
        df_gas = df_gas[~df_gas.index.duplicated(keep='first')]
        df_gas = df_gas.resample('h').ffill()
    except:
        df_gas = pd.DataFrame()
    
    # Yan Hizmetler
    try:
        df_anc_25 = pd.read_csv(BASE_DIR / 'ancillary_2025.csv')
        df_anc_26 = pd.read_csv(BASE_DIR / 'ancillary_2026.csv')
        df_anc = pd.concat([df_anc_25, df_anc_26], ignore_index=True)
        df_anc['date'] = pd.to_datetime(df_anc['date'], utc=True)
        df_anc.set_index('date', inplace=True)
        # Sadece fiyatlari alalim, miktarlar modelde yok
        df_anc = df_anc[['sfk_price', 'pfk_price']]
        df_anc = df_anc[~df_anc.index.duplicated(keep='first')]
    except:
        df_anc = pd.DataFrame()
    
    df = df_ptf.join(df_load, how='outer') \
               .join(df_kgup, how='outer') \
               .join(df_smp, how='outer') \
               .join(df_rtg, how='outer') \
               .join(df_pio, how='outer')
    
    if not df_gas.empty:
        df = df.join(df_gas, how='left')

    if not df_anc.empty:
        df = df.join(df_anc, how='left')
               
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def fetch_future_plan(target_date_str):
    import streamlit as st
    username = None
    password = None
    try:
        username = st.secrets["epias"]["username"]
        password = st.secrets["epias"]["password"]
    except Exception:
        username = None
        password = None

    username = username or os.getenv("EPIAS_USERNAME")
    password = password or os.getenv("EPIAS_PASSWORD")

    if not username or not password:
        with open(BASE_DIR / "credentials.txt", "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            username = lines[0]
            password = lines[1]
    
    eptr = EPTR2(username=username, password=password)
    
    plans = {'load': pd.DataFrame(), 'kgup': pd.DataFrame(), 'pio': pd.DataFrame()}
    
    try:
        df_load = eptr.call("load-plan", start_date=target_date_str, end_date=target_date_str)
        if not df_load.empty:
            df_load['date'] = pd.to_datetime(df_load['date'], utc=True)
            df_load.set_index('date', inplace=True)
            plans['load'] = df_load[['lep']]
    except Exception as e:
        print(f"Load Plan hatası: {e}")
        
    try:
        df_kgup = eptr.call("kgup", start_date=target_date_str, end_date=target_date_str)
        if not df_kgup.empty:
            df_kgup['date'] = pd.to_datetime(df_kgup['date'], utc=True)
            df_kgup.set_index('date', inplace=True)
            df_kgup = df_kgup[~df_kgup.index.duplicated(keep='first')]
            plans['kgup'] = df_kgup[['toplam', 'ruzgar', 'gunes', 'dogalgaz']].rename(columns={'toplam': 'planned_total_gen', 'dogalgaz': 'planned_gas_gen'})
    except Exception as e:
        print(f"KGUP hatası: {e}")
        
    try:
        df_pio = eptr.call("pi-offer", start_date=target_date_str, end_date=target_date_str)
        if not df_pio.empty:
            df_pio['date'] = pd.to_datetime(df_pio['date'], utc=True)
            df_pio.set_index('date', inplace=True)
            plans['pio'] = df_pio[['offerVolume']].rename(columns={'offerVolume': 'price_independent_sales'})
    except Exception as e:
        print(f"PI Offer hatası: {e}")
        
    return plans

def predict_future_day(target_date_str):
    import requests
    df_past = load_recent_raw_data(days=15)
    plans = fetch_future_plan(target_date_str)
    
    # Hava Durumu Tahmini (Target Date için Open-Meteo Forecast)
    print(f"[*] {target_date_str} için hava durumu tahmini cekiliyor...")
    try:
        w_url = "https://api.open-meteo.com/v1/forecast"
        w_params = {
            "latitude": 39.93, "longitude": 32.86, # Ankara temsilci
            "start_date": target_date_str, "end_date": target_date_str,
            "hourly": "temperature_2m,windspeed_10m,direct_radiation,precipitation,cloudcover",
            "timezone": LOCAL_TZ
        }
        response = requests.get(w_url, params=w_params, timeout=10)
        response.raise_for_status()
        w_res = response.json()
        hourly = w_res.get("hourly", {})
        required_weather_cols = [
            "temperature_2m", "windspeed_10m", "direct_radiation",
            "precipitation", "cloudcover"
        ]
        if not all(col in hourly for col in required_weather_cols):
            raise ValueError("Open-Meteo yanıtında beklenen saatlik alanlar yok.")

        df_weather = pd.DataFrame({
            "temperature": hourly["temperature_2m"],
            "wind_speed": hourly["windspeed_10m"],
            "solar_radiation": hourly["direct_radiation"],
            "precipitation": hourly["precipitation"],
            "cloud_cover": hourly["cloudcover"]
        })
        if "time" in hourly:
            df_weather.index = pd.to_datetime(hourly["time"]).tz_localize(LOCAL_TZ).tz_convert('UTC')
        else:
            df_weather.index = hourly_weather_fallback(target_date_str).index
        df_weather = df_weather.reindex(hourly_weather_fallback(target_date_str).index).ffill().bfill()
    except Exception as e:
        print(f"[-] Hava durumu tahmini cekilemedi: {e}")
        df_weather = hourly_weather_fallback(target_date_str)
    
    target_dt = pd.Timestamp(target_date_str)
    t_minus_7 = target_dt - pd.Timedelta(days=7)
    t_minus_7_str = t_minus_7.strftime('%Y-%m-%d')
    df_t7 = same_local_day(df_past, t_minus_7_str)
    
    if df_t7.empty:
        return None, False, "T-7 referans verisi bulunamadığı için tahmin yapılamıyor."
        
    target_index = pd.date_range(
        target_date_str,
        periods=24,
        freq='h',
        tz=LOCAL_TZ
    ).tz_convert('UTC')
    df_t7 = df_t7.reindex(columns=df_past.columns)
    df_t7.index = target_index[:len(df_t7)]
    df_future = pd.DataFrame(index=df_t7.index)
    
    simulated_parts = []
    
    if not plans['load'].empty:
        df_future['lep'] = plans['load']['lep'].reindex(df_future.index).ffill().bfill()
    else:
        load_fallback = smart_fallback_day(df_past, target_date_str, ['lep'])
        df_future['lep'] = load_fallback['lep'].reindex(df_future.index).ffill().bfill()
        simulated_parts.append("Yük (akıllı)")
        
    if not plans['kgup'].empty:
        for col in ['planned_total_gen', 'ruzgar', 'gunes', 'planned_gas_gen']:
            df_future[col] = plans['kgup'][col].reindex(df_future.index).ffill().bfill()
    else:
        kgup_cols = ['planned_total_gen', 'ruzgar', 'gunes', 'planned_gas_gen']
        kgup_fallback = smart_fallback_day(df_past, target_date_str, kgup_cols)
        for col in kgup_cols:
            df_future[col] = kgup_fallback[col].reindex(df_future.index).ffill().bfill()
        simulated_parts.append("KGÜP (akıllı)")
    
    # Hava durumunu ekle
    df_future['temperature'] = df_weather['temperature']
    df_future['wind_speed'] = df_weather['wind_speed']
    df_future['solar_radiation'] = df_weather['solar_radiation']
    df_future['precipitation'] = df_weather['precipitation']
    df_future['cloud_cover'] = df_weather['cloud_cover']
        
    if not plans['pio'].empty:
        df_future['price_independent_sales'] = plans['pio']['price_independent_sales'].reindex(df_future.index).ffill().bfill()
    else:
        pio_fallback = smart_fallback_day(df_past, target_date_str, ['price_independent_sales'])
        df_future['price_independent_sales'] = pio_fallback['price_independent_sales'].reindex(df_future.index).ffill().bfill()
        simulated_parts.append("Satış (akıllı)")
        
    is_simulated = len(simulated_parts) > 0
    sim_msg = f"Simülasyon: {', '.join(simulated_parts)}" if is_simulated else "Tüm Veriler Orijinal (EPİAŞ)"

    # 3. Geçmiş ve Geleceği birleştir
    df_combined = pd.concat([df_past, df_future])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    
    # İleriye doğru doldurma (SMP ve RTG'nin o güne ait bilinmeyen kısımlarını son saatten ffill yaparız)
    df_combined.ffill(inplace=True)
    
    # 4. Özellikleri Yarat
    df = df_combined.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    tr_holidays = get_turkish_holidays()
    df['is_holiday'] = df.index.date
    df['is_holiday'] = df['is_holiday'].apply(lambda x: 1 if x in tr_holidays else 0)
    
    df['price_lag_24'] = df['price'].shift(24)
    df['price_lag_48'] = df['price'].shift(48)
    df['price_lag_168'] = df['price'].shift(168)
    df['smp_lag_24'] = df['systemMarginalPrice'].shift(24)
    df['sfk_lag_24'] = df['sfk_price'].shift(24)
    df['pfk_lag_24'] = df['pfk_price'].shift(24)
    
    df['gen_deviation'] = df['actual_total_gen'] - df['planned_total_gen']
    df['gen_deviation_lag_24'] = df['gen_deviation'].shift(24)
    
    df['price_volatility_24h'] = df['price'].shift(24).rolling(window=24).std()
    df['price_momentum_24h'] = df['price'].shift(24) - df['price'].shift(48)
    df['price_rolling_24h_mean'] = df['price'].shift(24).rolling(window=24).mean()
    df['price_rolling_168h_mean'] = df['price'].shift(24).rolling(window=168).mean()
    
    df['demand_supply_ratio'] = df['lep'] / (df['planned_total_gen'] + 1)
    df['pi_offer_ratio'] = df['price_independent_sales'] / (df['planned_total_gen'] + 1)
    
    # Dogalgaz ozellikleri
    if 'gas_price' in df.columns:
        df['gas_price_lag_24'] = df['gas_price'].shift(24)
        df['elec_gas_ratio'] = df['price'].shift(24) / (df['gas_price'].shift(24) + 1)
    
    # Sızdırma sütunlarını sil (Gerçek hayatta sadece model özellikleri kalmalı)
    df.drop(['systemMarginalPrice', 'actual_total_gen', 'gen_deviation'], axis=1, inplace=True)
    
    # Sadece hedef günü filtrele
    df_target = df[local_date_mask(df.index, target_date_str)].copy()
    
    if len(df_target) == 0:
        return None, False, "Hedef gün için özellik oluşturulamadı."
        
    # 5. Modeli Yükle ve Tahmin Et
    model, features = load_model_and_features(
        file_mtime("ptf_xgboost_model.json"),
        file_mtime("model_features.pkl")
    )
    # --- KESİN TEMİZLİK (Sinsi sütunları sil) ---
    for col in ['sfk_amount', 'pfk_amount', 'sfk_amount_lag_24', 'pfk_amount_lag_24']:
        if col in df_target.columns:
            df_target.drop(col, axis=1, inplace=True)
    # --------------------------------------------
    # Modelin bekledigi ama veride olmayan bir sey var mi?
    missing = [c for c in features if c not in df_target.columns]
    if missing:
        raise Exception(f"Model {missing} sutunlarini bekliyor ama veride yok!")

    X = df_target[features]
    predictions = model.predict(X)
    df_target['predicted_price'] = predictions
    
    # Kural Tabanlı Keskinleştirme (Snapping)
    df_target['predicted_price'] = apply_price_snapping(df_target)
    
    cols_to_return = [
        'predicted_price', 'lep', 'planned_total_gen', 'planned_gas_gen', 
        'ruzgar', 'gunes', 'temperature', 'wind_speed', 
        'precipitation', 'cloud_cover', 'price_independent_sales'
    ]
    # Sadece var olanlari dondur
    final_cols = [c for c in cols_to_return if c in df_target.columns]
    
    return df_target[final_cols], is_simulated, sim_msg

if __name__ == "__main__":
    # Test
    target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Yarın ({target}) için tahmin test ediliyor...")
    res, sim, msg = predict_future_day(target)
    if res is not None:
        print(f"Simüle edildi mi?: {sim}")
        print(res.head())
    else:
        print(msg)
