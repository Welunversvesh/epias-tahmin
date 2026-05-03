"""Hava durumu (Open-Meteo) ve KGUP dogalgaz verilerini ceker."""
import pandas as pd
import requests
import os

def fetch_weather_data(start_date, end_date, output_file):
    """Open-Meteo API ile Turkiye geneli hava durumu verisi cekilir."""
    print(f"[*] Hava durumu verisi {start_date} - {end_date} arasi cekiliyor (Open-Meteo)...")
    
    # Turkiye'nin enerji agirligi olan 3 bolgesinin ortalamasi
    locations = [
        (39.93, 32.86, "Ankara"),     # Ic Anadolu - Sicaklik/Talep merkezi
        (39.66, 27.88, "Balikesir"),   # Marmara - Ruzgar yogun bolge
        (37.75, 30.28, "Burdur"),      # Gunes kusagi
    ]
    
    all_dfs = []
    for lat, lon, name in locations:
        print(f"    -> {name} ({lat}, {lon})...")
        try:
            r = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
                'latitude': lat, 'longitude': lon,
                'start_date': start_date, 'end_date': end_date,
                'hourly': 'temperature_2m,windspeed_10m,direct_radiation,precipitation,cloudcover',
                'timezone': 'Europe/Istanbul'
            }, timeout=30)
            data = r.json()
            
            df = pd.DataFrame({
                'time': data['hourly']['time'],
                f'temp_{name}': data['hourly']['temperature_2m'],
                f'wind_{name}': data['hourly']['windspeed_10m'],
                f'solar_{name}': data['hourly']['direct_radiation'],
                f'precip_{name}': data['hourly']['precipitation'],
                f'cloud_{name}': data['hourly']['cloudcover'],
            })
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            all_dfs.append(df)
        except Exception as e:
            print(f"    [-] {name} hatasi: {e}")
    
    if not all_dfs:
        print("[-] Hic veri cekilemedi!")
        return False
    
    # Tum lokasyonlari birlestir
    combined = pd.concat(all_dfs, axis=1)
    
    # Ortalamalarini al (Turkiye geneli temsili deger)
    temp_cols = [c for c in combined.columns if c.startswith('temp_')]
    wind_cols = [c for c in combined.columns if c.startswith('wind_')]
    solar_cols = [c for c in combined.columns if c.startswith('solar_')]
    precip_cols = [c for c in combined.columns if c.startswith('precip_')]
    cloud_cols = [c for c in combined.columns if c.startswith('cloud_')]
    
    result = pd.DataFrame({
        'temperature': combined[temp_cols].mean(axis=1),
        'wind_speed': combined[wind_cols].mean(axis=1),
        'solar_radiation': combined[solar_cols].mean(axis=1),
        'precipitation': combined[precip_cols].mean(axis=1),
        'cloud_cover': combined[cloud_cols].mean(axis=1),
    })
    
    result.index.name = 'date'
    # UTC+3 -> UTC
    result.index = result.index.tz_localize('Europe/Istanbul').tz_convert('UTC')
    result = result[~result.index.duplicated(keep='first')]
    
    print(f"[+] {len(result)} saatlik hava durumu verisi cekildi.")
    result.to_csv(output_file)
    print(f"    -> {output_file} kaydedildi.")
    return True

if __name__ == "__main__":
    fetch_weather_data("2025-01-01", "2025-12-31", "weather_2025.csv")
    fetch_weather_data("2026-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"), "weather_2026.csv")
    print("\n[+] Tamamlandi!")
