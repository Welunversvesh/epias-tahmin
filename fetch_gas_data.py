"""Dogalgaz GRF fiyatlarini EPİAŞ'tan çekip CSV olarak kaydeder."""
import pandas as pd
from eptr2 import EPTR2
import os

def load_credentials():
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if username and password:
        return username, password

    with open("credentials.txt", "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines[0], lines[1]

def fetch_gas_prices(eptr, start_date, end_date, output_file):
    print(f"[*] Dogalgaz fiyatlari {start_date} - {end_date} arasi cekiliyor...")
    
    # 3 aylik parcalara bol
    date_range = pd.date_range(start_date, end_date, freq='3ME')
    dates = [start_date] + [d.strftime('%Y-%m-%d') for d in date_range] + [end_date]
    dates = sorted(list(set(dates)))
    
    all_data = []
    for i in range(len(dates) - 1):
        s, e = dates[i], dates[i+1]
        print(f"    -> {s} - {e}")
        try:
            df = eptr.call('ng-spot-prices', start_date=s, end_date=e)
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as ex:
            print(f"    [-] Hata: {ex}")
    
    if not all_data:
        print("[-] Hic veri bulunamadi!")
        return False
    
    final = pd.concat(all_data, ignore_index=True)
    final['date'] = pd.to_datetime(final['gasDay'], utc=True)
    final.set_index('date', inplace=True)
    final = final[~final.index.duplicated(keep='first')]
    final = final[['weightedAverage']].rename(columns={'weightedAverage': 'gas_price'})
    final.sort_index(inplace=True)
    
    print(f"[+] {len(final)} gun dogalgaz fiyati cekildi.")
    final.to_csv(output_file)
    print(f"    -> {output_file} kaydedildi.")
    return True

if __name__ == "__main__":
    username, password = load_credentials()
    eptr = EPTR2(username=username, password=password)
    
    fetch_gas_prices(eptr, "2025-01-01", "2025-12-31", "gas_prices_2025.csv")
    fetch_gas_prices(eptr, "2026-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"), "gas_prices_2026.csv")
    print("\n[+] Tamamlandi!")
