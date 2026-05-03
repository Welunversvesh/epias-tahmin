from eptr2 import EPTR2
import pandas as pd
from datetime import datetime
import os

def load_credentials():
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if username and password:
        return username, password

    with open("credentials.txt", "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines[0], lines[1]

def fetch_ancillary(start_date, end_date, filename, eptr=None):
    if eptr is None:
        username, password = load_credentials()
        eptr = EPTR2(username=username, password=password)
    print(f'[*] {start_date} - {end_date} arasi Yan Hizmetler cekiliyor...')
    
    # SFK Fiyat ve Miktar
    sfk_p = eptr.call('anc-sfk', start_date=start_date, end_date=end_date)
    sfk_p['date'] = pd.to_datetime(sfk_p['date'], utc=True)
    sfk_p = sfk_p.set_index('date')[['price']].rename(columns={'price': 'sfk_price'})
    
    sfk_q = eptr.call('anc-sf-qty', start_date=start_date, end_date=end_date)
    sfk_q['date'] = pd.to_datetime(sfk_q['date'], utc=True)
    sfk_q = sfk_q.set_index('date')[['amount']].rename(columns={'amount': 'sfk_amount'})

    # PFK Fiyat ve Miktar
    pfk_p = eptr.call('anc-pfk', start_date=start_date, end_date=end_date)
    pfk_p['date'] = pd.to_datetime(pfk_p['date'], utc=True)
    pfk_p = pfk_p.set_index('date')[['price']].rename(columns={'price': 'pfk_price'})
    
    pfk_q = eptr.call('anc-pf-qty', start_date=start_date, end_date=end_date)
    pfk_q['date'] = pd.to_datetime(pfk_q['date'], utc=True)
    pfk_q = pfk_q.set_index('date')[['amount']].rename(columns={'amount': 'pfk_amount'})
    
    combined = sfk_p.join(sfk_q, how='outer').join(pfk_p, how='outer').join(pfk_q, how='outer').ffill().fillna(0)
    combined.to_csv(filename)
    print(f'[+] {filename} kaydedildi.')

if __name__ == "__main__":
    fetch_ancillary('2025-01-01', '2025-12-31', 'ancillary_2025.csv')
    fetch_ancillary('2026-01-01', datetime.now().strftime('%Y-%m-%d'), 'ancillary_2026.csv')
