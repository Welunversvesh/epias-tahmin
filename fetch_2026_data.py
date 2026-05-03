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

def fetch_and_save_data_chunked(eptr, call_name, start_date, end_date, output_file):
    print(f"[*] '{call_name}' verisi {start_date} ile {end_date} arası indiriliyor...")
    
    # 3 aylık limit olduğu için parçalara böl
    dates = [start_date, "2026-03-31", "2026-06-30", end_date]
    dates = [d for d in dates if d <= end_date]
    if dates[-1] != end_date:
        dates.append(end_date)
    # Tekrarları kaldır ve sırala
    dates = sorted(list(set(dates)))
        
    all_data = []
    for i in range(len(dates)-1):
        s_date = dates[i]
        e_date = dates[i+1]
        print(f"    -> Çekiliyor: {s_date} - {e_date}")
        try:
            df = eptr.call(call_name, start_date=s_date, end_date=e_date)
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"    [-] '{call_name}' parçası hata verdi ({s_date}-{e_date}): {str(e)}")
            
    if not all_data:
        print(f"[-] '{call_name}' için hiç veri bulunamadı.")
        return False
        
    final_df = pd.concat(all_data, ignore_index=True)
    if 'date' in final_df.columns:
        final_df['date'] = pd.to_datetime(final_df['date'], utc=True)
        final_df.set_index('date', inplace=True)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        
    print(f"[+] Başarılı! '{call_name}' toplam veri çekildi ({len(final_df)} kayıt).")
    final_df.to_csv(output_file)
    print(f"    -> {output_file} olarak kaydedildi.")
    return True

def main():
    print("=== EPTR2 ile 2026 Verilerinin Çekilmesi ===")
    
    username, password = load_credentials()

    try:
        eptr = EPTR2(username=username, password=password)
        print("\n[!] Giriş başarılı! 2026 verileri çekilmeye başlanıyor...\n")
        
        start_date = "2026-01-01"
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        # 1. PTF
        fetch_and_save_data_chunked(eptr, "mcp", start_date, end_date, "ptf_2026_saatlik_ham.csv")
        
        # 2. Yük Tahmin
        fetch_and_save_data_chunked(eptr, "load-plan", start_date, end_date, "load_plan_2026.csv")
        
        # 3. KGUP
        fetch_and_save_data_chunked(eptr, "kgup", start_date, end_date, "kgup_2026.csv")
        
        # 4. SMF
        fetch_and_save_data_chunked(eptr, "smp", start_date, end_date, "smp_2026.csv")
        
        # 5. RTG
        fetch_and_save_data_chunked(eptr, "rt-gen", start_date, end_date, "rtg_2026.csv")
        
        # 6. PI Offer
        fetch_and_save_data_chunked(eptr, "pi-offer", start_date, end_date, "pi_offer_2026.csv")
        
        print("\n[+] İşlem tamamlandı. 2026 verileri başarıyla oluşturuldu.")
        
    except Exception as e:
        print(f"\n[-] Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
