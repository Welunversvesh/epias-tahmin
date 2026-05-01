import pandas as pd
import xgboost as xgb
import pickle

def load_model_and_predict():
    print("=== EPİAŞ PTF Ortalama Tahmin Aracı ===")
    
    # Model ve Özellik İsimlerini Yükle
    model = xgb.XGBRegressor()
    model.load_model("ptf_xgboost_model.json")
    
    with open("model_features.pkl", "rb") as f:
        features = pickle.load(f)
        
    # Veriyi Yükle (Gerçek senaryoda burası dünden çekilen veriler olacaktır)
    # Biz test verisi üzerinden (örneğin Aralık ayı) tahmini gösteriyoruz.
    df = pd.read_csv("model_ready_data.csv", index_col='date', parse_dates=True)
    
    # Tüm Aralık ayı (Aralık 2025) üzerinden tahmin yapalım
    df_predict = df[df.index >= "2025-12-01"].copy()
    X_predict = df_predict[features]
    
    print("[*] Tahminler Hesaplanıyor...\n")
    df_predict['predicted_price'] = model.predict(X_predict)
    
    # 1. ERTESİ GÜN (Örnek: 25 Aralık 2025)
    target_day = "2025-12-25"
    df_day = df_predict[df_predict.index.date == pd.to_datetime(target_day).date()]
    if not df_day.empty:
        daily_avg = df_day['predicted_price'].mean()
        print(f"[!] {target_day} Günü Tahmini Ortalama PTF: {daily_avg:.2f} TL/MWh")
    
    # 2. HAFTALIK ORTALAMA (Örnek: Aralık ayının son haftası)
    start_week = "2025-12-24"
    end_week = "2025-12-30"
    df_week = df_predict[(df_predict.index >= start_week) & (df_predict.index <= end_week)]
    if not df_week.empty:
        weekly_avg = df_week['predicted_price'].mean()
        print(f"[!] {start_week} ile {end_week} Arası Haftalık Ortalama PTF: {weekly_avg:.2f} TL/MWh")
        
    # 3. AYLIK ORTALAMA (Örnek: Aralık 2025'in tamamı)
    start_month = "2025-12-01"
    end_month = "2025-12-31"
    df_month = df_predict[(df_predict.index >= start_month) & (df_predict.index <= end_month)]
    if not df_month.empty:
        monthly_avg = df_month['predicted_price'].mean()
        actual_monthly_avg = df_month['price'].mean() # Gerçek değer (Karşılaştırma için)
        print(f"[!] {start_month[:7]} Ayı Tahmini Ortalama PTF: {monthly_avg:.2f} TL/MWh")
        print(f"    (Gerçekleşen Aralık 2025 Ortalaması: {actual_monthly_avg:.2f} TL/MWh)")
        
    print("\n[+] Model tahminlerini saatlik olarak görmek isterseniz, 'df_predict' tablosunu dışa aktarabilirsiniz.")
    df_predict[['price', 'predicted_price']].to_csv("aralik_tahmin_sonuclari.csv")
    print("[+] Tüm saatlik tahminler 'aralik_tahmin_sonuclari.csv' olarak kaydedildi.")

if __name__ == "__main__":
    load_model_and_predict()
