import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pickle

def train_and_evaluate():
    print("=== XGBoost PTF Tahmin Modeli Eğitimi ===")
    
    # 1. Veriyi Yükle
    print("[*] 'model_ready_data.csv' yükleniyor...")
    df = pd.read_csv("model_ready_data.csv", index_col='date', parse_dates=True)
    
    # 2. Özellikleri ve Hedefi Belirle
    target = 'price'
    exclude_cols = [target, 'sfk_price', 'pfk_price', 'sfk_amount', 'pfk_amount', 'actual_total_gen', 'gen_deviation']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"[*] Kullanılacak Özellikler ({len(features)} adet):")
    print(", ".join(features))
    
    # 3. Eğitim ve Test Seti Ayrımı (Zaman serisi mantığına uygun - sondan 30 gün test)
    # df.index zaten datetime tipinde
    test_days = 30
    test_cutoff = df.index.max() - pd.Timedelta(days=test_days)
    
    train_df = df[df.index <= test_cutoff]
    test_df = df[df.index > test_cutoff]
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    # 4. Model Tanımlama ve Eğitim
    print("\n[*] XGBoost Modeli eğitiliyor... (Bu işlem birkaç saniye sürebilir)")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,            
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,         
        reg_alpha=0.0,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    # Eval set early stopping için
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=100
    )
    
    # 5. Değerlendirme
    print("\n[*] Test seti üzerinde tahmin yapılıyor ve değerlendiriliyor...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print("================ SONUÇLAR ================")
    print(f"Ortalama Mutlak Hata (MAE): {mae:.2f} TL/MWh")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.2f} TL/MWh")
    print(f"Ortalama Yüzdesel Hata (MAPE): %{mape:.2f}")
    print("==========================================")
    
    # 6. Modeli Kaydetme
    model.save_model("ptf_xgboost_model.json")
    
    # Sütun isimlerini de kaydedelim ki predict ederken sıralama bozulmasın
    with open("model_features.pkl", "wb") as f:
        pickle.dump(features, f)
        
    print("\n[+] Model 'ptf_xgboost_model.json' olarak başarıyla kaydedildi!")
    print("[+] Kullanılacak özellik listesi 'model_features.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    train_and_evaluate()
