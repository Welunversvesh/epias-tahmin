import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pickle
import json
from datetime import datetime, timezone

def safe_mape(y_true, y_pred, min_actual=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) >= min_actual
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def train_and_evaluate():
    print("=== XGBoost PTF Tahmin Modeli Eğitimi ===")
    
    # 1. Veriyi Yükle
    print("[*] 'model_ready_data.csv' yükleniyor...")
    df = pd.read_csv("model_ready_data.csv", index_col='date', parse_dates=True)
    
    # 2. Özellikleri ve Hedefi Belirle
    target = 'price'
    exclude_cols = [target, 'sfk_price', 'pfk_price', 'actual_total_gen', 'gen_deviation']
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
    raw_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mape = safe_mape(y_test, y_pred)
    
    print("================ SONUÇLAR ================")
    print(f"Ortalama Mutlak Hata (MAE): {mae:.2f} TL/MWh")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.2f} TL/MWh")
    if mape is not None:
        print(f"Ortalama Yüzdesel Hata (MAPE, PTF >= 1): %{mape:.2f}")
    else:
        print("Ortalama Yüzdesel Hata (MAPE): Hesaplanamadı")
    print("==========================================")
    
    # 6. Modeli Kaydetme
    model.save_model("ptf_xgboost_model.json")
    
    # Sütun isimlerini de kaydedelim ki predict ederken sıralama bozulmasın
    with open("model_features.pkl", "wb") as f:
        pickle.dump(features, f)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "data_start": df.index.min().isoformat(),
        "data_end": df.index.max().isoformat(),
        "test_start": test_df.index.min().isoformat() if not test_df.empty else None,
        "test_end": test_df.index.max().isoformat() if not test_df.empty else None,
        "feature_count": len(features),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if mape is not None else None,
        "raw_mape": float(raw_mape),
        "mape_note": "MAPE, gerçek PTF değeri 1 TL/MWh ve üzerindeki saatler için hesaplandı.",
    }
    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    print("\n[+] Model 'ptf_xgboost_model.json' olarak başarıyla kaydedildi!")
    print("[+] Kullanılacak özellik listesi 'model_features.pkl' olarak kaydedildi.")
    print("[+] Eğitim metrikleri 'model_metrics.json' olarak kaydedildi.")

if __name__ == "__main__":
    train_and_evaluate()
