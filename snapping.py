import pandas as pd
import numpy as np

def apply_price_snapping(df, price_col='predicted_price'):
    """
    XGBoost'un matematiksel olarak bulmaktan kaçındığı ekstrem uçları (0 ve Dinamik Tavan)
    Yük, Rüzgar, Güneş ve Bağımsız Satış rasyolarına bakarak kural tabanlı yapıştırır (Snap).
    """
    snapped_prices = []
    
    # Eşik Tarih: 4 Nisan 2026
    threshold_date = pd.to_datetime('2026-04-04').date()
    
    for idx, row in df.iterrows():
        price = row[price_col]
        lep = row.get('lep', 1)
        if lep == 0: lep = 1 # Div by zero koruması
        
        ruzgar = row.get('ruzgar', 0)
        gunes = row.get('gunes', 0)
        pi_sales = row.get('price_independent_sales', 0)
        
        ren_ratio = (ruzgar + gunes) / lep
        pi_ratio = pi_sales / lep
        
        # Tarihe göre Tavan Fiyatı Belirle
        if isinstance(idx, str):
            current_date = pd.to_datetime(idx).date()
        else:
            current_date = idx.date()
            
        if current_date >= threshold_date:
            ceiling_price = 4500.00
            high_price_threshold = 2800.00 
            snap_threshold = 3200.00
        else:
            ceiling_price = 3400.00
            high_price_threshold = 2600.00
            snap_threshold = 2900.00
        
        new_price = price
        
        # 1. SIFIRA YAPIŞTIRMA KURALI (Taban Fiyat)
        if price < 500:
            if price < 150:
                new_price = 0.00
            elif ren_ratio > 0.35:
                new_price = 0.00
                
        # 2. TAVANA YAPIŞTIRMA KURALI (Dinamik Tavan Fiyat)
        elif price > high_price_threshold:
            if price > snap_threshold:
                new_price = ceiling_price
            elif pi_ratio < 0.20:
                new_price = ceiling_price
        
        # Kesin Limitler
        if new_price < 0: new_price = 0.00
        if new_price > ceiling_price: new_price = ceiling_price
        
        snapped_prices.append(new_price)
        
    return snapped_prices
