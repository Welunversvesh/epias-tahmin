---
title: EPIAS PTF Tahmin
emoji: ⚡
colorFrom: teal
colorTo: indigo
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
---

# EPIAS PTF Tahmin

Streamlit tabanli EPİAŞ PTF tahmin uygulamasi.

## Hugging Face Spaces Kurulumu

Space ayarlarinda asagidaki secret'lari tanimlayin:

- `EPIAS_USERNAME`
- `EPIAS_PASSWORD`

Uygulama `app.py` dosyasindan calisir. Model ve CSV dosyalari repo ile birlikte yuklenir.

Not: Hugging Face free CPU ortaminda dosya sistemi restart sonrasi kalici degildir. Uygulama icinden modeli yeniden egitirseniz sonuc calisan Space icinde kullanilir, fakat kalici model guncellemesi icin egitim islemini GitHub Actions gibi ayri bir pipeline'a tasimak daha dogrudur.
