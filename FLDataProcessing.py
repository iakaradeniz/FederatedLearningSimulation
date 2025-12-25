# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import os
from PIL import Image
from tqdm import tqdm # İlerleme çubuğu görmek için

# Ayarlar
INPUT_DIR = "D:\SAUDersler\Bilgisayar Mühendisliği Tasarımı\VeriSetleri\DIBaS Bacterial Colony Dataset"  # Orijinal verinin yolu
OUTPUT_DIR = "D:\SAUDersler\Bilgisayar Mühendisliği Tasarımı\VeriSetleri\Preprocessed_Dataset" # Yeni verinin kaydedileceği yer
TARGET_SIZE = (256, 256) # Önden küçültme boyutu (Eğitimde 224'e crop yapacağız)

def preprocess_dataset():
    # Çıktı klasörü yoksa oluştur
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Tüm alt klasörleri (bakteri sınıflarını) gez
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                
                # Dosya yollarını ayarla
                file_path = os.path.join(root, file)
                
                # Sınıf klasörünü koru (Örn: Acinetobacter.baumanii)
                class_name = os.path.basename(root)
                target_class_dir = os.path.join(OUTPUT_DIR, class_name)
                
                if not os.path.exists(target_class_dir):
                    os.makedirs(target_class_dir)
                
                try:
                    # 1. Görüntüyü Aç
                    img = Image.open(file_path)
                    
                    # 2. RGB'ye Zorla (Güvelik amaçlı)
                    # CMYK, Grayscale, RGBA ne gelirse gelsin standart RGB olur.
                    img = img.convert('RGB')
                    
                    # 3. Yeniden Boyutlandır (Resize)
                    # LANCZOS filtresi yüksek kaliteli küçültme sağlar
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # 4. JPG veya PNG olarak kaydet
                    # JPG diskte az yer kaplar, PNG kayıpsızdır.
                    # Bu projede JPG (quality=90) yeterlidir ve çok hızlıdır.
                    save_name = os.path.splitext(file)[0] + ".jpg"
                    img.save(os.path.join(target_class_dir, save_name), 'JPEG', quality=95)
                    
                except Exception as e:
                    print(f"Hata oluşan dosya: {file_path} | Hata: {e}")

    print("Ön işleme tamamlandı! Veriler RGB formatında ve küçültülmüş olarak hazır.")

# Fonksiyonu çalıştır
preprocess_dataset()



