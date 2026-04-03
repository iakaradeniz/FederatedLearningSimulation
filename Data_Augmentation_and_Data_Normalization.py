
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import random

DATA_DIR = "D:\\SAUDersler\\Bilgisayar Mühendisliği Tasarımı\\VeriSetleri\\Preprocessed_Dataset"

# ImageNet normalizasyon değerleri
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_client_transforms(client_id):
    """
    Eğitim (Training) seti için İSTEMCİYE ÖZGÜ dönüşümler.
    Geometrik kurallar (kesme, dönme, çevirme) sabit bırakılmış,
    renk/ışık (Jitter) kuralları her istemcinin kendi cihazını 
    simüle etmesi için farklılaştırılmıştır.
    """
    # 1. ADIM: İstemciye özel fotometrik (renk/ışık) dönüşüm kuralları
    if client_id == 0:
        # 1. Daha yüksek parlaklık ve kontrast dalgalanması
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2)
    elif client_id == 1:
        # 2. Daha yüksek renk tonu (hue) ve doygunluk değişimi
        jitter = transforms.ColorJitter(saturation=0.3, hue=0.1)
    else:
        # 3. Standart, daha hafif koşullar
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    # 2. ADIM: Tüm dönüşümleri birleştir (Geometrik + Özel Jitter)
    return transforms.Compose([
        # 1. Rastgele Kes ve Büyüt (Crop & Resize)
        # Resmin %80-%100 arası bir alanını seçip 224x224 yapar.
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        
        # GEOMETRİK Çoğaltma
        transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle yatay çevir
        transforms.RandomVerticalFlip(p=0.5),   # %50 ihtimalle dikey çevir
        transforms.RandomRotation(degrees=90),  # Rastgele 90 dereceye kadar döndür
        
        # RENK Çoğaltma (istemciye özel ışık değişimi)
        jitter,
        
        # Formatı Tensör'e çevir (0-255 arasını 0-1 arasına çeker ve (C,H,W) yapar)
        transforms.ToTensor(),
        
        # NORMALİZASYON
        transforms.Normalize(mean=MEAN, std=STD)
    ])
#Modelin başarısını ölçerken veri sabit kalmalıdır.
def get_validation_transforms():
    """
    Test/Validasyon seti için dönüşümler.
    Burada rastgelelik Yoktur. Standartlaştırma vardır.
    (Bu kısım aynı kalmıştır)
    """
    return transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def visualize_augmented_data(client_id=0):
    """
    Dataseti yükler, belirtilen istemciye ait dönüşümleri uygular ve 
    rastgele bir görüntüyü ekrana basarak kontrol etmemizi sağlar.
    """
    # 1. Klasör Kontrolü
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Belirtilen klasör bulunamadı: {DATA_DIR}")
        print("Lütfen DATA_DIR değişkenini kendi bilgisayarınıza göre düzeltin.")
        return

    # 2. Dataseti İstemciye Özel Kurallarla Yükle
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=get_client_transforms(client_id))
    
    print("--- Veri Seti Bilgisi ---")
    print(f"Dataset Yolu: {DATA_DIR}")
    print(f"Uygulanan İstemci ID: {client_id}")
    print(f"Toplam Resim Sayısı: {len(full_dataset)}")
    print("-" * 30)
    
    # 3. Rastgele Bir Resim Seç
    random_idx = random.randint(0, len(full_dataset) - 1)
    img_tensor, label_id = full_dataset[random_idx] 
    
    # 4. Görüntüyü İnsan Gözü İçin Geri Çevir (Denormalize)
    # Tensor (C, H, W) formatındadır, matplotlib (H, W, C) ister.
    img_np = img_tensor.numpy().transpose((1, 2, 0)) 
    
    # Normalizasyonu tersine çevir: (img * std) + mean
    mean = np.array(MEAN)
    std = np.array(STD)
    img_np = std * img_np + mean 
    
    # Değerleri 0 ile 1 arasına sabitle (Matematiksel taşmaları önlemek için)
    img_np = np.clip(img_np, 0, 1)
    
    # 5. Ekrana Çizdir
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    class_name = full_dataset.classes[label_id]
    plt.title(f"Sınıf: {class_name}\nİstemci {client_id} Augmentation Uygulandı")
    plt.axis('off')
    plt.show()

# Test etmek için bir örnek:
if __name__ == "__main__":
    print("Dönüşüm kuralları güncellendi (İstemciye Özel Yapı).")
    test_client = 1 # Test etmek istediğiniz hastanenin ID'sini buradan değiştirebilirsiniz
    train_transform = get_client_transforms(client_id=test_client)
    print(f"\nİstemci {test_client} için tanımlanan kurallar:\n{train_transform}")
    visualize_augmented_data(client_id=test_client)