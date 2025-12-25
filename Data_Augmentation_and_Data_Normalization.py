# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:29:07 2025

@author: Huawei
"""

import torch
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import random

DATA_DIR = "D:\SAUDersler\Bilgisayar Mühendisliği Tasarımı\VeriSetleri\Preprocessed_Dataset"

# ImageNet normalizasyon değerleri
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_training_transforms():
    """
    Eğitim (Training) seti için dönüşümler.
    Burada veri çoğaltma (Augmentation) uygulanır.
    Amaç: Modeli zorlamak ve ezberlemeyi önlemek.
    """
    return transforms.Compose([
        # 1. Rastgele Kes ve Büyüt (Crop & Resize)
        # Resmin %80-%100 arası bir alanını seçip 224x224 yapar.
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        
        #GEOMETRİK Çogaltma
        transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle yatay çevir
        transforms.RandomVerticalFlip(p=0.5),   # %50 ihtimalle dikey çevir
        transforms.RandomRotation(degrees=90),  # Rastgele 90 dereceye kadar döndür
        
        # ADIM 3: RENK Çoğaltma  (Işık Değişimi)
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        
        # Formatı Tensör'e çevir (0-255 arasını 0-1 arasına çeker ve (C,H,W) yapar)
        transforms.ToTensor(),
        
        # ADIM 4: NORMALİZASYON
        transforms.Normalize(mean=MEAN, std=STD)
    ])

#Modelin başarısını (Accuracy/Loss) ölçerken veri sabit kalmalıdır.
def get_validation_transforms():
    """
    Test/Validasyon seti için dönüşümler.
    Burada rastgelelik Yoktur. Standartlaştırma vardır.
    """
    return transforms.Compose([
        # Test ederken rastgele kırpmayız, tam ortadan bakarız veya resize ederiz.
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def visualize_augmented_data():
    """
    Dataseti yükler, dönüşümleri uygular ve rastgele bir görüntüyü
    ekrana basarak kontrol etmemizi sağlar.
    """
    
    # 1. Klasör Kontrolü
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Belirtilen klasör bulunamadı: {DATA_DIR}")
        print("Lütfen DATA_DIR değişkenini kendi bilgisayarınıza göre düzeltin.")
        return

    # 2. Dataseti Yükle (Eğitim Kurallarıyla)
    # ImageFolder, klasör isimlerini otomatik olarak sınıf etiketi yapar.
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=get_training_transforms())
    
    print("--- Veri Seti Bilgisi ---")
    print(f"Dataset Yolu: {DATA_DIR}")
    print(f"Toplam Resim Sayısı: {len(full_dataset)}")
    print(f"Bulunan Sınıflar (Bakteriler): {full_dataset.classes}")
    print("-" * 30)
    
    # 3. Rastgele Bir Resim Seç
    random_idx = random.randint(0, len(full_dataset) - 1)
    img_tensor, label_id = full_dataset[random_idx] # Dönüşüm burada uygulanır!
    
    print(f"\nSeçilen İndeks: {random_idx}")
    print(f"Tensor Boyutu (Modelin Gördüğü): {img_tensor.shape}") 
    # Çıktı: torch.Size([3, 224, 224])
    
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
    plt.title(f"Sınıf: {class_name}\n(Augmentation Uygulandı: Dönmüş/Kesilmiş Olabilir)")
    plt.axis('off')
    plt.show()

# Test etmek için bir örnek:
if __name__ == "__main__":
    print("Dönüşüm kuralları tanımlandı.")
    train_transform = get_training_transforms()
    print(train_transform)
    visualize_augmented_data()