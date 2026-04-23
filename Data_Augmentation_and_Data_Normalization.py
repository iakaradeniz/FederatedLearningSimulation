import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# --- GÜNCEL KLASÖR YAPISI ---
# Yeni oluşturduğumuz fiziksel bölünmüş veri seti yolu
BASE_DIR = "/content/drive/MyDrive/Federated_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "Train_Data")
VAL_DIR = os.path.join(BASE_DIR, "Validation_Data")

# ImageNet normalizasyon değerleri
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_client_transforms(client_id):
    """
    Eğitim (Training) seti için İSTEMCİYE ÖZGÜ dönüşümler.
    """
    if client_id == 0:
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2)
    elif client_id == 1:
        jitter = transforms.ColorJitter(saturation=0.3, hue=0.1)
    else:
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        jitter,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def get_validation_transforms():
    """
    Validation/Test seti için standart dönüşümler.
    """
    return transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def visualize_federated_data(client_id=0, mode='train'):
    """
    Yeni klasör yapısına göre veriyi yükler ve görselleştirir.
    mode: 'train' (istemci verisi) veya 'val' (sunucu doğrulama verisi)
    """
    
    # 1. Klasör Yolunu Belirle
    if mode == 'train':
        # Örn: /Federated_Dataset/Train_Data/client_0
        target_path = os.path.join(TRAIN_DIR, f"client_{client_id}")
        transform = get_client_transforms(client_id)
        title_prefix = f"İstemci {client_id} (Eğitim Seti)"
    else:
        # Örn: /Federated_Dataset/Validation_Data
        target_path = VAL_DIR
        transform = get_validation_transforms()
        title_prefix = "Global Validation Seti"

    if not os.path.exists(target_path):
        print(f"HATA: Klasör bulunamadı: {target_path}")
        return

    # 2. Dataseti Yükle
    dataset = datasets.ImageFolder(root=target_path, transform=transform)
    
    print(f"--- {title_prefix} Bilgisi ---")
    print(f"Klasör Yolu: {target_path}")
    print(f"Sınıf Sayısı: {len(dataset.classes)}")
    print(f"Bu Klasördeki Toplam Resim: {len(dataset)}")
    print("-" * 30)
    
    # 3. Rastgele Bir Resim Seç ve Hazırla
    random_idx = random.randint(0, len(dataset) - 1)
    img_tensor, label_id = dataset[random_idx] 
    
    img_np = img_tensor.numpy().transpose((1, 2, 0)) 
    img_np = np.array(STD) * img_np + np.array(MEAN) 
    img_np = np.clip(img_np, 0, 1)
    
    # 4. Çizdir
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    class_name = dataset.classes[label_id]
    plt.title(f"{title_prefix}\nSınıf: {class_name}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Örnek 1: İstemci 1'in kendi fiziksel klasöründeki veriyi gör
    visualize_federated_data(client_id=1, mode='train')
    
    # Örnek 2: Sunucunun elindeki Validation verisini gör
    # visualize_federated_data(mode='val')