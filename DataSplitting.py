import os
import shutil
import random

# --- AYARLAR ---
SOURCE_DIR = "/content/drive/MyDrive/Preprocessed_Dataset" # Eski veri setiniz
TARGET_DIR = "/content/drive/MyDrive/Federated_Dataset"    # Yeni oluşacak klasör

VAL_RATIO = 0.15 # Doğrulama için ayrılacak oran (%15)
CLIENT_RATIOS = {
    "client_0": 0.50, # 1. Hastane
    "client_1": 0.35, # 2. Hastane
    "client_2": 0.15  # 3. Hastane
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def physical_federated_split():
    print("Federatif Veri Dağıtımı Başlıyor...\n")
    
    # 1. Ana hedef klasörleri oluştur
    val_dir = os.path.join(TARGET_DIR, "Validation_Data")
    train_dir = os.path.join(TARGET_DIR, "Train_Data")
    create_dir(val_dir)
    
    client_dirs = {}
    for c_name in CLIENT_RATIOS.keys():
        c_path = os.path.join(train_dir, c_name)
        create_dir(c_path)
        client_dirs[c_name] = c_path

    # Kaynak klasördeki bakteri türlerini (sınıfları) bul
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_moved = 0
    
    for cls in classes:
        # Her bakteri sınıfı (klasörü) için...
        cls_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(cls_path)
        
        # Rastgele karıştır ki adil dağılsın!
        random.seed(42) # Her çalıştırmada aynı dağılımı vermesi için sabitliyoruz
        random.shuffle(images)
        
        # 1. Validation Ayrımı
        val_count = int(len(images) * VAL_RATIO)
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        # Validation resimlerini kopyala
        create_dir(os.path.join(val_dir, cls))
        for img in val_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
            total_moved += 1
            
        # 2. Train Verisini İstemcilere Dağıtma (%50, %35, %15)
        c0_count = int(len(train_images) * CLIENT_RATIOS["client_0"])
        c1_count = int(len(train_images) * CLIENT_RATIOS["client_1"])
        
        c0_images = train_images[:c0_count]
        c1_images = train_images[c0_count : c0_count+c1_count]
        c2_images = train_images[c0_count+c1_count :] # Geriye kalanlar
        
        client_splits = {
            "client_0": c0_images,
            "client_1": c1_images,
            "client_2": c2_images
        }
        
        # İstemci resimlerini ilgili klasörlere kopyala
        for c_name, c_images in client_splits.items():
            c_target_path = os.path.join(client_dirs[c_name], cls)
            create_dir(c_target_path)
            for img in c_images:
                shutil.copy(os.path.join(cls_path, img), os.path.join(c_target_path, img))
                total_moved += 1
                
        print(f"[{cls}] sınıfı başarıyla dağıtıldı. (Val: {len(val_images)}, C0: {len(c0_images)}, C1: {len(c1_images)}, C2: {len(c2_images)})")

    print(f"\nDağıtım Tamamlandı! Toplam {total_moved} adet resim fiziksel olarak kopyalandı.")
    print(f"Yeni Veri Seti Yolunuz: {TARGET_DIR}")

# Çalıştır
if __name__ == "__main__":
    physical_federated_split()