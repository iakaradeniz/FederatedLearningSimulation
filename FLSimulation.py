import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# MobileViT için gerekli kütüphane (pip install timm)
try:
    import timm
except ImportError:
    print("HATA: 'timm' kütüphanesi eksik. Lütfen terminale 'pip install timm' yazarak kurun.")
    exit()

# Dışarıdan ön işleme fonksiyonlarını çek (İstemciye Özel ve Validasyon)
from Data_Augmentation_and_Data_Normalization import get_client_transforms, get_validation_transforms

# --- AYARLAR VE PARAMETRELER ---
DATA_DIR = "/content/drive/MyDrive/Preprocessed_Dataset"
NUM_ROUNDS = 180      # Toplam federatif eğitim turu
LOCAL_EPOCHS = 1      # Her istemcinin kendi verisinde yapacağı tur sayısı
BATCH_SIZE = 32       # Bir kerede işlenecek görüntü sayısı
LEARNING_RATE = 0.001 # Sabit öğrenme oranı
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---MODELLER---
MODELS_TO_RUN = ['ResNet18', 'MobileNetV2', 'DenseNet121', 'MobileViT']
CLIENT_DATA_RATIOS = [0.50, 0.35, 0.15] # Quantity Skew (Asimetrik Veri Dağılımı)
NUM_CLIENTS = len(CLIENT_DATA_RATIOS)
FEDPROX_MU = 0.01     # FedProx ceza parametresi


def prepare_data_non_iid():
    """ Veriyi asimetrik oranlarda böler ve her istemciye kendi özel augmentation kuralını atar """
    # Sadece boyutu öğrenmek için standart bir yükleme yap
    temp_dataset = datasets.ImageFolder(root=DATA_DIR)
    dataset_size = len(temp_dataset)
    num_classes = len(temp_dataset.classes)
    
    val_size = int(dataset_size * 0.15)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    client_loaders = []
    current_idx = 0
    
    # Her istemci için kendi özel veri seti kurallarını (Color Jitter vb.) uygula
    for i, ratio in enumerate(CLIENT_DATA_RATIOS):
        client_ds = datasets.ImageFolder(root=DATA_DIR, transform=get_client_transforms(client_id=i))
        
        client_len = int(train_size * ratio)
        if i == NUM_CLIENTS - 1: # Küsürat kalırsa son istemciye ekle
            client_len = train_size - current_idx
            
        client_indices = train_idx[current_idx : current_idx + client_len]
        subset = Subset(client_ds, client_indices)
        client_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2))
        
        current_idx += client_len

    # Global Validation Set (Tüm modelleri adil test etmek için sabit kurallar)
    val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=get_validation_transforms())
    global_val_dataset = Subset(val_dataset, val_idx)
    val_loader = DataLoader(global_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return client_loaders, val_loader, num_classes

def initialize_model(model_name, num_classes):
    """ İstenen mimariyi ImageNet ağırlıklarıyla yükler ve çıkış katmanını ayarlar """
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'MobileViT':
        # timm kütüphanesi num_classes verildiğinde çıkış katmanını otomatik ayarlar
        model = timm.create_model('mobilevit_s', pretrained=True, num_classes=num_classes)
    
    return model.to(DEVICE)

def train_client_fedprox(model, global_model, train_loader, lr):
    """ İstemci tarafında FEDPROX algoritması ile eğitim """
    model.train()
    global_model.eval() # Global modelin ağırlıkları sadece referans alınacak
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 1. Standart Hata
        loss = criterion(outputs, labels)
        
        # 2. FEDPROX EKLENTİSİ (Proximal Term / Lastik Bant)
        if FEDPROX_MU > 0:
            proximal_term = 0.0
            for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                proximal_term += ((local_param - global_param) ** 2).sum()
            loss += (FEDPROX_MU / 2) * proximal_term
            
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return model.state_dict(), running_loss / total, correct / total

def validate_model(model, val_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

def fed_avg_weighted(global_model, client_weights):
    """ Ağırlıklı FedAvg: Verisi çok olanın sözü daha çok geçer """
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        # Her istemcinin ağırlığını kendi veri oranı (CLIENT_DATA_RATIOS) ile çarpıp topla
        weighted_sum = sum(CLIENT_DATA_RATIOS[i] * client_weights[i][key].float() for i in range(NUM_CLIENTS))
        global_dict[key] = weighted_sum
    global_model.load_state_dict(global_dict)
    return global_model

def plot_benchmark_figures(all_histories):
    """ Akademik Sunum İçin Profesyonel Çift Figürlü Çizim Fonksiyonu (Sadeleştirilmiş) """
    epochs = range(1, NUM_ROUNDS + 1)
    colors = ['red', 'blue', 'green', 'purple']
    
    # ---------------------------------------------------------
    # FİGÜR 1: MODEL KARŞILAŞTIRMA (BENCHMARK)
    # ---------------------------------------------------------
    fig1, axs = plt.subplots(2, 2, figsize=(16, 10))
    

    # Format: (Sözlük Anahtarı, Grafik Başlığı, Y Ekseni Adı, Çizilecek Alt Grafik)
    metrics = [
        ('train_acc', 'Training Accuracy', 'Accuracy', axs[0, 0]),
        ('val_acc', 'Validation Accuracy', 'Accuracy', axs[0, 1]),
        ('train_loss', 'Training Loss', 'Loss', axs[1, 0]),
        ('val_loss', 'Validation Loss', 'Loss', axs[1, 1])
    ]

    for metric_key, title, ylabel, ax in metrics:
        for (model_name, hist), color in zip(all_histories.items(), colors):
            ax.plot(epochs, hist[metric_key], label=model_name, color=color, linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('Figure1_Model_Comparison.png')
    
    # ---------------------------------------------------------
    # FİGÜR 2: LOSS (AŞIRI ÖĞRENME) ANALİZİ
    # ---------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
    # NOT: fig2.suptitle tamamen kaldırıldı.
    
    axes_flat = axs2.flatten()
    for idx, (model_name, hist) in enumerate(all_histories.items()):
        ax = axes_flat[idx]
        ax.plot(epochs, hist['train_loss'], label='Train Loss', color='blue', linestyle='--')
        ax.plot(epochs, hist['val_loss'], label='Validation Loss', color='red')
        
        # Sadelik için başlık sadece modelin adı oldu
        ax.set_title(model_name, fontsize=14, fontweight='bold') 
        ax.set_xlabel('Round')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('Figure2_Loss_Analysis.png')
    
    plt.show()

def run_simulation():
    print(f"Çalışma Ortamı: {DEVICE}")
    print("Veri Seti Hazırlanıyor (Non-IID Dağılım & Müşteriye Özel Augmentation)...")
    client_loaders, val_loader, num_classes = prepare_data_non_iid()
    
    # Tüm modellerin sonuçlarını tutacak büyük sözlük
    all_histories = {}

    # Her modeli sırayla eğit
    for model_name in MODELS_TO_RUN:
        print(f"\n{'='*50}")
        print(f"BAŞLIYOR: {model_name} Modeli (FedProx mu={FEDPROX_MU})")
        print(f"{'='*50}")
        
        global_model = initialize_model(model_name, num_classes)
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # --- YENİ EKLENEN KISIM: En iyi başarıyı takip edecek değişkenler ---
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(global_model.state_dict())
        # ---------------------------------------------------------------
        
        start_time = time.time()

        for round_idx in range(NUM_ROUNDS):
            local_weights, local_losses, local_accs = [], [], []

            # 1. ADIM: İstemcileri Eğit (FedProx ile)
            for i in range(NUM_CLIENTS):
                client_model = copy.deepcopy(global_model)
                w, loss, acc = train_client_fedprox(client_model, global_model, client_loaders[i], LEARNING_RATE)
                local_weights.append(w)
                local_losses.append(loss)
                local_accs.append(acc)

            # 2. ADIM: Ağırlıklı Ortalamayı Al (Veri oranına göre - 50, 35, 15)
            global_model = fed_avg_weighted(global_model, local_weights)

            # 3. ADIM: Test (Validasyon)
            v_loss, v_acc = validate_model(global_model, val_loader)
            
            # --- YENİ EKLENEN KISIM: En iyi modeli kaydetme mantığı ---
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_model_wts = copy.deepcopy(global_model.state_dict())
                # Modeli kendi ismiyle dinamik olarak klasöre kaydet
                torch.save(best_model_wts, f'best_{model_name}_weights.pth')
            # ----------------------------------------------------------
            
            # Eğitim metriklerinin de veri ağırlıklı ortalamasını al
            avg_t_loss = sum(CLIENT_DATA_RATIOS[i] * local_losses[i] for i in range(NUM_CLIENTS))
            avg_t_acc = sum(CLIENT_DATA_RATIOS[i] * local_accs[i] for i in range(NUM_CLIENTS))

            history['train_loss'].append(avg_t_loss)
            history['train_acc'].append(avg_t_acc)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)

            # Ekrana her tur basma, kalabalık yapmasın (10 turda bir bas)
            if (round_idx + 1) % 10 == 0 or round_idx == 0:
                print(f"{model_name} - Tur {round_idx + 1:03d}/{NUM_ROUNDS} | Train Acc: {avg_t_acc:.4f} | Val Acc: {v_acc:.4f} | Train Loss: {avg_t_loss:.4f} | Val Loss: {v_loss:.4f}")

        all_histories[model_name] = history
        # Bitiş mesajına ulaşılan en iyi başarıyı da ekledim
        print(f"\n>>> {model_name} Tamamlandı! En İyi Test Başarısı: %{best_val_acc*100:.2f} | Geçen Süre: {(time.time() - start_time) / 60:.2f} Dakika <<<")

    print("\nTÜM MODELLERİN EĞİTİMİ BİTTİ. GRAFİKLER ÇİZİLİYOR...")
    plot_benchmark_figures(all_histories)

if __name__ == "__main__":
    run_simulation()






