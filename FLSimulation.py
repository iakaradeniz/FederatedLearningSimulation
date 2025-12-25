import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import os

# --- AYARLAR VE PARAMETRELER ---
DATA_DIR = "/content/drive/MyDrive/Preprocessed_Dataset"
NUM_CLIENTS = 3       # Simüle edilecek istemci (hastane) sayısı
NUM_ROUNDS = 180      # Toplam federatif eğitim turu
LOCAL_EPOCHS = 1      # Her istemcinin kendi verisinde yapacağı tur sayısı
BATCH_SIZE = 32       # Bir kerede işlenecek görüntü sayısı
LEARNING_RATE = 0.001 # Sabit öğrenme oranı
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Donanım seçimi (GPU varsa kullanılır)

# Veri ön işleme fonksiyonlarını dış dosyadan çekiyoruz
from Data_Augmentation_and_Data_Normalization import get_training_transforms, get_validation_transforms

def prepare_data():
    """ Veriyi %85 eğitim ve %15 test (validation) olarak böler ve istemcilere paylaştırır """
    # Eğitim için rastgele değişimler (augmentation) içeren seti oluştur
    train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=get_training_transforms())
    # Test için sadece standart boyutlandırma içeren seti oluştur
    val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=get_validation_transforms())

    dataset_size = len(train_dataset)
    val_size = int(dataset_size * 0.15)
    train_size = dataset_size - val_size

    # Veriyi karıştırmak için rastgele indisler oluştur
    indices = torch.randperm(dataset_size).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # Eğitim verisini istemci sayısına eşit olarak böl
    client_train_datasets = []
    split_step = train_size // NUM_CLIENTS
    for i in range(NUM_CLIENTS):
        start = i * split_step
        end = (i + 1) * split_step if i != NUM_CLIENTS - 1 else train_size
        client_train_datasets.append(Subset(train_dataset, train_idx[start:end]))

    # Global test seti (Validation) oluştur
    global_val_dataset = Subset(val_dataset, val_idx)

    # Python DataLoader yapılarını oluştur (Veriyi modele beslemek için)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for ds in client_train_datasets]
    val_loader = DataLoader(global_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return client_loaders, val_loader, len(train_dataset.classes)

def initialize_model(num_classes):
    """ Önceden eğitilmiş ResNet18 modelini yükler ve son katmanını günceller """
    model = models.resnet18(pretrained=True) # ImageNet bilgisiyle yüklü model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Çıkış katmanını bakteri sınıf sayısına göre ayarla
    model = model.to(DEVICE)
    return model

def train_client(model, train_loader, lr):
    """ İstemci tarafındaki yerel eğitim süreci """
    model.train() # Modeli eğitim moduna al
    criterion = nn.CrossEntropyLoss() # Hata fonksiyonu
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # Optimizasyon algoritması

    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad() # Gradyanları sıfırla
        outputs = model(inputs) # Tahmin yap
        loss = criterion(outputs, labels) # Hatayı hesapla
        loss.backward() # Geriye yayılım (Backpropagation)
        optimizer.step() # Ağırlıkları güncelle

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return model.state_dict(), running_loss / total, correct / total

def validate_model(model, val_loader):
    """ Modelin hiç görmediği veri üzerindeki performansını ölçer """
    model.eval() # Modeli test moduna al (dropout vb. kapatılır)
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad(): # Gradyan takibini kapat (bellek tasarrufu sağlar)
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

def fed_avg(global_model, client_weights):
    """ Federated Averaging (FedAvg): İstemci ağırlıklarının ortalamasını alır """
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        # Tüm istemcilerden gelen ağırlıkları topla ve ortalamasını al
        global_dict[key] = torch.stack([client_weights[i][key].float() for i in range(len(client_weights))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def get_all_preds(model, loader):
    """ Confusion Matrix için tüm tahminleri toplar """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def run_simulation():
    # Başlangıç hazırlıkları
    client_loaders, val_loader, num_classes = prepare_data()
    class_names = datasets.ImageFolder(root=DATA_DIR).classes
    global_model = initialize_model(num_classes)

    # En iyi modeli ve metriklerini saklamak için değişkenler
    best_val_acc = 0.0
    best_metrics = {}
    best_model_wts = copy.deepcopy(global_model.state_dict())

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Çalışma Ortamı: {DEVICE}")
    print("\n--- Federatif Öğrenme Simülasyonu Başlıyor ---")
    start_time = time.time()

    for round_idx in range(NUM_ROUNDS):
        # NOT: LR Scheduler kısmı buradan kaldırıldı.

        local_weights, local_losses, local_accs = [], [], []

        # 1. ADIM: İstemcileri eğit
        for i in range(NUM_CLIENTS):
            # Global modeli her istemciye kopyala
            client_model = copy.deepcopy(global_model)
            # Değişiklik: current_lr yerine sabit LEARNING_RATE kullanılıyor
            w, loss, acc = train_client(client_model, client_loaders[i], LEARNING_RATE)
            local_weights.append(w)
            local_losses.append(loss)
            local_accs.append(acc)

        # 2. ADIM: Sunucuda ağırlıkları birleştir
        global_model = fed_avg(global_model, local_weights)

        # 3. ADIM: Doğrulama (Validation)
        v_loss, v_acc = validate_model(global_model, val_loader)
        avg_t_loss = sum(local_losses) / len(local_losses)
        avg_t_acc = sum(local_accs) / len(local_accs)

        # En iyi modeli metrikleriyle birlikte kaydet
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, 'best_bacteria_model.pth')
            # En iyi turun verilerini sakla
            best_metrics = {
                'round': round_idx + 1,
                't_acc': avg_t_acc,
                'v_acc': v_acc,
                't_loss': avg_t_loss,
                'v_loss': v_loss
            }

        # İstatistikleri listelere ekle (Grafik için)
        history['train_loss'].append(avg_t_loss)
        history['train_acc'].append(avg_t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # İstenen çıktı formatı
        print(f"ROUND {round_idx + 1}/{NUM_ROUNDS} | Train Acc: {avg_t_acc:.4f} | Val Acc: {v_acc:.4f} | Train Loss: {avg_t_loss:.4f} | Val Loss: {v_loss:.4f}")

    # Eğitim bittiğinde özet yazdır
    print(f"\n--- Eğitim Tamamlandı! ---")
    print(f"Toplam Süre: {time.time() - start_time:.2f} saniye")
    print(f"EN İYİ MODEL (Round {best_metrics['round']}): Train Acc: {best_metrics['t_acc']:.4f} | Val Acc: {best_metrics['v_acc']:.4f} | Train Loss: {best_metrics['t_loss']:.4f} | Val Loss: {best_metrics['v_loss']:.4f}")

    # --- GRAFİKLER VE MATRİS ---
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)

    plt.plot(history['train_loss'], label='Train Loss', color='blue', linestyle='--')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Kayıp Grafiği'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)

    plt.plot(history['train_acc'], label='Train Acc', color='blue', linestyle='--')
    plt.plot(history['val_acc'], label='Val Acc', color='green')
    plt.title('Başarım Grafiği'); plt.legend(); plt.grid(True)
    plt.show()

    # En iyi model ağırlıklarını yükleyerek matrisi çiz
    global_model.load_state_dict(best_model_wts)
    y_true, y_pred = get_all_preds(global_model, val_loader)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(len(class_names)),
                yticklabels=range(len(class_names)))
    plt.title('Final Karmaşıklık Matrisi'); plt.show()

if __name__ == "__main__":
    run_simulation()