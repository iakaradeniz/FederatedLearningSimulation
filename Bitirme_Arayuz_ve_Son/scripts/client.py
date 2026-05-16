"""
Federated Learning gRPC İstemcisi (PyTorch Tabanlı)

Kullanım:
  python client.py --client-id 0 [--model ResNet18] [--server localhost:50051] [--rounds 180]

Bu istemci:
  1. Kendi fiziksel klasöründen (Federated_Dataset/Train_Data/client_X) veri yükler.
  2. İstemciye özel augmentation kuralları uygular.
  3. FedProx algoritmasıyla PyTorch modelini eğitir.
  4. Eğitilmiş ağırlıkları gRPC üzerinden sunucuya gönderir.
  5. Sunucudan dönen global modeli yükler ve bir sonraki tura geçer.
"""

import numpy as np
import argparse
import logging
import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import grpc
import federated_pb2
import federated_pb2_grpc
import os

# MobileViT için gerekli kütüphane
try:
    import timm
except ImportError:
    print("UYARI: 'timm' kütüphanesi eksik. MobileViT modeli kullanılamaz.")
    print("Kurmak için: pip install timm")
    timm = None

GLOBAL_CLASS_TO_IDX = {
    'Acinetobacter.baumanii': 0, 'Actinomyces.israeli': 1, 'Bacteroides.fragilis': 2, 
    'Bifidobacterium.spp': 3, 'Candida.albicans': 4, 'Clostridium.perfringens': 5, 
    'Enterococcus.faecalis': 6, 'Enterococcus.faecium': 7, 'Escherichia.coli': 8, 
    'Fusobacterium': 9, 'Lactobacillus.casei': 10, 'Lactobacillus.crispatus': 11, 
    'Lactobacillus.delbrueckii': 12, 'Lactobacillus.gasseri': 13, 'Lactobacillus.jehnsenii': 14, 
    'Lactobacillus.johnsonii': 15, 'Lactobacillus.paracasei': 16, 'Lactobacillus.plantarum': 17, 
    'Lactobacillus.reuteri': 18, 'Lactobacillus.rhamnosus': 19, 'Lactobacillus.salivarius': 20, 
    'Listeria.monocytogenes': 21, 'Micrococcus.spp': 22, 'Neisseria.gonorrhoeae': 23, 
    'Porfyromonas.gingivalis': 24, 'Propionibacterium.acnes': 25, 'Proteus': 26, 
    'Pseudomonas.aeruginosa': 27, 'Staphylococcus.aureus': 28, 'Staphylococcus.epidermidis': 29, 
    'Staphylococcus.saprophiticus': 30, 'Streptococcus.agalactiae': 31, 'Veionella': 32
}
# --- CIHAZ SEÇİMİ ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FederatedImageFolder(ImageFolder):
    def find_classes(self, directory: str):
        # 1. O istemcide GERÇEKTEN var olan klasörleri (sınıfları) bul
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        # 2. Sadece var olan bu klasörleri, bizim GLOBAL sözlüğümüzdeki orijinal ID'ler ile eşleştir
        class_to_idx = {
            cls_name: GLOBAL_CLASS_TO_IDX[cls_name] 
            for cls_name in classes 
            if cls_name in GLOBAL_CLASS_TO_IDX
        }
        
        return classes, class_to_idx

# -----------------------------------------------------------------------
# MODEL BAŞLATMA (FLSimulation.py'den alındı)
# -----------------------------------------------------------------------
def initialize_model(model_name: str, num_classes: int):
    """
    İstenen mimariyi ImageNet ağırlıklarıyla yükler ve çıkış katmanını
    veri setindeki sınıf sayısına göre ayarlar.
    """
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "MobileViT":
        if timm is None:
            raise RuntimeError("MobileViT kullanmak için 'pip install timm' gerekli.")
        model = timm.create_model("mobilevit_s", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return model.to(DEVICE)


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# -----------------------------------------------------------------------
# VERİ YÜKLEME (İstemciye Özel Klasörden)
# -----------------------------------------------------------------------
def load_client_data(client_id: int, data_dir: str, batch_size: int):
    """
    İstemciye ait fiziksel klasörden veri yükler ve istemciye özel
    augmentation kurallarını uygular.

    Klasör yapısı:
      data_dir/Train_Data/client_{id}/sınıf_adı/görüntüler...

    Returns:
        train_loader: DataLoader
        num_classes: int (sınıf sayısı)
        num_samples: int (toplam örnek sayısı)
    """
    client_path = os.path.join(data_dir, "Train_Data", f"client_{client_id}")

    if not os.path.exists(client_path):
        raise FileNotFoundError(
            f"İstemci veri klasörü bulunamadı: {client_path}\n"
            f"Lütfen veri setinin doğru dizinde olduğundan emin olun."
        )

    # İstemciye özel augmentation kurallarını al
    transform = get_client_transforms(client_id)
    dataset = FederatedImageFolder(root=client_path, transform=transform)

    print(dataset.class_to_idx)

    num_classes = len(dataset.classes)
    num_samples = len(dataset)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    logging.info(f"Veri yüklendi: {client_path}")
    logging.info(f"  Sınıf sayısı: {num_classes} | Örnek sayısı: {num_samples}")
    logging.info(f"  Sınıflar: {dataset.classes[:5]}{'...' if num_classes > 5 else ''}")

    return train_loader, num_classes, num_samples


# -----------------------------------------------------------------------
# FEDPROX EĞİTİM FONKSİYONU (FLSimulation.py'den alındı)
# -----------------------------------------------------------------------

def local_train(
        num_classes,
        global_model,
        loader,
        model_name,
        epochs=1
    ):
        prox_mu = 0.01
        model = initialize_model(model_name, num_classes)
        model.load_state_dict(global_model.state_dict(), strict=False)
        model.to(DEVICE)
        model.train()

        global_model_params = copy.deepcopy(global_model.state_dict())
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.001,
            momentum=0.9
        )

        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        for _ in range(epochs):
            for x,y in loader:
                x,y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                out = model(x)
                loss = criterion(out,y)

                prox_term = 0
                for name, param in model.named_parameters():
                    prox_term += torch.norm(param - global_model_params[name].to(DEVICE)) ** 2

                loss += (prox_mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)

                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total

        return model.state_dict(), avg_loss, accuracy


# -----------------------------------------------------------------------
# PYTORCH STATE_DICT ↔ PROTO DÖNÜŞÜM FONKSİYONLARI
# -----------------------------------------------------------------------

def state_dict_to_proto(state_dict: dict):
    """
    PyTorch state_dict'i protobuf LayerWeights listesine dönüştürür.

    Her katman:
      - name: Katman adı (ör: "layer1.0.conv1.weight")
      - values: Düzleştirilmiş float dizisi
      - shape: Orijinal tensor boyutları
    """
    layers = []
    for name, tensor in state_dict.items():
        arr = tensor.cpu().numpy().astype(np.float16)
        lw = federated_pb2.LayerWeights(
            name=name,
            values=arr.tobytes(),
            shape=list(arr.shape),
        )
        layers.append(lw)
    return layers


def proto_to_state_dict(layers) -> dict:
    """
    Protobuf LayerWeights listesini PyTorch state_dict formatına dönüştürür.
    """
    state_dict = {}
    for layer in layers:
        arr = np.frombuffer(layer.values, dtype=np.float16).astype(np.float32)
        tensor = torch.tensor(arr).reshape(list(layer.shape))
        state_dict[layer.name] = tensor
    return state_dict


# -----------------------------------------------------------------------
# ANA İSTEMCİ DÖNGÜSÜ
# -----------------------------------------------------------------------
def run_client(
    client_id: int,
    server_addr: str,
    model_name: str,
    total_rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    fedprox_mu: float,
    data_dir: str,
):
    """
    Federatif öğrenme istemcisi ana döngüsü.

    Her turda:
      1. Yerel veriyle FedProx eğitimi yap
      2. Ağırlıkları gRPC ile sunucuya gönder
      3. Sunucudan dönen global modeli yükle
      4. Sonraki tura geç
    """
    
    print("Loading certificates...")

    try:
        with open('./client/ca.crt', 'rb') as f:
            trusted_certs = f.read()
        with open('./client/client.key', 'rb') as f:
            private_key = f.read()
        with open('./client/client.crt', 'rb') as f:
            certificate_chain = f.read()
    except FileNotFoundError as e:
        print(f"Sertifika dosyası bulunamadı: {e}")
        exit(1)

    credentials = grpc.ssl_channel_credentials(
        root_certificates=trusted_certs,
        private_key=private_key,
        certificate_chain=certificate_chain
    )

    # --- gRPC Kanal Bağlantısı ---
    channel = grpc.secure_channel(
        server_addr,
        credentials,
        options=[
            ("grpc.max_send_message_length", 500 * 1024 * 1024),   # 500 MB
            ("grpc.max_receive_message_length", 500 * 1024 * 1024),
            # Sıkıştırmayı aktif et
            ("grpc.default_compression_algorithm", grpc.Compression.Gzip),
        ],
    )
    stub = federated_pb2_grpc.FederatedLearningStub(channel)

    # --- Veri Yükleme ---
    logging.info(f"Veri seti yükleniyor: client_{client_id}")
    train_loader, num_classes, num_samples = load_client_data(
        client_id, data_dir, batch_size
    )

    # --- Model Başlatma ---
    logging.info(f"Model başlatılıyor: {model_name} ({num_classes} sınıf)")
    model = initialize_model(model_name, len(GLOBAL_CLASS_TO_IDX))
    
    global_model = initialize_model(model_name, len(GLOBAL_CLASS_TO_IDX))
        
    logging.info(f"Çalışma ortamı: {DEVICE}")

    start_time = time.time()

    for fed_round in range(1, total_rounds + 1):
        logging.info(f"\n{'='*60}")
        logging.info(
            f"[ Federated Tur {fed_round}/{total_rounds} ] "
            f"Yerel FedProx eğitimi başlıyor (mu={fedprox_mu})..."
        )

        # Global modelin ağırlıklarını referans modele kopyala
        global_model.load_state_dict(model.state_dict())

        # Yerel eğitim (FedProx)
        trained_state_dict, train_loss, train_acc = local_train(
            len(GLOBAL_CLASS_TO_IDX),
            global_model=global_model,
            loader=train_loader,
            epochs=1,
            model_name=model_name
        )

        logging.info(
            f"Yerel eğitim tamamlandı | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

        # --- Ağırlıkları sunucuya gönder ---
        logging.info("Ağırlıklar sunucuya gönderiliyor...")
        request = federated_pb2.WeightsRequest(
            client_id=f"client_{client_id}",
            round=fed_round,
            num_samples=num_samples,
            layers=state_dict_to_proto(trained_state_dict),
        )

        try:
            response = stub.SendWeights(request, timeout=600)
        except grpc.RpcError as e:
            logging.error(f"gRPC hatası: {e.code()} — {e.details()}")
            break

        # --- Sunucu yanıtını işle ---
        if response.status == "finished":
            logging.info("Sunucu: Tüm turlar tamamlandı.")
            break

        # Global modeli yükle
        global_state_dict = proto_to_state_dict(response.layers)
        model.load_state_dict(global_state_dict)

        logging.info(
            f"Global model alındı (tur {response.round}) | "
            f"Geçen süre: {(time.time() - start_time) / 60:.2f} dk"
        )
        
        # Her tur sonunda GPU belleğini temizle (tıkanmaları önlemek için)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = (time.time() - start_time) / 60
    logging.info(f"\nFederated Learning tamamlandı! Toplam süre: {elapsed:.2f} dakika")
    channel.close()


# -----------------------------------------------------------------------
# KOMUT SATIRI ARAYÜZÜ
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Federated Learning gRPC Istemcisi (PyTorch)"
    )
    parser.add_argument(
        "--client-id", type=int, default=0,
        help="Istemci kimligi (0, 1, 2 -- klasor adiyla eslesir)"
    )
    parser.add_argument(
        "--server", type=str, default="100.77.33.86:50051",
        help="Sunucu adresi (varsayilan: localhost:50051)"
    )
    parser.add_argument(
        "--model", type=str, default="MobileNetV2",
        choices=["ResNet18", "MobileNetV2", "DenseNet121", "MobileViT"],
        help="Kullanilacak model mimarisi (varsayilan: ResNet18)"
    )
    parser.add_argument(
        "--rounds", type=int, default=75,
        help="Toplam federatif tur sayisi (varsayilan: 180)"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=1,
        help="Her turda yerel epoch sayisi (varsayilan: 1)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Ogrenme hizi (varsayilan: 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch boyutu (varsayilan: 32)"
    )
    parser.add_argument(
        "--fedprox-mu", type=float, default=0.01,
        help="FedProx ceza parametresi (0 = FedAvg, varsayilan: 0.01)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./Federated_Dataset_Yeni/Federated_Dataset",
        help="Veri seti kok dizini (varsayilan: ./Federated_Dataset/Federated_Dataset)"
    )

    args = parser.parse_args()

    logging.getLogger().name = f"client_{args.client_id}"

    run_client(
        client_id=args.client_id,
        server_addr=args.server,
        model_name=args.model,
        total_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        fedprox_mu=args.fedprox_mu,
        data_dir=args.data_dir,
    )
