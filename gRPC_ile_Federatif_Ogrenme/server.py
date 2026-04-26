"""
Federated Learning gRPC Sunucusu (PyTorch Tabanlı)

Kullanım:
  python server.py [--port 50051] [--min-clients 3] [--rounds 180] [--model ResNet18]

Bu sunucu:
  1. PyTorch modelini başlatır (global model).
  2. İstemcilerden gelen ağırlıkları toplar.
  3. FedAvg ile ağırlıklı ortalama hesaplar (num_samples bazında).
  4. Her turda global modeli validasyon seti üzerinde test eder.
  5. En iyi modeli otomatik olarak kaydeder.
  6. Global modeli istemcilere geri gönderir.
"""

import numpy as np
import argparse
import logging
import time
import os
import copy
import threading
from concurrent import futures
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import grpc
import federated_pb2
import federated_pb2_grpc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# MobileViT için gerekli kütüphane
try:
    import timm
except ImportError:
    print("UYARI: 'timm' kütüphanesi eksik. MobileViT modeli kullanılamaz.")
    print("Kurmak için: pip install timm")


# --- CIHAZ SEÇİMİ ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.2)


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


# -----------------------------------------------------------------------
# VALİDASYON VERİ SETİ YÜKLEME
# -----------------------------------------------------------------------
def load_validation_data(data_dir: str, batch_size: int):
    """
    Sunucu tarafındaki global validasyon veri setini yükler.

    Klasör yapısı:
      data_dir/Validation_Data/sınıf_adı/görüntüler...

    Returns:
        val_loader: DataLoader
        num_classes: int (sınıf sayısı)
    """
    val_path = os.path.join(data_dir, "Validation_Data")

    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Validasyon veri klasörü bulunamadı: {val_path}\n"
            f"Lütfen veri setinin doğru dizinde olduğundan emin olun."
        )

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    num_classes = len(val_dataset.classes)
    num_samples = len(val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    logging.info(f"Validasyon verisi yüklendi: {val_path}")
    logging.info(f"  Sınıf sayısı: {num_classes} | Örnek sayısı: {num_samples}")

    return val_loader, num_classes


# -----------------------------------------------------------------------
# VALİDASYON FONKSİYONU (FLSimulation.py'den alındı)
# -----------------------------------------------------------------------
def validate_model(model, val_loader):
    """
    Global modeli validasyon seti üzerinde test eder.

    Returns:
        val_loss: Ortalama kayıp değeri
        val_acc: Doğruluk oranı
    """
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

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


# -----------------------------------------------------------------------
# PYTORCH STATE_DICT TABANLI FEDAVG (FLSimulation.py'den alındı)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# PROTO ↔ PYTORCH STATE_DICT DÖNÜŞÜM FONKSİYONLARI
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
# Sunucu Hizmet Sınıfı (PyTorch Tabanlı)
# -----------------------------------------------------------------------
class FederatedLearningServicer(federated_pb2_grpc.FederatedLearningServicer):

    def __init__(
        self,
        min_clients: int,
        total_rounds: int,
        model_name: str,
        val_loader: DataLoader,
        num_classes: int,
    ):
        self.min_clients = min_clients
        self.total_rounds = total_rounds
        self.current_round = 0

        self._cond = threading.Condition()
        self._lock = threading.Lock()
        self._round_weights: List[Dict[str, torch.Tensor]] = []
        self._round_samples: List[int] = []
        self._registered_clients: set = set()

        # --- PyTorch Global Model ---
        self.model_name = model_name
        self.global_model = initialize_model(model_name, num_classes)
        self.global_weights: Dict[str, torch.Tensor] | None = None

        # --- Validasyon ---
        self.val_loader = val_loader

        # --- En İyi Model Takibi (FLSimulation.py'den) ---
        self.best_val_acc = 0.0
        self.best_model_wts = copy.deepcopy(self.global_model.state_dict())

        # --- Eğitim Geçmişi ---
        self.history = {"val_loss": [], "val_acc": []}

        self._cond = threading.Condition()
        self.start_time = time.time()

        logging.info(
            f"Sunucu başlatıldı | model={model_name} | "
            f"min_clients={min_clients} | rounds={total_rounds}"
        )

    def SendWeights(self, request, context):
        client_id = request.client_id
        round_num = request.round

        if self.current_round >= self.total_rounds:
            logging.warning(f"[{client_id}] Tüm turlar tamamlandı, istek reddedildi.")
            return federated_pb2.GlobalModelResponse(
                round=self.current_round,
                status="finished",
                layers=state_dict_to_proto(self.global_weights)
            )

        # Proto → PyTorch state_dict dönüşümü
        weights = proto_to_state_dict(request.layers)
        logging.info(
            f"[{client_id}] Tur {round_num} ağırlıkları alındı "
            f"| örnekler={request.num_samples}"
        )

        trigger_aggregation = False
        with self._lock:
            self._registered_clients.add(client_id)
            self._round_weights.append(weights)
            self._round_samples.append(request.num_samples)
            collected = len(self._round_weights)

            if collected >= self.min_clients:
                trigger_aggregation = True

        if trigger_aggregation:
            self._do_aggregation()
        else:
            logging.info(
                f"Henüz yeterli istemci yok "
                f"({collected}/{self.min_clients}). Bekleniyor..."
            )
            with self._cond:
                target_round = self.current_round
                self._cond.wait_for(lambda: self.current_round > target_round, timeout=600)

        return federated_pb2.GlobalModelResponse(
            round=self.current_round,
            status="ok",
            layers=state_dict_to_proto(self.global_model.state_dict())
        )

    def GetStatus(self, request, context):
        with self._lock:
            clients_this_round = len(self._round_weights)
            total_clients = len(self._registered_clients)

        return federated_pb2.StatusResponse(
            current_round=self.current_round,
            clients_registered=total_clients,
            clients_this_round=clients_this_round,
            message=(
                f"Tur {self.current_round}/{self.total_rounds} | "
                f"Bu turda {clients_this_round}/{self.min_clients} istemci hazır."
            ),
        )

    def _do_aggregation(self):
        """
        İstemcilerden gelen ağırlıkları FedAvg ile birleştirir,
        global modeli günceller ve validasyon yapar.
        """
        with self._lock:
            weights_snapshot = list(self._round_weights)
            samples_snapshot = list(self._round_samples)
            self._round_weights.clear()
            self._round_samples.clear()

        self.current_round += 1

        # 1. ADIM: FedAvg — Ağırlıklı ortalama (veri sayısına göre)
        global_weights = copy.deepcopy(
            weights_snapshot[0]
        )

        total = sum(samples_snapshot)
        for key in global_weights.keys():
            orig_type = global_weights[key].dtype
            
            temp_weight = global_weights[key].float() * (samples_snapshot[0] / total)

            for i in range(1, len(weights_snapshot)):
                temp_weight += (
                    weights_snapshot[i][key].float() *
                    (samples_snapshot[i] / total)
                )
            
            global_weights[key] = temp_weight.to(orig_type)

        self.global_model.load_state_dict(global_weights)

        logging.info(
            f"{'='*60}\n"
            f"  Tur {self.current_round}/{self.total_rounds} tamamlandı: "
            f"FedAvg uygulandı ({len(weights_snapshot)} istemci)\n"
            f"{'='*60}"
        )

        # 3. ADIM: Validasyon (FLSimulation.py'den)
        v_loss, v_acc = validate_model(self.global_model, self.val_loader)
        self.history["val_loss"].append(v_loss)
        self.history["val_acc"].append(v_acc)

        logging.info(
            f"  Validasyon | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}"
        )

        # 4. ADIM: En iyi modeli kaydet (FLSimulation.py'den)
        if v_acc > self.best_val_acc:
            self.best_val_acc = v_acc
            self.best_model_wts = copy.deepcopy(self.global_model.state_dict())
            save_path = f"best_{self.model_name}_weights.pth"
            torch.save(self.best_model_wts, save_path)
            logging.info(
                f"  ★ Yeni en iyi model! Val Acc: {v_acc:.4f} → {save_path}"
            )

        elapsed = (time.time() - self.start_time) / 60
        logging.info(f"  Geçen süre: {elapsed:.2f} dakika")

        with self._cond:
            self._cond.notify_all()

        if self.current_round >= self.total_rounds:
            logging.info(
                f"\n{'='*60}\n"
                f"  TÜM TURLAR TAMAMLANDI!\n"
                f"  Model: {self.model_name}\n"
                f"  En İyi Val Acc: %{self.best_val_acc * 100:.2f}\n"
                f"  Toplam Süre: {elapsed:.2f} dakika\n"
                f"{'='*60}"
            )


# -----------------------------------------------------------------------
# SUNUCU BAŞLATMA
# -----------------------------------------------------------------------
def serve(
    port: int,
    min_clients: int,
    rounds: int,
    model_name: str,
    data_dir: str,
    batch_size: int,
):
    """Sunucuyu başlatır: model, validasyon verisi, gRPC hizmeti."""

    # Validasyon veri setini yükle
    logging.info(f"Çalışma ortamı: {DEVICE}")
    logging.info("Validasyon veri seti yükleniyor...")
    val_loader, num_classes = load_validation_data(data_dir, batch_size)

    # gRPC sunucusu
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),
        options=[
            ("grpc.max_send_message_length", 500 * 1024 * 1024),   # 500 MB
            ("grpc.max_receive_message_length", 500 * 1024 * 1024),
            # Sıkıştırmayı aktif et
            ("grpc.default_compression_algorithm", grpc.Compression.Gzip),
        ],
    )

    servicer = FederatedLearningServicer(
        min_clients=min_clients,
        total_rounds=rounds,
        model_name=model_name,
        val_loader=val_loader,
        num_classes=num_classes,
    )
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)

    address = f"0.0.0.0:{port}"
    server.add_insecure_port(address)
    server.start()
    logging.info(f"Sunucu dinleniyor: {address}")
    logging.info(
        f"Ayarlar → model={model_name} | min_clients={min_clients} | rounds={rounds}"
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Sunucu durduruluyor...")
        server.stop(grace=5)


# -----------------------------------------------------------------------
# KOMUT SATIRI ARAYÜZÜ
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SERVER] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Federated Learning gRPC Sunucusu (PyTorch)"
    )
    parser.add_argument(
        "--port", type=int, default=50051,
        help="Dinlenecek port (varsayilan: 50051)"
    )
    parser.add_argument(
        "--min-clients", type=int, default=3,
        help="Aggregation icin gereken minimum istemci sayisi (varsayilan: 3)"
    )
    parser.add_argument(
        "--rounds", type=int, default=75,
        help="Toplam egitim turu sayisi (varsayilan: 180)"
    )
    parser.add_argument(
        "--model", type=str, default="MobileNetV2",
        choices=["ResNet18", "MobileNetV2", "DenseNet121", "MobileViT"],
        help="Kullanilacak model mimarisi (varsayilan: ResNet18)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./Federated_Dataset_Yeni/Federated_Dataset",
        help="Veri seti kok dizini (varsayilan: ./Federated_Dataset/Federated_Dataset)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Validasyon batch boyutu (varsayilan: 32)"
    )

    args = parser.parse_args()

    serve(
        port=args.port,
        min_clients=args.min_clients,
        rounds=args.rounds,
        model_name=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )