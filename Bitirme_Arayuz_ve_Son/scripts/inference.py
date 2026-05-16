"""
Bakteriyel Görüntü Sınıflandırma - Inference Modülü
=====================================================

Kullanım (komut satırından):
  python inference.py --image ./test.jpg --model-path ./models/resnet18.pth --arch ResNet18

Kullanım (Electron main process'ten):
  python inference.py --image <path> --model-path <path> --arch <ResNet18|MobileNetV2|DenseNet121|MobileViT>

Çıktı: JSON formatında stdout'a yazılır.
  {
    "success": true,
    "prediction": "Staphylococcus.aureus",
    "confidence": 0.9231,
    "top5": [
      {"class": "Staphylococcus.aureus",    "score": 0.9231},
      {"class": "Staphylococcus.epidermidis","score": 0.0412},
      ...
    ],
    "all_scores": {"Acinetobacter.baumanii": 0.0001, ...},
    "inference_time_ms": 42.3,
    "device": "cpu",
    "model_arch": "ResNet18"
  }
"""

import sys
import json
import time
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# MobileViT için timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# -----------------------------------------------------------------------
# SABİTLER
# -----------------------------------------------------------------------
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

NUM_CLASSES = len(GLOBAL_CLASS_TO_IDX)  # 33
IDX_TO_CLASS = {v: k for k, v in GLOBAL_CLASS_TO_IDX.items()}

# ImageNet normalizasyonu (eğitimde kullanılanla aynı)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------
# MODEL YÜKLEME
# -----------------------------------------------------------------------
def build_model_skeleton(arch: str) -> nn.Module:
    """
    Mimariye göre boş model iskeleti oluşturur (ağırlıksız).
    Çıkış katmanı NUM_CLASSES'a göre ayarlanır.
    """
    if arch == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif arch == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif arch == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    elif arch == "MobileViT":
        if not TIMM_AVAILABLE:
            raise RuntimeError(
                "MobileViT için 'timm' kütüphanesi gerekli. "
                "Kurmak için: pip install timm"
            )
        model = timm.create_model("mobilevit_s", pretrained=False, num_classes=NUM_CLASSES)

    else:
        raise ValueError(f"Bilinmeyen mimari: '{arch}'. "
                         f"Geçerli seçenekler: ResNet18, MobileNetV2, DenseNet121, MobileViT")

    return model


def load_model(model_path: str, arch: str, device: torch.device) -> nn.Module:
    """
    .pth dosyasından model ağırlıklarını yükler.

    .pth dosyası şunlardan biri olabilir:
      - Sadece state_dict (en yaygın)
      - {'model': state_dict, ...} gibi bir dict
      - Doğrudan model nesnesi (torch.save(model, ...))
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Checkpoint formatını tespit et
    if isinstance(checkpoint, dict):
        # state_dict doğrudan mı yoksa sarmalanmış mı?
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Doğrudan state_dict olduğunu varsay
            state_dict = checkpoint
    else:
        # torch.save(model, ...) ile kaydedilmiş tam model
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    # DataParallel ile kaydedilmişse "module." önekini temizle
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model = build_model_skeleton(arch)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if missing:
        sys.stderr.write(f"[UYARI] Eksik katmanlar ({len(missing)}): {missing[:3]}...\n")
    if unexpected:
        sys.stderr.write(f"[UYARI] Beklenmeyen katmanlar ({len(unexpected)}): {unexpected[:3]}...\n")

    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------
def predict(image_path: str, model: nn.Module, device: torch.device, top_k: int = 5):
    """
    Tek bir görüntü için inference çalıştırır.

    Returns:
        dict: prediction, confidence, top_k sonuçlar, tüm softmax skorları
    """
    # Görüntüyü yükle ve dönüştür
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Görüntü açılamadı: {image_path} — {e}")

    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Forward pass
    t_start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                    # [1, 33]
        probs = F.softmax(logits, dim=1)[0]       # [33]
    inference_ms = (time.perf_counter() - t_start) * 1000

    # Top-K sonuçlar
    top_k = min(top_k, NUM_CLASSES)
    top_probs, top_indices = torch.topk(probs, top_k)

    top_k_list = [
        {
            "class": IDX_TO_CLASS[idx.item()],
            "score": round(top_probs[i].item(), 6)
        }
        for i, idx in enumerate(top_indices)
    ]

    # En iyi tahmin
    best_idx = top_indices[0].item()
    best_class = IDX_TO_CLASS[best_idx]
    best_score = round(top_probs[0].item(), 6)

    # Tüm softmax skorları (UI'da bar chart için)
    all_scores = {
        IDX_TO_CLASS[i]: round(probs[i].item(), 6)
        for i in range(NUM_CLASSES)
    }

    return {
        "prediction": best_class,
        "confidence": best_score,
        "top5": top_k_list,
        "all_scores": all_scores,
        "inference_time_ms": round(inference_ms, 2),
    }


# -----------------------------------------------------------------------
# ANA FONKSİYON
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Bakteriyel görüntü sınıflandırma - Inference"
    )
    parser.add_argument("--image",      required=True,  help="Tahmin yapılacak görüntü dosyası yolu")
    parser.add_argument("--model-path", required=True,  help="Eğitilmiş .pth model dosyası yolu")
    parser.add_argument(
        "--arch", required=True,
        choices=["ResNet18", "MobileNetV2", "DenseNet121", "MobileViT"],
        help="Model mimarisi"
    )
    parser.add_argument("--top-k",  type=int, default=5,   help="Kaç adet top sonuç dönsün (varsayılan: 5)")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto (varsayılan: auto)")
    args = parser.parse_args()

    # Cihaz seçimi
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    result = {"success": False}

    try:
        model = load_model(args.model_path, args.arch, device)
        pred  = predict(args.image, model, device, top_k=args.top_k)

        result.update({
            "success":          True,
            "model_arch":       args.arch,
            "device":           str(device),
            **pred,
        })

    except Exception as e:
        result["error"] = str(e)

    # Electron main process sadece stdout'u okur
    print(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()