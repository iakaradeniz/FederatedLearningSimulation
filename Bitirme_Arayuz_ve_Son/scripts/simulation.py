import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import os
import time
import matplotlib.pyplot as plt

# ==========================================
# 1) Model Tanımı (ResNet18)
# ==========================================
NUM_CLASSES = 33

def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )
    return model

# ==========================================
# 2) Client Class
# ==========================================
class Client:
    def __init__(self, client_id, trainloader, device):
        self.id = client_id
        self.loader = trainloader
        self.device = device


    def local_train(
        self,
        global_model,
        epochs=1
    ):
        prox_mu = 0.01
        model = get_model()
        model.load_state_dict(global_model.state_dict(), strict=False)
        model.to(self.device)
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
            for x,y in self.loader:
                x,y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                out = model(x)
                loss = criterion(out,y)

                prox_term = 0
                for name, param in model.named_parameters():
                    prox_term += torch.norm(param - global_model_params[name].to(self.device)) ** 2

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

# ==========================================
# 3) Server Class
# ==========================================
class Server:
    def __init__(self):
        self.global_model = get_model()

    def aggregate(
        self,
        client_weights,
        client_sizes
    ):
        global_weights = copy.deepcopy(
            client_weights[0]
        )

        total = sum(client_sizes)
        for key in global_weights.keys():
            orig_type = global_weights[key].dtype
            
            temp_weight = global_weights[key].float() * (client_sizes[0] / total)

            for i in range(1, len(client_weights)):
                temp_weight += (
                    client_weights[i][key].float() *
                    (client_sizes[i] / total)
                )
            
            global_weights[key] = temp_weight.to(orig_type)

        self.global_model.load_state_dict(
            global_weights
        )

    def validate_model(self, global_model, val_loader, device):
        global_model = global_model.to(device)
        global_model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = global_model(inputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / total, correct / total


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
# ==========================================
# 4) Federated Training Loop
# ==========================================
if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformation for ResNet
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Base path for datasets
    base_data_path = os.path.join("Federated_Dataset", "Train_Data")

    # Load datasets
    print("Loading datasets...")
    dataset_hospital1 = datasets.ImageFolder(os.path.join(base_data_path, "client_0"), transform=get_client_transforms(0))
    dataset_hospital2 = datasets.ImageFolder(os.path.join(base_data_path, "client_1"), transform=get_client_transforms(1))
    dataset_hospital3 = datasets.ImageFolder(os.path.join(base_data_path, "client_2"), transform=get_client_transforms(2))
    dataset_validation = datasets.ImageFolder(os.path.join("Federated_Dataset", "Validation_Data"), transform=transform)

    # Create dataloaders
    batch_size = 64
    loader_hospital1 = DataLoader(
        dataset_hospital1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    loader_hospital2 = DataLoader(
        dataset_hospital2,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    loader_hospital3 = DataLoader(
        dataset_hospital3,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        dataset_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print("Datasets loaded successfully.")

    server = Server()

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(server.global_model.state_dict())

    clients = [
        Client(0, loader_hospital1, device),
        Client(1, loader_hospital2, device),
        Client(2, loader_hospital3, device)
    ]

    ROUNDS = 10
    round_losses = []
    round_accs = []
    val_losses = []
    val_accs = []
    
    print("Starting Federated Training...")
    for rnd in range(ROUNDS):

        local_weights = []
        client_sizes = []

        for client in clients:
            print(f"  Round {rnd} - Client {client.id} training...")

            start = time.time()
            weights, loss, accuracy = client.local_train(
                server.global_model,
                epochs=1
            )

            print(f"Client {client.id} - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
            print(time.time()-start)

            local_weights.append(weights)

            client_sizes.append(
                len(client.loader.dataset)
            )

            round_losses.append(loss)
            round_accs.append(accuracy)

        print(f"  Round {rnd} - Aggregating weights...")
        server.aggregate(
            local_weights,
            client_sizes
        )

        avg_loss = sum(round_losses) / len(round_losses)
        avg_acc = sum(round_accs) / len(round_accs)

        val_loss, val_acc = server.validate_model(server.global_model, val_loader, device)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(server.global_model.state_dict())
            torch.save(best_model_wts, f'best_ResNet18_weights.pth')

        print(f"""
        Round {rnd}
        Loss: {avg_loss:.4f}
        Accuracy: {avg_acc:.4f}
        Val Loss: {val_loss:.4f}
        Val Accuracy: {val_acc:.4f}
        """)

        print(
            f"Round {rnd} done\n" + "-"*30
        )
    
    # 1 satır ve 2 sütundan oluşan (yan yana) bir grafik alanı oluşturuyoruz.
    # figsize=(14, 6) ile genişliği artırarak iki grafiğin rahat sığmasını sağlıyoruz.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Grafik: Validation Accuracy (Sol Taraftaki Grafik - axes[0]) ---
    axes[0].plot(
        range(1, len(val_accs) + 1),
        val_accs,
        label="Validation Accuracy",
        color="green",
        marker="o"
    )
    axes[0].set_title("Federated Learning Training - Validation Accuracy")
    axes[0].set_xlabel("Rounds")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # --- 2. Grafik: Validation Loss (Sağ Taraftaki Grafik - axes[1]) ---
    axes[1].plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss",
        color="red",
        marker="o"
    )
    axes[1].set_title("Federated Learning Training - Validation Loss")
    axes[1].set_xlabel("Rounds")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    # Grafikler iç içe geçmesin diye otomatik boşluk ayarlaması yapar
    plt.tight_layout()

    # Tüm figürü tek bir görsel olarak kaydeder
    plt.savefig("federated_learning_metrics.png")

    # Grafiği ekranda gösterir
    plt.show()