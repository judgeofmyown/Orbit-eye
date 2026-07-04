"""
Training skeleton for TinyCloudNet. This is intentionally NOT run for you —
point DATA_DIR at your prepared dataset (see data/DATASETS.md) and run:

    python models/train_cloud_classifier.py --data_dir data/cloud --epochs 15

Saves the best checkpoint to models/weights/cloud_classifier.pth, which is exactly
where edge/inference_engine.py looks for it.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cloud_classifier import TinyCloudNet, CLOUD_CLASSES

WEIGHTS_OUT = os.path.join(os.path.dirname(__file__), "weights", "cloud_classifier.pth")

TRAIN_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train(data_dir: str, epochs: int, batch_size: int, lr: float, device: str):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=TRAIN_TF)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=VAL_TF)

    assert train_ds.classes == CLOUD_CLASSES, (
        f"Expected class folders {CLOUD_CLASSES}, found {train_ds.classes}. "
        "Rename your class subfolders to match, or edit CLOUD_CLASSES in "
        "cloud_classifier.py to match your folder names."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TinyCloudNet(num_classes=len(CLOUD_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        train_loss = running_loss / len(train_ds)
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLOUD_CLASSES,
                "val_acc": val_acc,
            }, WEIGHTS_OUT)
            print(f"  saved new best checkpoint (val_acc={val_acc:.4f}) -> {WEIGHTS_OUT}")

    print(f"Training done. Best val_acc={best_acc:.4f}. Weights at {WEIGHTS_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/cloud")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)
