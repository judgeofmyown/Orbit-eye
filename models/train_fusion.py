"""
Training script for OrbitFusionNet — joint training on the same data/cloud and
data/events ImageFolder trees used for the two separate models, but as one shared
model with partial supervision: each cloud batch only updates the cloud head (+
shared trunk), each event batch only updates the event head (+ shared trunk).

The cloud dataset is much smaller than the events dataset (~700 vs ~62k images), so
the cloud loader is cycled to match the events loader's length each epoch.

    python models/train_fusion.py --epochs 15

Saves the best checkpoint (by mean of cloud_val_acc and event_val_acc) to
models/weights/fusion_net.pth.
"""
import argparse
import itertools
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fusion_net import OrbitFusionNet, CLOUD_CLASSES, EVENT_CLASSES

WEIGHTS_OUT = os.path.join(os.path.dirname(__file__), "weights", "fusion_net.pth")

TRAIN_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def eval_head(model, loader, device, head: str):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        cloud_logits, event_logits = model(imgs)
        logits = cloud_logits if head == "cloud" else event_logits
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train(cloud_dir, events_dir, epochs, batch_size, lr, device):
    cloud_train = datasets.ImageFolder(os.path.join(cloud_dir, "train"), transform=TRAIN_TF)
    cloud_val = datasets.ImageFolder(os.path.join(cloud_dir, "val"), transform=VAL_TF)
    event_train = datasets.ImageFolder(os.path.join(events_dir, "train"), transform=TRAIN_TF)
    event_val = datasets.ImageFolder(os.path.join(events_dir, "val"), transform=VAL_TF)

    assert cloud_train.classes == CLOUD_CLASSES, (
        f"cloud folder order {cloud_train.classes} != expected {CLOUD_CLASSES}")
    assert event_train.classes == EVENT_CLASSES, (
        f"events folder order {event_train.classes} != expected {EVENT_CLASSES}")

    cloud_loader = DataLoader(cloud_train, batch_size=batch_size, shuffle=True, num_workers=2)
    event_loader = DataLoader(event_train, batch_size=batch_size, shuffle=True, num_workers=4)
    cloud_val_loader = DataLoader(cloud_val, batch_size=batch_size, shuffle=False, num_workers=2)
    event_val_loader = DataLoader(event_val, batch_size=batch_size, shuffle=False, num_workers=2)

    model = OrbitFusionNet(len(CLOUD_CLASSES), len(EVENT_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cloud_criterion = nn.CrossEntropyLoss()
    # events is heavily imbalanced (oil_spill/other_anomaly rarer than wildfire/none)
    event_counts = torch.bincount(torch.tensor(event_train.targets), minlength=len(EVENT_CLASSES))
    event_weights = (1.0 / event_counts.float())
    event_weights = event_weights / event_weights.sum() * len(EVENT_CLASSES)
    event_criterion = nn.CrossEntropyLoss(weight=event_weights.to(device))

    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    best_score = 0.0

    for epoch in range(epochs):
        model.train()
        running_cloud_loss, running_event_loss = 0.0, 0.0
        n_steps = 0
        cloud_iter = itertools.cycle(cloud_loader)  # smaller dataset, loop to match events

        for event_imgs, event_labels in event_loader:
            cloud_imgs, cloud_labels = next(cloud_iter)
            cloud_imgs, cloud_labels = cloud_imgs.to(device), cloud_labels.to(device)
            event_imgs, event_labels = event_imgs.to(device), event_labels.to(device)

            optimizer.zero_grad()

            cloud_logits, _ = model(cloud_imgs)
            cloud_loss = cloud_criterion(cloud_logits, cloud_labels)

            _, event_logits = model(event_imgs)
            event_loss = event_criterion(event_logits, event_labels)

            (cloud_loss + event_loss).backward()
            optimizer.step()

            running_cloud_loss += cloud_loss.item()
            running_event_loss += event_loss.item()
            n_steps += 1
        scheduler.step()

        cloud_acc = eval_head(model, cloud_val_loader, device, "cloud")
        event_acc = eval_head(model, event_val_loader, device, "event")
        score = (cloud_acc + event_acc) / 2

        print(f"epoch {epoch+1}/{epochs}  cloud_loss={running_cloud_loss/n_steps:.4f}  "
              f"event_loss={running_event_loss/n_steps:.4f}  cloud_val_acc={cloud_acc:.4f}  "
              f"event_val_acc={event_acc:.4f}")

        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "cloud_classes": CLOUD_CLASSES,
                "event_classes": EVENT_CLASSES,
                "cloud_val_acc": cloud_acc,
                "event_val_acc": event_acc,
            }, WEIGHTS_OUT)
            print(f"  saved new best checkpoint (mean_acc={score:.4f}) -> {WEIGHTS_OUT}")

    print(f"Training done. Best mean_acc={best_score:.4f}. Weights at {WEIGHTS_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud_dir", default="data/cloud")
    parser.add_argument("--events_dir", default="data/events")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.cloud_dir, args.events_dir, args.epochs, args.batch_size, args.lr, args.device)
