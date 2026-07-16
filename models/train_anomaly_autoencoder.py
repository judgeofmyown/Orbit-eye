"""
Trains TinyAutoencoder on every "known/normal" frame across both prepared datasets —
all 3 cloud classes (clear/partly_cloudy/overcast are all legitimate sky states, not
anomalies) plus event=none. The point isn't to reconstruct clouds vs. events well,
it's to learn what *any* recognized category looks like, so frames that don't belong
to any of them reconstruct poorly and stand out.

    python models/train_anomaly_autoencoder.py --epochs 12

Saves models/weights/anomaly_autoencoder.pth with a calibrated `ood_threshold` (95th
percentile reconstruction error on held-out normal frames) baked into the checkpoint,
so inference_engine.py doesn't need to guess a cutoff.
"""
import argparse
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from anomaly_autoencoder import TinyAutoencoder

WEIGHTS_OUT = os.path.join(os.path.dirname(__file__), "weights", "anomaly_autoencoder.pth")

TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

NORMAL_SUBDIRS = [
    ("cloud", "clear"), ("cloud", "partly_cloudy"), ("cloud", "overcast"),
    ("events", "none"),
]


class FlatImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def collect_paths(data_root, split):
    paths = []
    for dataset_name, class_name in NORMAL_SUBDIRS:
        d = os.path.join(data_root, dataset_name, split, class_name)
        if os.path.isdir(d):
            paths.extend(glob.glob(os.path.join(d, "*.jpg")) +
                         glob.glob(os.path.join(d, "*.jpeg")) +
                         glob.glob(os.path.join(d, "*.png")))
    return paths


def train(data_root, epochs, batch_size, lr, device):
    train_paths = collect_paths(data_root, "train")
    val_paths = collect_paths(data_root, "val")
    print(f"normal frames: {len(train_paths)} train, {len(val_paths)} val")

    train_loader = DataLoader(FlatImageDataset(train_paths, TF), batch_size=batch_size,
                               shuffle=True, num_workers=4)
    val_loader = DataLoader(FlatImageDataset(val_paths, TF), batch_size=batch_size,
                             shuffle=False, num_workers=2)

    model = TinyAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_paths)

        model.eval()
        val_loss_total = 0.0
        per_sample_errors = []
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                recon = model(imgs)
                per_pixel = torch.mean((recon - imgs) ** 2, dim=(1, 2, 3))
                per_sample_errors.extend(per_pixel.cpu().tolist())
                val_loss_total += per_pixel.sum().item()
        val_loss = val_loss_total / len(val_paths)

        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            errors_sorted = sorted(per_sample_errors)
            threshold = errors_sorted[int(0.95 * len(errors_sorted))]
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "ood_threshold": threshold,
            }, WEIGHTS_OUT)
            print(f"  saved new best checkpoint (val_loss={val_loss:.5f}, "
                  f"ood_threshold={threshold:.5f}) -> {WEIGHTS_OUT}")

    print(f"Training done. Best val_loss={best_val_loss:.5f}. Weights at {WEIGHTS_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.data_root, args.epochs, args.batch_size, args.lr, args.device)
