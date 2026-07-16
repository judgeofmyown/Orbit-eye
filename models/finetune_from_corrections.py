"""
Applies ground-station corrections queued in shared_data/corrections.jsonl (see the
"Correct a misclassification" section of ground_station/app.py) — a stand-in for
uplinking a small model update after ops flags mistakes, instead of waiting for a
full retrain-and-redeploy cycle.

For each correction:
  1. Copies the source frame into the appropriate data/{cloud,events}/train/<label>/
     folder, so it permanently joins the training corpus (not just used once).
  2. Fine-tunes the relevant model (cloud_classifier.pth / event_detector.pth,
     whichever exist) for a few epochs at a low learning rate on the corrected
     samples mixed with a random replay sample of existing training data — pure
     new-sample fine-tuning on a handful of images would just overfit/forget.
     (If you're running on the fusion_net backend, corrections still get folded into
     data/{cloud,events}/train/ so the next full `train_fusion.py` run picks them up
     — this script only hot-patches the two separate-model checkpoints in place.)

    python models/finetune_from_corrections.py

Clears corrections.jsonl after a successful pass (archived to corrections_applied.jsonl).
"""
import argparse
import json
import os
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cloud_classifier import TinyCloudNet, CLOUD_CLASSES
from event_detector import TinyEventNet, EVENT_CLASSES
from fusion_net import OrbitFusionNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "models", "weights")

TRAIN_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

REPLAY_MULTIPLIER = 5  # how many existing samples to replay per corrected sample


def load_corrections(shared_output: str):
    path = os.path.join(shared_output, "corrections.jsonl")
    if not os.path.exists(path):
        return [], path
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries, path


def apply_corrections_to_dataset(entries, head: str, data_dir: str):
    """Copies corrected frames into data/{cloud,events}/train/<label>/, returns the
    list of newly added file paths."""
    added = []
    relevant = [e for e in entries if e["head"] == head]
    for e in relevant:
        src = e["source_path"]
        if not os.path.exists(src):
            print(f"  WARNING: source frame missing, skipping: {src}")
            continue
        label = e["corrected_label"]
        dest_dir = os.path.join(data_dir, "train", label)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"corrected_{os.path.basename(src)}")
        shutil.copy(src, dest)
        added.append(dest)
    return added


def finetune_head(model, classes, data_dir, corrected_paths, device, epochs=3, lr=1e-4):
    if not corrected_paths:
        return False

    full_train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=TRAIN_TF)
    corrected_indices = [i for i, (p, _) in enumerate(full_train.samples)
                          if os.path.abspath(p) in {os.path.abspath(c) for c in corrected_paths}]
    n_replay = min(len(full_train) - len(corrected_indices), len(corrected_indices) * REPLAY_MULTIPLIER)
    other_indices = [i for i in range(len(full_train)) if i not in set(corrected_indices)]
    replay_indices = random.sample(other_indices, n_replay) if n_replay > 0 else []

    subset = Subset(full_train, corrected_indices + replay_indices)
    loader = DataLoader(subset, batch_size=min(16, len(subset)), shuffle=True)

    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"    finetune epoch {epoch+1}/{epochs}  loss={total_loss/len(subset):.4f}")

    model.eval()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared_output", default=os.path.join(ROOT, "shared_data"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    entries, corrections_path = load_corrections(args.shared_output)
    if not entries:
        print("No pending corrections in shared_data/corrections.jsonl.")
        return

    print(f"Applying {len(entries)} correction(s)...")

    cloud_dir = os.path.join(ROOT, "data", "cloud")
    events_dir = os.path.join(ROOT, "data", "events")
    cloud_added = apply_corrections_to_dataset(entries, "cloud", cloud_dir)
    event_added = apply_corrections_to_dataset(entries, "event", events_dir)

    cloud_path = os.path.join(WEIGHTS_DIR, "cloud_classifier.pth")
    event_path = os.path.join(WEIGHTS_DIR, "event_detector.pth")

    if cloud_added and os.path.exists(cloud_path):
        print(f"  fine-tuning cloud_classifier on {len(cloud_added)} corrected frame(s)...")
        model = TinyCloudNet(len(CLOUD_CLASSES))
        ckpt = torch.load(cloud_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        if finetune_head(model, CLOUD_CLASSES, cloud_dir, cloud_added, args.device, args.epochs, args.lr):
            torch.save({"model_state_dict": model.state_dict(), "classes": CLOUD_CLASSES,
                        "val_acc": ckpt.get("val_acc")}, cloud_path)
            print(f"  updated {cloud_path}")

    if event_added and os.path.exists(event_path):
        print(f"  fine-tuning event_detector on {len(event_added)} corrected frame(s)...")
        model = TinyEventNet(len(EVENT_CLASSES))
        ckpt = torch.load(event_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        if finetune_head(model, EVENT_CLASSES, events_dir, event_added, args.device, args.epochs, args.lr):
            torch.save({"model_state_dict": model.state_dict(), "classes": EVENT_CLASSES,
                        "val_acc": ckpt.get("val_acc")}, event_path)
            print(f"  updated {event_path}")

    if not cloud_added and not event_added:
        print("  no corrections matched an existing trained model — nothing to fine-tune.")

    # archive so re-running the script doesn't reapply the same corrections
    archive_path = os.path.join(args.shared_output, "corrections_applied.jsonl")
    with open(archive_path, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    os.remove(corrections_path)
    print(f"Done. Archived {len(entries)} correction(s) to {archive_path}.")


if __name__ == "__main__":
    main()
