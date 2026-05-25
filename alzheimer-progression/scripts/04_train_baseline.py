#!/usr/bin/env python3
"""
STEP 4: Train baseline 3D CNN classifier
Architecture: ResNet-18 3D (from MONAI)
Task: 4-class classification (CN=0, EMCI=1, LMCI=2, AD=3)

Handles class imbalance via WeightedRandomSampler.
Logs to TensorBoard.

Usage:
    python 04_train_baseline.py
    # Monitor with: tensorboard --logdir runs/
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                             confusion_matrix)
import matplotlib.pyplot as plt
import json

# ============================================================
# CONFIGURE
# ============================================================
DATASET_CSV  = "/lustre/cursos/curso06/alzheimer/dataset.csv"
OUTPUT_DIR   = "/lustre/cursos/curso06/alzheimer/runs/baseline"
CHECKPOINT   = "/lustre/cursos/curso06/alzheimer/checkpoints/baseline_best.pth"

NUM_CLASSES  = 4
BATCH_SIZE   = 4     # Keep small for 96^3 volumes
NUM_EPOCHS   = 500
LR           = 1e-4
WEIGHT_DECAY = 1e-5
INPUT_SHAPE  = (96, 96, 96)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================


# ── Dataset ────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        """Simple augmentations for 3D volumes."""
        # Random flip along each axis
        for axis in [1, 2, 3]:  # C,D,H,W → skip channel
            if torch.rand(1) > 0.5:
                volume = torch.flip(volume, dims=[axis])

        # Random intensity shift (±5% of range)
        if torch.rand(1) > 0.5:
            shift = (torch.rand(1) - 0.5) * 0.1
            volume = torch.clamp(volume + shift, 0.0, 1.0)

        # Random gaussian noise
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(volume) * 0.01
            volume = torch.clamp(volume + noise, 0.0, 1.0)

        return volume

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = row["processed_path"]
        label = int(row["label"])

        # Load NIfTI
        img    = nib.load(path)
        volume = img.get_fdata().astype(np.float32)

        # Add channel dimension: (D,H,W) → (1,D,H,W)
        volume = torch.from_numpy(volume).unsqueeze(0)

        if self.augment:
            volume = self._augment(volume)

        return volume, label


# ── Model ───────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """3D ResNet-18 from MONAI."""
    try:
        from monai.networks.nets import resnet18
        model = resnet18(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
        )
    except ImportError:
        # Fallback: simple 3D CNN if MONAI not available
        print("MONAI not found, using simple 3D CNN fallback")
        model = Simple3DCNN(num_classes)
    return model


class Simple3DCNN(nn.Module):
    """Simple fallback if MONAI is unavailable."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training helpers ────────────────────────────────────────
def get_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Upsample minority classes so each batch is roughly balanced."""
    labels  = df["label"].values
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for volumes, labels in tqdm(loader, desc="  train", leave=False):
        volumes = volumes.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        outputs = model(volumes)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels  = labels.to(device)

        outputs = model(volumes)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return total_loss / n, acc, bal_acc, all_preds, all_labels


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"STEP 4: Baseline Training | Device: {DEVICE}")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATASET_CSV)
    print(f"Total subjects: {len(df)}")

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Datasets & loaders
    train_ds = MRIDataset(train_df, augment=True)
    val_ds   = MRIDataset(val_df,   augment=False)
    test_ds  = MRIDataset(test_df,  augment=False)

    sampler    = get_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=1)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=1)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=1)

    # Model
    model     = build_model(NUM_CLASSES).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss: weighted to handle remaining imbalance
    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = torch.FloatTensor(1.0 / class_counts).to(DEVICE)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_bal_acc": []}

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_bal_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)

        # Save best model
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_bal_acc": val_bal_acc,
            }, CHECKPOINT)
            flag = " ← best"
        else:
            flag = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} "
                  f"bal_acc={val_bal_acc:.3f}{flag}")

    # Final evaluation on test set
    print("\n--- Test Set Evaluation (best checkpoint) ---")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc, test_bal_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE)

    class_names = ["CN", "EMCI", "LMCI", "AD"]
    print(f"Test accuracy:          {test_acc:.4f}")
    print(f"Test balanced accuracy: {test_bal_acc:.4f}")
    print("\nClassification report:")
    unique_labels = sorted(set(test_labels))
    used_names = [class_names[i] for i in unique_labels]
    print(classification_report(test_labels, test_preds,
                                labels=unique_labels,
                                target_names=used_names, digits=3))

    # Save outputs
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        Path(OUTPUT_DIR) / "confusion_matrix.png"
    )

    with open(Path(OUTPUT_DIR) / "history.json", "w") as f:
        json.dump(history, f)

    results = {
        "test_accuracy":          test_acc,
        "test_balanced_accuracy": test_bal_acc,
        "best_val_bal_acc":       best_val_acc,
        "epochs_trained":         NUM_EPOCHS,
    }
    with open(Path(OUTPUT_DIR) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best val balanced acc: {best_val_acc:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nNext step: run 05_gradcam.py")


if __name__ == "__main__":
    main()
