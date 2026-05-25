#!/usr/bin/env python3
"""
STEP 4c: Fine-tune binary classifier (pMCI vs sMCI)
Loads the SSL-pretrained encoder from step 4b and fine-tunes it
for binary progression prediction: Stable(0) vs Progressor(1).

Uses 5-fold cross-validation because the dataset is small (~82 subjects).

Outputs per fold:
    - Best checkpoint
    - Classification metrics (AUC, sensitivity, specificity, balanced acc)
Final output:
    - Mean ± std across folds
    - Best fold checkpoint for Grad-CAM

Also runs a SUPERVISED BASELINE (same architecture, trained from scratch)
for comparison — this shows the benefit of SSL pretraining.

Usage:
    python 04c_finetune_binary.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# CONFIG
# ============================================================
BINARY_DATASET_CSV = "/lustre/cursos/curso06/alzheimer_binary/binary_dataset_processed.csv"
ENCODER_CKPT       = "/lustre/cursos/curso06/alzheimer_binary/ssl/encoder_pretrained.pth"
OUTPUT_DIR         = "/lustre/cursos/curso06/alzheimer_binary/finetune"
BEST_CKPT          = "/lustre/cursos/curso06/alzheimer_binary/finetune/best_model.pth"

N_FOLDS       = 5
BATCH_SIZE    = 4
NUM_EPOCHS    = 100
LR_FROZEN     = 1e-3   # LR when encoder is frozen (linear probe phase)
LR_FINETUNE   = 1e-4   # LR when unfreezing encoder
WEIGHT_DECAY  = 1e-4
FREEZE_EPOCHS = 15     # Epochs to train only the head before unfreezing encoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Stable", "Progressor"]
# ============================================================


# ── Dataset ─────────────────────────────────────────────────
class BinaryMRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, v: torch.Tensor) -> torch.Tensor:
        for axis in [1, 2, 3]:
            if torch.rand(1) > 0.5:
                v = torch.flip(v, dims=[axis])
        if torch.rand(1) > 0.5:
            shift = (torch.rand(1).item() - 0.5) * 0.1
            v = torch.clamp(v + shift, 0.0, 1.0)
        if torch.rand(1) > 0.5:
            v = torch.clamp(v + torch.randn_like(v) * 0.01, 0.0, 1.0)
        return v

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img    = nib.load(row['processed_path'])
        volume = img.get_fdata().astype(np.float32)
        tensor = torch.from_numpy(volume).unsqueeze(0)
        if self.augment:
            tensor = self._augment(tensor)
        return tensor, int(row['label'])


# ── Model ────────────────────────────────────────────────────
class BinaryClassifier(nn.Module):
    """
    SSL encoder backbone + binary classification head.
    The encoder is loaded from the SSL pretraining checkpoint.
    """
    def __init__(self, encoder, pool):
        super().__init__()
        self.encoder    = encoder
        self.pool       = pool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),   # binary: Stable vs Progressor
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)  # (B, 512)
        return self.classifier(feat)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


def load_ssl_encoder():
    """Load encoder from SSL checkpoint."""
    from monai.networks.nets import resnet18

    backbone = resnet18(pretrained=False, spatial_dims=3,
                        n_input_channels=1, num_classes=128)
    encoder = nn.Sequential(*list(backbone.children())[:-1])
    pool    = nn.AdaptiveAvgPool3d(1)

    # Load SSL weights into a temp SimCLREncoder to extract encoder
    from types import SimpleNamespace

    class TempSimCLR(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = encoder
            self.pool      = pool
            self.projector = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
            )

    temp = TempSimCLR()
    ckpt = torch.load(ENCODER_CKPT, map_location='cpu')
    temp.load_state_dict(ckpt['model_state'])
    print(f"  Loaded SSL encoder from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    return temp.encoder, temp.pool


def build_supervised_baseline():
    """Fresh ResNet-18 with no SSL pretraining — for comparison."""
    from monai.networks.nets import resnet18
    backbone = resnet18(pretrained=False, spatial_dims=3,
                        n_input_channels=1, num_classes=128)
    encoder = nn.Sequential(*list(backbone.children())[:-1])
    pool    = nn.AdaptiveAvgPool3d(1)
    return BinaryClassifier(encoder, pool)


# ── Training helpers ─────────────────────────────────────────
def get_sampler(labels):
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.DoubleTensor(weights),
                                 len(weights), replacement=True)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        total_loss  += loss.item() * len(y)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_probs.extend(probs)
        all_labels.extend(y.cpu().numpy())
    n = len(all_labels)
    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    auc     = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / n, acc, bal_acc, auc, all_preds, all_probs, all_labels


def run_cv(df, model_fn, tag, output_dir):
    """
    Run 5-fold stratified cross-validation.
    model_fn() returns a fresh model for each fold.
    tag is a string label ('ssl' or 'supervised')
    """
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    labels  = df['label'].values
    subjects = df.index.values

    fold_results = []
    best_overall_auc   = 0.0
    best_overall_ckpt  = None

    print(f"\n{'='*60}")
    print(f"Cross-validation: {tag.upper()} | {N_FOLDS} folds")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_ds = BinaryMRIDataset(train_df, augment=True)
        val_ds   = BinaryMRIDataset(val_df,   augment=False)

        sampler      = get_sampler(train_df['label'].values)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=1)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=1)

        model = model_fn().to(DEVICE)

        # Class weights for loss
        counts       = train_df['label'].value_counts().sort_index().values
        class_weights = torch.FloatTensor(1.0 / counts).to(DEVICE)
        class_weights = class_weights / class_weights.sum()
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        fold_ckpt   = Path(output_dir) / f"{tag}_fold{fold}_best.pth"
        best_val_auc = 0.0

        # Phase 1: freeze encoder, train only head
        if hasattr(model, 'freeze_encoder'):
            model.freeze_encoder()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FROZEN, weight_decay=WEIGHT_DECAY
            )
            print(f"  Phase 1 (frozen encoder, {FREEZE_EPOCHS} epochs):")
            for epoch in range(1, FREEZE_EPOCHS + 1):
                tl, ta = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
                if epoch == FREEZE_EPOCHS:
                    _, va, vba, vauc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
                    print(f"    Epoch {epoch}: train_loss={tl:.4f} acc={ta:.3f} "
                          f"| val_acc={va:.3f} bal_acc={vba:.3f} AUC={vauc:.3f}")

            # Phase 2: unfreeze encoder, lower LR
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
            remaining = NUM_EPOCHS - FREEZE_EPOCHS
            print(f"  Phase 2 (full fine-tune, {remaining} epochs):")
        else:
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
            remaining = NUM_EPOCHS

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-6
        )

        for epoch in range(1, remaining + 1):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            _, va, vba, vauc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            if vauc > best_val_auc:
                best_val_auc = vauc
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'val_auc': vauc, 'val_bal_acc': vba}, str(fold_ckpt))

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}: train_loss={tl:.4f} acc={ta:.3f} "
                      f"| val_acc={va:.3f} bal_acc={vba:.3f} AUC={vauc:.3f}")

        # Load best checkpoint for this fold
        ckpt = torch.load(str(fold_ckpt), map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])

        _, _, val_bal_acc, val_auc, val_preds, val_probs, val_labels = \
            evaluate(model, val_loader, criterion, DEVICE)

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_result = {
            'fold':        fold,
            'val_bal_acc': val_bal_acc,
            'val_auc':     val_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'n_val':       len(val_labels),
        }
        fold_results.append(fold_result)
        print(f"  → Fold {fold} best: bal_acc={val_bal_acc:.3f} "
              f"AUC={val_auc:.3f} sens={sensitivity:.3f} spec={specificity:.3f}")

        # Track best overall model (for Grad-CAM)
        if val_auc > best_overall_auc:
            best_overall_auc = val_auc
            best_overall_ckpt = str(fold_ckpt)

    # Summary
    results_df = pd.DataFrame(fold_results)
    print(f"\n{'='*60}")
    print(f"CV Summary ({tag.upper()}):")
    print(f"  Balanced Acc: {results_df['val_bal_acc'].mean():.3f} ± "
          f"{results_df['val_bal_acc'].std():.3f}")
    print(f"  AUC-ROC:      {results_df['val_auc'].mean():.3f} ± "
          f"{results_df['val_auc'].std():.3f}")
    print(f"  Sensitivity:  {results_df['sensitivity'].mean():.3f} ± "
          f"{results_df['sensitivity'].std():.3f}")
    print(f"  Specificity:  {results_df['specificity'].mean():.3f} ± "
          f"{results_df['specificity'].std():.3f}")
    print(f"{'='*60}")

    # Save results
    results_df.to_csv(Path(output_dir) / f"{tag}_cv_results.csv", index=False)

    return results_df, best_overall_ckpt


# ── Main ─────────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"STEP 4c: Binary Fine-tuning | Device: {DEVICE}")
    print("=" * 60)

    if not Path(BINARY_DATASET_CSV).exists():
        print(f"ERROR: {BINARY_DATASET_CSV} not found. Run 03c first.")
        return

    df = pd.read_csv(BINARY_DATASET_CSV)
    print(f"Dataset: {len(df)} subjects")
    print(f"  Progressors (label=1): {(df['label']==1).sum()}")
    print(f"  Stables     (label=0): {(df['label']==0).sum()}")

    # ── SSL fine-tune ──────────────────────────────────────
    if Path(ENCODER_CKPT).exists():
        print(f"\nLoading SSL pretrained encoder from {ENCODER_CKPT}...")
        ssl_encoder, ssl_pool = load_ssl_encoder()

        def ssl_model_fn():
            enc_copy  = type(ssl_encoder)()  # fresh copy
            # Deepcopy to avoid sharing weights between folds
            import copy
            enc  = copy.deepcopy(ssl_encoder)
            pool = copy.deepcopy(ssl_pool)
            return BinaryClassifier(enc, pool)

        ssl_results, ssl_best_ckpt = run_cv(df, ssl_model_fn, 'ssl', OUTPUT_DIR)
    else:
        print(f"WARNING: SSL encoder not found at {ENCODER_CKPT}")
        print("Run 04b_ssl_pretrain.py first.")
        print("Continuing with supervised baseline only...\n")
        ssl_results    = None
        ssl_best_ckpt  = None

    # ── Supervised baseline (from scratch) ────────────────
    print("\n--- Running supervised baseline for comparison ---")
    sup_results, sup_best_ckpt = run_cv(
        df, build_supervised_baseline, 'supervised', OUTPUT_DIR
    )

    # ── Compare SSL vs Supervised ──────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: SSL fine-tune vs Supervised from scratch")
    print("=" * 60)
    print(f"{'Metric':<20} {'SSL':>12} {'Supervised':>12} {'Δ (SSL-Sup)':>12}")
    print("-" * 56)

    metrics = ['val_bal_acc', 'val_auc', 'sensitivity', 'specificity']
    comparison = {}
    for m in metrics:
        if ssl_results is not None:
            ssl_m  = ssl_results[m].mean()
            ssl_s  = ssl_results[m].std()
        else:
            ssl_m, ssl_s = float('nan'), float('nan')
        sup_m  = sup_results[m].mean()
        sup_s  = sup_results[m].std()
        delta  = ssl_m - sup_m
        print(f"{m:<20} {ssl_m:.3f}±{ssl_s:.3f}  {sup_m:.3f}±{sup_s:.3f}  "
              f"{delta:+.3f}")
        comparison[m] = {'ssl': ssl_m, 'ssl_std': ssl_s,
                         'supervised': sup_m, 'supervised_std': sup_s,
                         'delta': delta}

    # Save comparison
    with open(Path(OUTPUT_DIR) / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Copy best overall model for Grad-CAM
    if ssl_best_ckpt:
        import shutil
        shutil.copy(ssl_best_ckpt, BEST_CKPT)
        print(f"\nBest SSL model saved to: {BEST_CKPT}")

    print("\nNext step: run 05b_gradcam_binary.py")


if __name__ == "__main__":
    main()
