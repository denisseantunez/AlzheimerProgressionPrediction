#!/usr/bin/env python3
"""
STEP 4b: SimCLR Self-Supervised Pretraining
Trains a 3D ResNet-18 encoder using SimCLR contrastive learning.
Uses ALL available preprocessed volumes (original 192 + binary 82)
WITHOUT labels — the goal is to learn general brain representations.

Architecture:
    Encoder: ResNet-18 3D (MONAI) — last FC removed
    Projection head: Linear(512→256) → ReLU → Linear(256→128)
    Loss: NT-Xent (normalized temperature-scaled cross-entropy)

After pretraining, the encoder weights are saved and used as
initialization for the binary classifier (step 04c).

Reference:
    Chen et al. 2020 — SimCLR
    Kaczmarek et al. 2025 — SimCLR for 3D brain MRI
    Elmannai et al. 2025 — SSA-Net (SimCLR + attention for AD)

Usage:
    python 04b_ssl_pretrain.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import json

# ============================================================
# CONFIG
# ============================================================
# All preprocessed volumes to use for SSL (no labels needed)
ORIGINAL_DATASET_CSV = "/lustre/cursos/curso06/alzheimer/dataset.csv"
BINARY_DATASET_CSV   = "/lustre/cursos/curso06/alzheimer_binary/binary_dataset_processed.csv"

OUTPUT_DIR     = "/lustre/cursos/curso06/alzheimer_binary/ssl"
ENCODER_CKPT   = "/lustre/cursos/curso06/alzheimer_binary/ssl/encoder_pretrained.pth"

# SimCLR hyperparameters
BATCH_SIZE     = 2      # Reducido por OOM — MI210 no aguanta 96^3 * 2 vistas * 8
ACCUM_STEPS    = 4      # Gradient accumulation: effective batch = 2*4 = 8 (igual que antes)
NUM_EPOCHS     = 100
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TEMPERATURE    = 0.07   # NT-Xent temperature (lower = harder negatives)
PROJECTION_DIM = 128    # Output dim of projection head
HIDDEN_DIM     = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================


# ── Augmentations ───────────────────────────────────────────
class SimCLRAugment3D:
    """
    Creates two randomly augmented views of the same 3D MRI volume.
    Augmentations are domain-appropriate for structural MRI.
    Based on Kaczmarek et al. 2025 and Elmannai et al. 2025.
    """
    def __call__(self, volume: torch.Tensor):
        view1 = self._augment(volume.clone())
        view2 = self._augment(volume.clone())
        return view1, view2

    def _augment(self, v: torch.Tensor) -> torch.Tensor:
        # 1. Random flip along each spatial axis (50% each)
        for axis in [1, 2, 3]:
            if torch.rand(1) > 0.5:
                v = torch.flip(v, dims=[axis])

        # 2. Random crop + resize (simulate slight FOV differences)
        if torch.rand(1) > 0.5:
            v = self._random_crop_resize(v, crop_frac=0.9)

        # 3. Gaussian noise (p=0.3)
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(v) * 0.02
            v = torch.clamp(v + noise, 0.0, 1.0)

        # 4. Intensity jitter (p=0.5) — simulates scanner variability
        if torch.rand(1) > 0.5:
            # Random brightness shift
            shift = (torch.rand(1).item() - 0.5) * 0.2
            v = torch.clamp(v + shift, 0.0, 1.0)
            # Random contrast scaling
            scale = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            mean  = v.mean()
            v = torch.clamp((v - mean) * scale + mean, 0.0, 1.0)

        # 5. Gaussian blur (p=0.2)
        if torch.rand(1) > 0.8:
            v = self._gaussian_blur(v)

        return v

    def _random_crop_resize(self, v: torch.Tensor, crop_frac: float) -> torch.Tensor:
        """Randomly crop a fraction of the volume and resize back."""
        from torch.nn.functional import interpolate
        _, d, h, w = v.shape
        cd = int(d * crop_frac)
        ch = int(h * crop_frac)
        cw = int(w * crop_frac)
        # Random start positions
        sd = torch.randint(0, d - cd + 1, (1,)).item()
        sh = torch.randint(0, h - ch + 1, (1,)).item()
        sw = torch.randint(0, w - cw + 1, (1,)).item()
        cropped = v[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        # Resize back to original
        resized = interpolate(
            cropped.unsqueeze(0), size=(d, h, w),
            mode='trilinear', align_corners=False
        ).squeeze(0)
        return resized

    def _gaussian_blur(self, v: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Simple 3D Gaussian blur via 3 separable 1D convolutions."""
        kernel_size = 5
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply along each axis separately using F.conv3d
        v = v.unsqueeze(0)  # (1, 1, D, H, W)
        pad = kernel_size // 2

        k_d = kernel_1d.view(1, 1, -1, 1, 1)
        k_h = kernel_1d.view(1, 1, 1, -1, 1)
        k_w = kernel_1d.view(1, 1, 1, 1, -1)

        v = F.conv3d(v, k_d, padding=(pad, 0, 0))
        v = F.conv3d(v, k_h, padding=(0, pad, 0))
        v = F.conv3d(v, k_w, padding=(0, 0, pad))
        return v.squeeze(0)


# ── Dataset ─────────────────────────────────────────────────
class UnlabeledMRIDataset(Dataset):
    """
    Loads preprocessed NIfTI volumes WITHOUT labels.
    Returns two augmented views of each volume for SimCLR.
    """
    def __init__(self, paths: list[str]):
        self.paths   = paths
        self.augment = SimCLRAugment3D()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path   = self.paths[idx]
        img    = nib.load(path)
        volume = img.get_fdata().astype(np.float32)
        tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
        view1, view2 = self.augment(tensor)
        return view1, view2


# ── Model ────────────────────────────────────────────────────
class SimCLREncoder(nn.Module):
    """
    ResNet-18 3D encoder + MLP projection head.
    The projection head is ONLY used during SSL training.
    For downstream tasks, only the encoder backbone is used.
    """
    def __init__(self, projection_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        from monai.networks.nets import resnet18
        backbone = resnet18(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=projection_dim,  # temp placeholder
        )

        # Remove the final FC layer — we want the 512-dim feature vector
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.pool     = nn.AdaptiveAvgPool3d(1)

        # Projection head: 512 → hidden_dim → projection_dim
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        # x: (B, 1, D, H, W)
        feat = self.encoder(x)        # (B, 512, d, h, w)
        feat = self.pool(feat)        # (B, 512, 1, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 512)
        proj = self.projector(feat)   # (B, projection_dim)
        return feat, proj

    def get_encoder_only(self):
        """Returns just the encoder (no projection head) for fine-tuning."""
        return self.encoder, self.pool


# ── NT-Xent Loss ─────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    Given a batch of N pairs, treats the 2N representations as:
        - Positive pairs: the two views of the same image
        - Negative pairs: all other 2(N-1) representations in the batch
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (N, D) — L2-normalized projections
        """
        N = z1.size(0)
        device = z1.device

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_indices = torch.arange(N, device=device)
        pos_1 = torch.cat([pos_indices + N, pos_indices])  # (2N,)

        loss = F.cross_entropy(sim, pos_1)
        return loss


# ── Training ─────────────────────────────────────────────────
def train_simclr(model, loader, optimizer, scheduler, criterion,
                 device, epoch, accum_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, (view1, view2) in enumerate(
            tqdm(loader, desc=f"  SSL epoch {epoch}", leave=False)):

        view1 = view1.to(device)
        view2 = view2.to(device)

        _, proj1 = model(view1)
        _, proj2 = model(view2)

        loss = criterion(proj1, proj2)
        # Scale loss so gradients are equivalent to a larger batch
        loss = loss / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps

        # Update weights every accum_steps batches
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
    return total_loss / len(loader)


# ── Main ─────────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"STEP 4b: SimCLR SSL Pretraining | Device: {DEVICE}")
    print("=" * 60)

    # Collect all preprocessed paths (original + binary, no labels needed)
    all_paths = []

    if Path(ORIGINAL_DATASET_CSV).exists():
        orig_df = pd.read_csv(ORIGINAL_DATASET_CSV)
        orig_paths = orig_df['processed_path'].dropna().tolist()
        all_paths.extend(orig_paths)
        print(f"Original dataset: {len(orig_paths)} volumes")
    else:
        print(f"WARNING: {ORIGINAL_DATASET_CSV} not found — skipping")

    if Path(BINARY_DATASET_CSV).exists():
        bin_df = pd.read_csv(BINARY_DATASET_CSV)
        bin_paths = bin_df['processed_path'].dropna().tolist()
        # Avoid duplicates if a subject appears in both
        bin_paths = [p for p in bin_paths if p not in all_paths]
        all_paths.extend(bin_paths)
        print(f"Binary dataset:   {len(bin_paths)} volumes (new)")
    else:
        print(f"WARNING: {BINARY_DATASET_CSV} not found — skipping")

    # Verify all paths exist
    valid_paths = [p for p in all_paths if Path(p).exists()]
    missing     = len(all_paths) - len(valid_paths)
    if missing:
        print(f"WARNING: {missing} paths not found on disk — skipping")

    print(f"\nTotal volumes for SSL pretraining: {len(valid_paths)}")
    print(f"(No labels used — pure self-supervised learning)")

    if len(valid_paths) < 10:
        print("ERROR: Not enough volumes for SSL training.")
        return

    # Dataset & loader
    dataset = UnlabeledMRIDataset(valid_paths)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=1, pin_memory=False,
                         drop_last=True)  # drop_last importante para NT-Xent

    print(f"Batch size: {BATCH_SIZE} x {ACCUM_STEPS} accumulation steps "
          f"(effective = {2*BATCH_SIZE*ACCUM_STEPS} views per update)")
    print(f"Steps per epoch: {len(loader)}")

    # Model, loss, optimizer
    model     = SimCLREncoder(PROJECTION_DIM, HIDDEN_DIM).to(DEVICE)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"\nTraining for {NUM_EPOCHS} epochs...")

    history = []
    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_simclr(model, loader, optimizer, scheduler,
                            criterion, DEVICE, epoch, accum_steps=ACCUM_STEPS)
        history.append(loss)

        if loss < best_loss:
            best_loss = loss
            # Save full model (encoder + projector)
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'loss':        loss,
                'config': {
                    'projection_dim': PROJECTION_DIM,
                    'hidden_dim':     HIDDEN_DIM,
                    'temperature':    TEMPERATURE,
                    'n_volumes':      len(valid_paths),
                }
            }, ENCODER_CKPT)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"loss={loss:.4f} | best={best_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Save training history
    with open(Path(OUTPUT_DIR) / "ssl_history.json", "w") as f:
        json.dump({"loss": history}, f)

    print(f"\n{'='*60}")
    print(f"SSL pretraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder saved: {ENCODER_CKPT}")
    print("=" * 60)
    print("\nNext step: run 04c_finetune_binary.py")


if __name__ == "__main__":
    main()