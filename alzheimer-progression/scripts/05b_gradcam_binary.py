#!/usr/bin/env python3
"""
STEP 5b: Grad-CAM for binary pMCI/sMCI classifier
Visualizes which brain regions drive the progression prediction.
Clinically expected regions: hippocampus, entorhinal cortex,
parahippocampal gyrus, lateral temporal lobe.

Saves:
    - Per-subject PNG: 3 orthogonal views with heatmap overlay
    - Summary figure: grid of correct vs incorrect predictions
    - Average heatmap per class (Stable vs Progressor)

Usage:
    python 05b_gradcam_binary.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG
# ============================================================
BINARY_DATASET_CSV = "/lustre/cursos/curso06/alzheimer_binary/binary_dataset_processed.csv"
BEST_CKPT          = "/lustre/cursos/curso06/alzheimer_binary/finetune/best_model.pth"
ENCODER_CKPT       = "/lustre/cursos/curso06/alzheimer_binary/ssl/encoder_pretrained.pth"
OUTPUT_DIR         = "/lustre/cursos/curso06/alzheimer_binary/gradcam"

CLASS_NAMES = ["Stable (sMCI)", "Progressor (pMCI)"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================


# ── Grad-CAM 3D ─────────────────────────────────────────────
class GradCAM3D:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model         = model
        self.gradients     = None
        self.activations   = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, volume_tensor: torch.Tensor,
                 target_class: int) -> np.ndarray:
        """Returns normalized 3D heatmap (D, H, W) in [0,1]."""
        self.model.eval()
        x = volume_tensor.unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)

        out = self.model(x)
        self.model.zero_grad()
        out[0, target_class].backward()

        # Pool gradients spatially → (C,)
        pooled = self.gradients.mean(dim=(0, 2, 3, 4))

        # Weight activations
        acts = self.activations[0].clone()
        for c in range(acts.shape[0]):
            acts[c] *= pooled[c]

        heatmap = acts.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Upsample to input size
        input_size = x.shape[2:]
        factors = tuple(s / h for s, h in zip(input_size, heatmap.shape))
        heatmap = zoom(heatmap, factors, order=1)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


def get_last_conv(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            last = m
    return last


# ── Load model ───────────────────────────────────────────────
def load_model():
    from monai.networks.nets import resnet18
    import copy

    backbone = resnet18(pretrained=False, spatial_dims=3,
                        n_input_channels=1, num_classes=128)
    encoder = nn.Sequential(*list(backbone.children())[:-1])
    pool    = nn.AdaptiveAvgPool3d(1)

    class TempSimCLR(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = encoder
            self.pool      = pool
            self.projector = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
            )

    temp = TempSimCLR()
    ssl_ckpt = torch.load(ENCODER_CKPT, map_location='cpu')
    temp.load_state_dict(ssl_ckpt['model_state'])

    class BinaryClassifier(nn.Module):
        def __init__(self, enc, pool):
            super().__init__()
            self.encoder    = enc
            self.pool       = pool
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            f = self.encoder(x)
            f = self.pool(f).view(x.size(0), -1)
            return self.classifier(f)

    model = BinaryClassifier(copy.deepcopy(temp.encoder),
                             copy.deepcopy(temp.pool)).to(DEVICE)

    ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (AUC={ckpt.get('val_auc', '?'):.3f})")
    return model


# ── Visualization ────────────────────────────────────────────
def plot_subject(volume, heatmap, true_label, pred_label,
                 subject_id, confidence, save_path):
    d, h, w = volume.shape
    views = [
        ("Axial",    volume[d//2, :, :],   heatmap[d//2, :, :]),
        ("Coronal",  volume[:, h//2, :],   heatmap[:, h//2, :]),
        ("Sagittal", volume[:, :, w//2],   heatmap[:, :, w//2]),
    ]

    correct = (true_label == pred_label)
    status  = "✓ CORRECT" if correct else "✗ WRONG"
    color   = "green" if correct else "red"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{subject_id}  |  True: {CLASS_NAMES[true_label]}  |  "
        f"Pred: {CLASS_NAMES[pred_label]} ({confidence:.0%})  [{status}]",
        fontsize=11, color=color, fontweight='bold'
    )

    for ax, (title, mri_sl, heat_sl) in zip(axes, views):
        ax.imshow(mri_sl.T, cmap='gray', origin='lower')
        ax.imshow(heat_sl.T, cmap='hot', alpha=0.45, origin='lower')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_average_heatmaps(heatmaps_by_class, save_path):
    """Plot average Grad-CAM heatmap per class across all subjects."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Average Grad-CAM Activations by Class", fontsize=13)

    for row, (cls_idx, cls_name) in enumerate(enumerate(CLASS_NAMES)):
        if cls_idx not in heatmaps_by_class or not heatmaps_by_class[cls_idx]:
            continue
        avg = np.mean(heatmaps_by_class[cls_idx], axis=0)
        d, h, w = avg.shape

        views = [
            ("Axial",    avg[d//2, :, :]),
            ("Coronal",  avg[:, h//2, :]),
            ("Sagittal", avg[:, :, w//2]),
        ]
        for col, (title, sl) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(sl.T, cmap='hot', origin='lower', vmin=0, vmax=1)
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=11, fontweight='bold')
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.axis('off')

    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'),
                 ax=axes, shrink=0.6, label='Activation intensity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Average heatmap saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"STEP 5b: Grad-CAM Binary | Device: {DEVICE}")
    print("=" * 60)

    if not Path(BEST_CKPT).exists():
        print(f"ERROR: {BEST_CKPT} not found. Run 04c first.")
        return

    model   = load_model()
    last_conv = get_last_conv(model)
    if last_conv is None:
        print("ERROR: No Conv3d found in model.")
        return

    gradcam = GradCAM3D(model, last_conv)

    df = pd.read_csv(BINARY_DATASET_CSV)
    print(f"Subjects: {len(df)}")

    heatmaps_by_class = {0: [], 1: []}
    results = []

    print("\nGenerating Grad-CAM maps...\n")

    for _, row in df.iterrows():
        subject_id = row['subject_id']
        true_label = int(row['label'])

        img    = nib.load(row['processed_path'])
        volume = img.get_fdata().astype(np.float32)
        tensor = torch.from_numpy(volume).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            out  = model(tensor.unsqueeze(0).to(DEVICE))
            pred = out.argmax(1).item()
            prob = torch.softmax(out, dim=1)[0, pred].item()

        # Grad-CAM for TRUE class (shows what model sees for the correct answer)
        heatmap_true = gradcam.generate(tensor, true_label)
        # Also for predicted class
        heatmap_pred = gradcam.generate(tensor, pred)

        heatmaps_by_class[true_label].append(heatmap_true)

        correct = "correct" if pred == true_label else "wrong"
        fname   = (f"{correct}_{subject_id}_"
                   f"true-{CLASS_NAMES[true_label].split()[0]}_"
                   f"pred-{CLASS_NAMES[pred].split()[0]}_"
                   f"{prob:.2f}.png")

        plot_subject(volume, heatmap_pred, true_label, pred,
                     subject_id, prob,
                     str(Path(OUTPUT_DIR) / fname))

        results.append({
            'subject_id':  subject_id,
            'true_label':  CLASS_NAMES[true_label],
            'pred_label':  CLASS_NAMES[pred],
            'confidence':  prob,
            'correct':     pred == true_label,
        })

        print(f"  {subject_id:<15} true={CLASS_NAMES[true_label]:<22} "
              f"pred={CLASS_NAMES[pred]:<22} conf={prob:.2%} [{correct}]")

    # Average heatmaps per class
    plot_average_heatmaps(
        heatmaps_by_class,
        str(Path(OUTPUT_DIR) / "average_heatmaps_by_class.png")
    )

    # Save prediction summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(OUTPUT_DIR) / "gradcam_predictions.csv", index=False)

    acc = results_df['correct'].mean()
    print(f"\nOverall accuracy on full dataset: {acc:.3f}")
    print(f"\n{'='*60}")
    print(f"Grad-CAM outputs saved to: {OUTPUT_DIR}")
    print()
    print("Interpreting the heatmaps:")
    print("  Clinically expected activations for Progressors:")
    print("    → Hippocampus (medial temporal lobe)")
    print("    → Entorhinal cortex")
    print("    → Parahippocampal gyrus")
    print("  If activations appear in ventricles or skull → model")
    print("  learned scanner artifacts, not pathology")
    print("=" * 60)


if __name__ == "__main__":
    main()
