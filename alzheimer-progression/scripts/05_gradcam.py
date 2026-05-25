#!/usr/bin/env python3
"""
STEP 5: Grad-CAM interpretability
Visualizes which brain regions the model is using to make predictions.
Saves overlaid heatmaps for correct and incorrect predictions.

Usage:
    python 05_gradcam.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ============================================================
# CONFIGURE
# ============================================================
DATASET_CSV  = "/lustre/cursos/curso06/alzheimer/dataset.csv"
CHECKPOINT   = "/lustre/cursos/curso06/alzheimer/checkpoints/baseline_best.pth"
OUTPUT_DIR   = "/lustre/cursos/curso06/alzheimer/gradcam"
NUM_SAMPLES  = 8   # How many subjects to visualize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CN", "EMCI", "LMCI", "AD"]
# ============================================================


class GradCAM3D:
    """
    Grad-CAM for 3D CNNs.
    Hooks into the last convolutional layer and computes
    gradient-weighted feature maps.
    """
    def __init__(self, model: torch.nn.Module, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: int) -> np.ndarray:
        """
        Returns a 3D heatmap (D, H, W) normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        # Zero gradients and backprop on target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Pool gradients over spatial dims: (C,D,H,W) → (C,)
        pooled_grads = self.gradients.mean(dim=(0, 2, 3, 4))

        # Weight activations by pooled gradients
        activations = self.activations[0]  # (C, D, H, W)
        for c in range(activations.shape[0]):
            activations[c] *= pooled_grads[c]

        # Average over channels, ReLU, normalize
        heatmap = activations.mean(dim=0).cpu().numpy()  # (D, H, W)
        heatmap = np.maximum(heatmap, 0)

        # Upsample to input size using scipy
        from scipy.ndimage import zoom
        input_size = input_tensor.shape[2:]  # (D, H, W)
        factors    = tuple(s / h for s, h in zip(input_size, heatmap.shape))
        heatmap    = zoom(heatmap, factors, order=1)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


def get_last_conv_layer(model):
    """Find the last Conv3d layer in the model for Grad-CAM."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv = module
    return last_conv


def visualize_gradcam(volume: np.ndarray, heatmap: np.ndarray,
                      true_label: int, pred_label: int,
                      subject_id: str, save_path: str):
    """
    Saves a figure with 3 views (axial, coronal, sagittal)
    showing the MRI with Grad-CAM overlay.
    """
    # Take middle slices along each axis
    d, h, w = volume.shape
    slices = {
        "Axial (D)":    (volume[d//2, :, :],    heatmap[d//2, :, :]),
        "Coronal (H)":  (volume[:, h//2, :],    heatmap[:, h//2, :]),
        "Sagittal (W)": (volume[:, :, w//2],    heatmap[:, :, w//2]),
    }

    correct = (true_label == pred_label)
    status  = "CORRECT" if correct else "WRONG"
    color   = "green" if correct else "red"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"{subject_id} | True: {CLASS_NAMES[true_label]} | "
        f"Pred: {CLASS_NAMES[pred_label]} [{status}]",
        fontsize=11, color=color
    )

    for ax, (title, (mri_slice, heat_slice)) in zip(axes, slices.items()):
        ax.imshow(mri_slice.T, cmap="gray", origin="lower")
        ax.imshow(heat_slice.T, cmap="jet", alpha=0.4, origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print(f"STEP 5: Grad-CAM Visualization | Device: {DEVICE}")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load model
    from scripts.train_baseline import build_model  # reuse model builder
    try:
        from scripts._04_train_baseline import build_model
    except ImportError:
        # Inline import fallback
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        train_mod  = import_module("04_train_baseline")
        build_model = train_mod.build_model

    model = build_model(num_classes=4).to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_bal_acc={ckpt['val_bal_acc']:.4f})")

    # Set up Grad-CAM
    last_conv = get_last_conv_layer(model)
    if last_conv is None:
        print("ERROR: Could not find Conv3d layer in model.")
        return
    gradcam = GradCAM3D(model, last_conv)

    # Load test set
    df      = pd.read_csv(DATASET_CSV)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test subjects: {len(test_df)}")

    # Sample subjects (mix of groups)
    sample = test_df.groupby("group").apply(
        lambda g: g.sample(min(len(g), NUM_SAMPLES // 4), random_state=42)
    ).reset_index(drop=True)

    print(f"Generating Grad-CAM for {len(sample)} subjects...\n")

    for _, row in sample.iterrows():
        subject_id = row["subject_id"]
        true_label = int(row["label"])

        # Load volume
        img    = nib.load(row["processed_path"])
        volume = img.get_fdata().astype(np.float32)
        tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)

        # Get prediction
        with torch.no_grad():
            output = model(tensor.unsqueeze(0).to(DEVICE))
            pred_label = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_label].item()

        # Generate heatmap for PREDICTED class
        heatmap = gradcam.generate(tensor, pred_label)

        correct = "correct" if pred_label == true_label else "wrong"
        save_path = (Path(OUTPUT_DIR) /
                     f"{correct}_{subject_id}_"
                     f"true{CLASS_NAMES[true_label]}_"
                     f"pred{CLASS_NAMES[pred_label]}_"
                     f"conf{confidence:.2f}.png")

        visualize_gradcam(
            volume, heatmap, true_label, pred_label,
            subject_id, str(save_path)
        )
        print(f"  {subject_id}: true={CLASS_NAMES[true_label]} "
              f"pred={CLASS_NAMES[pred_label]} ({confidence:.2%}) [{correct}]")

    print(f"\n{'='*60}")
    print(f"Grad-CAM images saved to: {OUTPUT_DIR}")
    print("Look for activations in:")
    print("  - Hippocampus (temporal lobe, medial)")
    print("  - Entorhinal cortex")
    print("  - Lateral ventricles (enlarged in AD)")
    print("=" * 60)


if __name__ == "__main__":
    main()
