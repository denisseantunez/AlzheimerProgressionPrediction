#!/usr/bin/env python3
"""
STEP 2: Preprocessing pipeline
For each NIfTI volume:
  1. Load and verify dimensions
  2. Normalize intensity (z-score per volume)
  3. Resize to uniform 96x96x96
  4. Save preprocessed volume

NOTE: We skip skull stripping for now (requires FSL/FreeSurfer on cluster).
The model can still learn without it — just adds some noise.
We can add skull stripping later if YUCA has FSL available.

Usage:
    python 02_preprocess.py
"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURE THESE PATHS
# ============================================================
NIFTI_DIR      = "/lustre/cursos/curso06/alzheimer/nifti"
PROCESSED_DIR  = "/lustre/cursos/curso06/alzheimer/processed"
CONVERSION_LOG = "/lustre/cursos/curso06/alzheimer/conversion_log.csv"
PREPROC_LOG    = "/lustre/cursos/curso06/alzheimer/preproc_log.csv"

TARGET_SHAPE   = (96, 96, 96)   # resize target — fits in most GPUs
# ============================================================


def resize_volume(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize a 3D volume to target_shape using zoom (scipy).
    No fancy registration — just isotropic zoom.
    """
    from scipy.ndimage import zoom

    factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    )
    resized = zoom(volume, factors, order=1)  # order=1 = linear interp
    return resized


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per volume.
    Clips extreme values (artifacts) at ±3 std.
    """
    mean = volume.mean()
    std  = volume.std()

    if std < 1e-8:
        return volume  # avoid div by zero for empty volumes

    normalized = (volume - mean) / std
    normalized = np.clip(normalized, -3.0, 3.0)

    # Rescale to [0, 1] for model input
    normalized = (normalized + 3.0) / 6.0
    return normalized.astype(np.float32)


def preprocess_single(nifti_path: str, output_dir: str) -> dict:
    """
    Load, normalize, resize and save a single NIfTI volume.
    """
    nifti_path = Path(nifti_path)
    subject_id = nifti_path.parent.name
    filename   = nifti_path.stem.replace(".nii", "") + "_processed.nii.gz"
    out_subdir = Path(output_dir) / subject_id
    out_path   = out_subdir / filename

    out_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # Load
        img    = nib.load(str(nifti_path))
        volume = img.get_fdata()

        original_shape = volume.shape

        # Handle 4D volumes (take first volume)
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]

        # Skip if too small (bad conversion)
        if min(volume.shape) < 32:
            return {
                "subject_id":     subject_id,
                "nifti_path":     str(nifti_path),
                "processed_path": "",
                "original_shape": str(original_shape),
                "status":         "failed_too_small",
                "error":          f"shape {volume.shape} too small",
            }

        # Normalize
        volume = normalize_volume(volume)

        # Resize
        volume = resize_volume(volume, TARGET_SHAPE)

        # Save as NIfTI (keeps compatibility with visualization tools)
        new_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(new_img, str(out_path))

        return {
            "subject_id":     subject_id,
            "nifti_path":     str(nifti_path),
            "processed_path": str(out_path),
            "original_shape": str(original_shape),
            "status":         "ok",
            "error":          "",
        }

    except Exception as e:
        return {
            "subject_id":     subject_id,
            "nifti_path":     str(nifti_path),
            "processed_path": "",
            "original_shape": "",
            "status":         "error",
            "error":          str(e)[:200],
        }


def main():
    print("=" * 60)
    print("STEP 2: Preprocessing")
    print(f"Target shape: {TARGET_SHAPE}")
    print("=" * 60)

    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    # Load conversion log to get all nifti paths
    if not Path(CONVERSION_LOG).exists():
        print(f"ERROR: conversion log not found at {CONVERSION_LOG}")
        print("Run 01_dicom_to_nifti.py first.")
        return

    conv_df = pd.read_csv(CONVERSION_LOG)
    ok_scans = conv_df[conv_df["status"] == "ok"]
    print(f"Found {len(ok_scans)} successfully converted NIfTI files")

    results = []
    for _, row in tqdm(ok_scans.iterrows(), total=len(ok_scans),
                       desc="Preprocessing"):
        r = preprocess_single(row["nifti_path"], PROCESSED_DIR)
        results.append(r)

    # Save log
    df = pd.DataFrame(results)
    df.to_csv(PREPROC_LOG, index=False)

    # Summary
    ok     = df[df["status"] == "ok"]
    failed = df[df["status"] != "ok"]

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(ok)}/{len(df)} preprocessed successfully")
    print(f"Output shape: {TARGET_SHAPE} for all volumes")
    print(f"Log saved to: {PREPROC_LOG}")

    if len(failed):
        print(f"\nFAILED ({len(failed)}):")
        for _, f in failed.iterrows():
            print(f"  {f['subject_id']} | {f['status']} | {f['error']}")

    print("=" * 60)
    print("\nNext step: run 03_build_dataset.py")


if __name__ == "__main__":
    main()
