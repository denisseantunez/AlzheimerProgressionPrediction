#!/usr/bin/env python3
"""
STEP 3c: Convert + preprocess binary dataset
Runs dcm2niix + normalization + resize on the Progressors/Stables images.
Reuses the same preprocessing logic as the original pipeline (steps 1+2)
but targeted at the binary dataset only.

Reads:  /lustre/cursos/curso06/alzheimer_binary/binary_dataset_raw.csv
Writes: /lustre/cursos/curso06/alzheimer_binary/processed/  (NIfTI files)
        /lustre/cursos/curso06/alzheimer_binary/binary_dataset_processed.csv

Usage:
    python 03c_preprocess_binary.py
"""

import os
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom

# ============================================================
# PATHS
# ============================================================
BINARY_RAW_CSV  = "/lustre/cursos/curso06/alzheimer_binary/binary_dataset_raw.csv"
NIFTI_DIR       = "/lustre/cursos/curso06/alzheimer_binary/nifti"
PROCESSED_DIR   = "/lustre/cursos/curso06/alzheimer_binary/processed"
OUTPUT_CSV      = "/lustre/cursos/curso06/alzheimer_binary/binary_dataset_processed.csv"

TARGET_SHAPE    = (96, 96, 96)
# ============================================================


# ── DICOM → NIfTI ──────────────────────────────────────────
def convert_dicom_to_nifti(dcm_folder: str, subject_id: str,
                            image_id: str, output_dir: str) -> dict:
    """Run dcm2niix on a single subject folder."""
    out_path = Path(output_dir) / subject_id
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", image_id,
        "-o", str(out_path),
        "-s", "y",
        dcm_folder
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        nifti_files = list(out_path.glob(f"{image_id}*.nii.gz"))

        if nifti_files:
            return {"subject_id": subject_id, "image_id": image_id,
                    "nifti_path": str(nifti_files[0]), "status": "ok", "error": ""}
        else:
            # Try without -s flag (some sequences need it off)
            cmd2 = ["dcm2niix", "-z", "y", "-f", image_id,
                    "-o", str(out_path), dcm_folder]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)
            nifti_files2 = list(out_path.glob(f"{image_id}*.nii.gz"))
            if nifti_files2:
                return {"subject_id": subject_id, "image_id": image_id,
                        "nifti_path": str(nifti_files2[0]), "status": "ok", "error": ""}
            return {"subject_id": subject_id, "image_id": image_id,
                    "nifti_path": "", "status": "failed_no_output",
                    "error": result.stderr[:300]}

    except subprocess.TimeoutExpired:
        return {"subject_id": subject_id, "image_id": image_id,
                "nifti_path": "", "status": "timeout", "error": ">5min"}
    except Exception as e:
        return {"subject_id": subject_id, "image_id": image_id,
                "nifti_path": "", "status": "error", "error": str(e)}


# ── Preprocessing ───────────────────────────────────────────
def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization, clip ±3σ, rescale to [0,1]."""
    mean, std = volume.mean(), volume.std()
    if std < 1e-8:
        return volume
    v = (volume - mean) / std
    v = np.clip(v, -3.0, 3.0)
    v = (v + 3.0) / 6.0
    return v.astype(np.float32)


def resize_volume(volume: np.ndarray, target: tuple) -> np.ndarray:
    factors = tuple(t / s for t, s in zip(target, volume.shape))
    return zoom(volume, factors, order=1)


def preprocess_nifti(nifti_path: str, subject_id: str,
                     processed_dir: str) -> dict:
    """Normalize + resize a single NIfTI file."""
    out_subdir = Path(processed_dir) / subject_id
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{subject_id}_processed.nii.gz"

    try:
        img    = nib.load(nifti_path)
        volume = img.get_fdata()
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        if min(volume.shape) < 32:
            return {"subject_id": subject_id, "processed_path": "",
                    "status": "failed_too_small",
                    "error": f"shape {volume.shape}"}

        volume = normalize_volume(volume)
        volume = resize_volume(volume, TARGET_SHAPE)

        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(out_path))
        return {"subject_id": subject_id,
                "processed_path": str(out_path),
                "status": "ok", "error": ""}
    except Exception as e:
        return {"subject_id": subject_id, "processed_path": "",
                "status": "error", "error": str(e)[:200]}


# ── Main ────────────────────────────────────────────────────
def main():
    for d in [NIFTI_DIR, PROCESSED_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 3c: Convert + Preprocess binary dataset")
    print(f"Target shape: {TARGET_SHAPE}")
    print("=" * 60)

    if not Path(BINARY_RAW_CSV).exists():
        print(f"ERROR: {BINARY_RAW_CSV} not found. Run 03b first.")
        return

    df = pd.read_csv(BINARY_RAW_CSV)
    print(f"Loaded {len(df)} subjects")
    print(f"  Progressors: {(df['label']==1).sum()}")
    print(f"  Stables:     {(df['label']==0).sum()}")

    # ── Step A: DICOM → NIfTI ──────────────────────────────
    print(f"\n--- Step A: DICOM → NIfTI ---")
    conv_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        r = convert_dicom_to_nifti(
            row['dcm_folder'], row['subject_id'],
            row['image_id'], NIFTI_DIR
        )
        conv_results.append(r)

    conv_df = pd.DataFrame(conv_results)
    ok_conv  = conv_df[conv_df['status'] == 'ok']
    fail_conv = conv_df[conv_df['status'] != 'ok']

    print(f"\nConversion: {len(ok_conv)}/{len(df)} OK")
    if len(fail_conv):
        print(f"Failed ({len(fail_conv)}):")
        for _, r in fail_conv.iterrows():
            print(f"  {r['subject_id']}: {r['status']} | {r['error'][:80]}")

    # ── Step B: Preprocess ─────────────────────────────────
    print(f"\n--- Step B: Normalize + Resize ---")
    preproc_results = []

    for _, row in tqdm(ok_conv.iterrows(), total=len(ok_conv), desc="Preprocessing"):
        r = preprocess_nifti(row['nifti_path'], row['subject_id'], PROCESSED_DIR)
        preproc_results.append(r)

    preproc_df = pd.DataFrame(preproc_results)
    ok_preproc  = preproc_df[preproc_df['status'] == 'ok']
    fail_preproc = preproc_df[preproc_df['status'] != 'ok']

    print(f"\nPreprocessing: {len(ok_preproc)}/{len(ok_conv)} OK")
    if len(fail_preproc):
        print(f"Failed ({len(fail_preproc)}):")
        for _, r in fail_preproc.iterrows():
            print(f"  {r['subject_id']}: {r['status']} | {r['error']}")

    # ── Merge and save final CSV ───────────────────────────
    print(f"\n--- Saving final dataset CSV ---")
    final = df.merge(
        preproc_df[preproc_df['status'] == 'ok'][['subject_id', 'processed_path']],
        on='subject_id', how='inner'
    )

    final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Final dataset: {OUTPUT_CSV}")
    print(f"Total usable: {len(final)}")
    print(f"  Progressors (label=1): {(final['label']==1).sum()}")
    print(f"  Stables     (label=0): {(final['label']==0).sum()}")
    print("=" * 60)
    print("\nNext step: run 04b_ssl_pretrain.py")


if __name__ == "__main__":
    main()
