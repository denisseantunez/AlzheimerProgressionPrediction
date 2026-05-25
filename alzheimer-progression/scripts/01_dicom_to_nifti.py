#!/usr/bin/env python3
"""
STEP 1: DICOM → NIfTI conversion
Converts all subject folders from DICOM (.dcm) to NIfTI (.nii.gz)
One .nii.gz per subject.

Usage:
    python 01_dicom_to_nifti.py
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURE THESE PATHS
# ============================================================
ADNI_ROOT = "/lustre/cursos/curso06/Alzheimer's disease/new_final/ADNI"
OUTPUT_DIR = "/lustre/cursos/curso06/alzheimer/nifti"
LOG_FILE   = "/lustre/cursos/curso06/alzheimer/conversion_log.csv"
# ============================================================

def find_subject_dicom_folders(adni_root: str) -> list[dict]:
    """
    Walks the ADNI folder structure and finds the deepest folder
    containing .dcm files for each subject+image_id combo.

    Expected structure:
    ADNI/
      {subject_id}/
        {sequence_name}/
          {date}/
            {image_id}/
              *.dcm
    """
    results = []
    adni_path = Path(adni_root)

    if not adni_path.exists():
        print(f"ERROR: ADNI root not found: {adni_root}")
        return results

    # Walk up to 5 levels deep looking for folders with .dcm files
    for dcm_file in adni_path.rglob("*.dcm"):
        folder = dcm_file.parent
        parts  = folder.relative_to(adni_path).parts

        if len(parts) >= 1:
            subject_id = parts[0]
            image_id   = parts[-1] if len(parts) >= 4 else folder.name
            results.append({
                "subject_id": subject_id,
                "image_id":   image_id,
                "dcm_folder": str(folder),
            })

    # Deduplicate — one entry per dcm_folder
    seen = set()
    unique = []
    for r in results:
        if r["dcm_folder"] not in seen:
            seen.add(r["dcm_folder"])
            unique.append(r)

    print(f"Found {len(unique)} unique subject/scan folders")
    return unique


def convert_single(dcm_folder: str, subject_id: str, image_id: str,
                   output_dir: str) -> dict:
    """
    Runs dcm2niix on a single folder.
    Returns a dict with conversion result info.
    """
    out_path = Path(output_dir) / subject_id
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",           # gzip output (.nii.gz)
        "-f", image_id,      # filename = image_id
        "-o", str(out_path), # output folder
        "-s", "y",           # single file output
        dcm_folder
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min per subject max
        )

        # Find the output file
        nifti_files = list(out_path.glob(f"{image_id}*.nii.gz"))

        if nifti_files:
            return {
                "subject_id": subject_id,
                "image_id":   image_id,
                "dcm_folder": dcm_folder,
                "nifti_path": str(nifti_files[0]),
                "status":     "ok",
                "error":      "",
            }
        else:
            return {
                "subject_id": subject_id,
                "image_id":   image_id,
                "dcm_folder": dcm_folder,
                "nifti_path": "",
                "status":     "failed_no_output",
                "error":      result.stderr[:200],
            }

    except subprocess.TimeoutExpired:
        return {
            "subject_id": subject_id,
            "image_id":   image_id,
            "dcm_folder": dcm_folder,
            "nifti_path": "",
            "status":     "timeout",
            "error":      "conversion took >5 min",
        }
    except Exception as e:
        return {
            "subject_id": subject_id,
            "image_id":   image_id,
            "dcm_folder": dcm_folder,
            "nifti_path": "",
            "status":     "error",
            "error":      str(e),
        }


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: DICOM → NIfTI Conversion")
    print("=" * 60)
    print(f"Input:  {ADNI_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Find all subject folders
    subjects = find_subject_dicom_folders(ADNI_ROOT)

    if not subjects:
        print("No subjects found. Check ADNI_ROOT path.")
        return

    # Convert each one
    results = []
    failed  = []

    for s in tqdm(subjects, desc="Converting"):
        r = convert_single(
            s["dcm_folder"],
            s["subject_id"],
            s["image_id"],
            OUTPUT_DIR
        )
        results.append(r)
        if r["status"] != "ok":
            failed.append(r)

    # Save log
    df = pd.DataFrame(results)
    df.to_csv(LOG_FILE, index=False)
    print(f"\nConversion log saved to: {LOG_FILE}")

    # Summary
    ok     = sum(1 for r in results if r["status"] == "ok")
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {ok}/{total} converted successfully")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for f in failed:
            print(f"  {f['subject_id']} | {f['status']} | {f['error']}")
    print("=" * 60)
    print(f"\nNext step: run 02_preprocess.py")


if __name__ == "__main__":
    main()
