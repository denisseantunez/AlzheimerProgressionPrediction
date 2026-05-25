#!/usr/bin/env python3
"""
STEP 3: Build master dataset CSV
- Merges the ADNI metadata (from your CSV) with preprocessed paths
- Deduplicates subjects with multiple scans (keeps best quality)
- Assigns numeric labels for classification
- Creates train/val/test splits stratified by group

Labels for 4-class classification:
    0 = CN
    1 = EMCI
    2 = LMCI
    3 = AD

Usage:
    python 03_build_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURE THESE PATHS
# ============================================================
ADNI_METADATA_CSV = "/lustre/cursos/curso06/Alzheimer's disease/new_final/new_final_3_19_2026.csv"
PREPROC_LOG       = "/lustre/cursos/curso06/alzheimer/preproc_log.csv"
OUTPUT_DIR        = "/lustre/cursos/curso06/alzheimer"

# NOTE: Create adni_metadata.csv from your original CSV
# It should have these columns (from the IDA export):
# Image Data ID, Subject, Group, Sex, Age, Visit, Description, Acq Date
# ============================================================

LABEL_MAP = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}

# Subjects with duplicate scans — keep only the FIRST image_id listed
# (update this dict if you want to manually pick the best one)
DUPLICATES_KEEP = {
    "012_S_4188": "I905421",
    "032_S_2119": "I862016",
    "032_S_4277": "I881980",
    "037_S_4071": "I833977",
    "041_S_4037": "I873878",
    "041_S_4974": "I915133",
    "053_S_2396": "I1154052",
    "116_S_0382": "I1037249",
    "116_S_4855": "I977140",
}


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load and clean ADNI metadata CSV."""
    df = pd.read_csv(csv_path)

    # Normalize column names (strip spaces, lowercase)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to standard names
    rename = {
        "image_data_id": "image_id",
        "subject":       "subject_id",
        "group":         "group",
        "sex":           "sex",
        "age":           "age",
        "acq_date":      "acq_date",
        "description":   "description",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Keep only relevant columns
    keep = ["image_id", "subject_id", "group", "sex", "age",
            "acq_date", "description"]
    df = df[[c for c in keep if c in df.columns]]

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For subjects with multiple scans, keep only one.
    Uses DUPLICATES_KEEP dict above, falls back to first occurrence.
    """
    cleaned = []

    for subject_id, group in df.groupby("subject_id"):
        if len(group) == 1:
            cleaned.append(group.iloc[0])
        else:
            # Check if we have a preferred image_id
            if subject_id in DUPLICATES_KEEP:
                preferred = DUPLICATES_KEEP[subject_id]
                match = group[group["image_id"] == preferred]
                if len(match):
                    cleaned.append(match.iloc[0])
                    continue
            # Fallback: keep first
            cleaned.append(group.iloc[0])

    result = pd.DataFrame(cleaned).reset_index(drop=True)
    print(f"  After deduplication: {len(result)} unique subjects "
          f"(removed {len(df) - len(result)} duplicates)")
    return result


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Stratified train/val/test split by group.
    70% train | 15% val | 15% test
    Split is by SUBJECT (not by scan) to prevent data leakage.
    """
    df = df.copy()

    # Initial split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=0.30,
        random_state=seed,
        stratify=df["group"]
    )

    temp_df = df.loc[temp_idx]

    # Split temp 50/50 → 15% val, 15% test
    val_idx, test_idx = train_test_split(
        temp_df.index,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["group"]
    )

    df["split"] = "train"
    df.loc[val_idx, "split"]  = "val"
    df.loc[test_idx, "split"] = "test"

    return df


def main():
    print("=" * 60)
    print("STEP 3: Build master dataset")
    print("=" * 60)

    # Load metadata
    print("\n[1/5] Loading ADNI metadata...")
    if not Path(ADNI_METADATA_CSV).exists():
        print(f"ERROR: {ADNI_METADATA_CSV} not found.")
        print("Please copy your IDA export CSV to that path.")
        print("It should be the same CSV you had with columns:")
        print("  Image Data ID, Subject, Group, Sex, Age, ...")
        return
    meta = load_metadata(ADNI_METADATA_CSV)
    print(f"  Loaded {len(meta)} rows")

    # Load preprocessed paths
    print("\n[2/5] Loading preprocessed paths...")
    if not Path(PREPROC_LOG).exists():
        print(f"ERROR: {PREPROC_LOG} not found. Run 02_preprocess.py first.")
        return
    preproc = pd.read_csv(PREPROC_LOG)
    preproc_ok = preproc[preproc["status"] == "ok"][
        ["subject_id", "processed_path"]
    ]
    print(f"  {len(preproc_ok)} successfully preprocessed volumes")

    # Deduplicate
    print("\n[3/5] Deduplicating subjects...")
    meta_dedup = deduplicate(meta)

    # Merge metadata + preprocessed paths
    print("\n[4/5] Merging metadata with image paths...")
    merged = meta_dedup.merge(preproc_ok, on="subject_id", how="inner")
    print(f"  {len(merged)} subjects with both metadata and processed image")

    # Check for subjects missing processed image
    missing = meta_dedup[~meta_dedup["subject_id"].isin(merged["subject_id"])]
    if len(missing):
        print(f"  WARNING: {len(missing)} subjects have metadata but no processed image:")
        for _, r in missing.iterrows():
            print(f"    {r['subject_id']} ({r['group']})")

    # Add labels
    merged["label"] = merged["group"].map(LABEL_MAP)

    # Group distribution check
    print("\n  Group distribution:")
    for g, n in merged["group"].value_counts().items():
        print(f"    {g}: {n} subjects")

    # Create splits
    print("\n[5/5] Creating train/val/test splits (70/15/15, stratified)...")
    final = make_splits(merged)

    print("\n  Split distribution:")
    for split in ["train", "val", "test"]:
        sub = final[final["split"] == split]
        counts = sub["group"].value_counts().to_dict()
        print(f"    {split}: {len(sub)} total | {counts}")

    # Save
    out_path = Path(OUTPUT_DIR) / "dataset.csv"
    final.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Dataset saved to: {out_path}")
    print(f"Total subjects: {len(final)}")
    print("=" * 60)
    print("\nNext step: run 04_train_baseline.py")


if __name__ == "__main__":
    main()
