#!/usr/bin/env python3
"""
STEP 3b: Build binary progression dataset
Combines Progressors and Stables CSVs, picks one baseline MPRAGE scan
per subject, verifies labels with DXSUM, and outputs a clean CSV
ready for the preprocessing pipeline.

Labels:
    0 = Stable   (sMCI — stayed MCI throughout follow-up)
    1 = Progressor (pMCI — converted to Dementia at some visit)

Output: /lustre/cursos/curso06/alzheimer_binary/binary_dataset_raw.csv
        (one row per subject, with image_id and dcm folder path)

Usage:
    python 03b_build_binary_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS — adjust if needed
# ============================================================
PROGRESSORS_CSV  = "/lustre/cursos/curso06/Progressors/Progressors_5_04_2026.csv"
STABLES_CSV      = "/lustre/cursos/curso06/Stables/Stables_5_04_2026.csv"
DXSUM_CSV        = "/lustre/cursos/curso06/DXSUM.csv"

PROG_ADNI_ROOT   = "/lustre/cursos/curso06/Progressors/ADNI"
STAB_ADNI_ROOT   = "/lustre/cursos/curso06/Stables/ADNI"

OUTPUT_DIR       = "/lustre/cursos/curso06/alzheimer_binary"
OUTPUT_CSV       = f"{OUTPUT_DIR}/binary_dataset_raw.csv"
# ============================================================

# Sequences considered valid MPRAGE (in priority order)
MPRAGE_SEQS = [
    'MPRAGE',
    'MP-RAGE',
    'Accelerated Sagittal MPRAGE',
    'Sagittal 3D Accelerated MPRAGE',
    'MPRAGE GRAPPA2',
    'MPRAGE Repeat',
    'MP-RAGE REPEAT',
]
SEQ_PRIORITY = {s: i for i, s in enumerate(MPRAGE_SEQS)}

# Visits that count as baseline
BASELINE_VISITS = ['sc', 'bl', 'scmri', 'init', '4_init']


def pick_best_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each subject, select one baseline MPRAGE scan.
    Priority: MPRAGE > MP-RAGE > Accelerated > ... > Repeat variants
    If multiple rows share the same best sequence, keep the one
    with the earliest Acq Date.
    """
    mprage_bl = df[
        df['Description'].isin(MPRAGE_SEQS) &
        df['Visit'].isin(BASELINE_VISITS)
    ].copy()

    if mprage_bl.empty:
        return pd.DataFrame()

    mprage_bl['seq_priority'] = mprage_bl['Description'].map(SEQ_PRIORITY).fillna(99)
    mprage_bl['Acq Date'] = pd.to_datetime(mprage_bl['Acq Date'], errors='coerce')

    best = (
        mprage_bl
        .sort_values(['seq_priority', 'Acq Date'])
        .groupby('Subject')
        .first()
        .reset_index()
    )
    return best


def verify_labels_with_dxsum(df: pd.DataFrame, dxsum: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-check each subject's label against DXSUM longitudinal history.
    Progressor  → should have 'Dementia' in at least one follow-up visit
    Stable      → should never have 'Dementia'

    Adds columns:
        dxsum_confirmed  : True/False
        dxsum_note       : human-readable explanation
    """
    results = []

    for _, row in df.iterrows():
        subject   = row['Subject']
        label     = row['label']  # 0=stable, 1=progressor

        history = dxsum[dxsum['PTID'] == subject][['VISCODE', 'EXAMDATE', 'DIAGNOSIS']]
        history = history.sort_values('EXAMDATE')

        has_dementia = (history['DIAGNOSIS'] == 'Dementia').any()
        ever_mci     = (history['DIAGNOSIS'] == 'MCI').any()

        if label == 1:  # expected progressor
            confirmed = has_dementia
            note = "DXSUM confirms Dementia" if confirmed else "WARNING: no Dementia found in DXSUM"
        else:           # expected stable
            confirmed = not has_dementia
            note = "DXSUM confirms stable MCI" if confirmed else "WARNING: Dementia found — may be mislabeled"

        results.append({
            'Subject':          subject,
            'dxsum_confirmed':  confirmed,
            'dxsum_note':       note,
            'dxsum_last_dx':    history['DIAGNOSIS'].iloc[-1] if len(history) else 'NOT FOUND',
            'dxsum_n_visits':   len(history),
        })

    return pd.DataFrame(results)


def find_dcm_folder(adni_root: str, subject_id: str, image_id: str) -> str:
    """
    Navigate the ADNI folder structure to find the DCM folder for this image.

    Expected structure:
    ADNI/{subject_id}/{sequence}/{date}/{image_id}/
    """
    subj_path = Path(adni_root) / subject_id

    if not subj_path.exists():
        return ""

    # Search recursively for a folder named exactly image_id
    matches = list(subj_path.rglob(image_id))
    if matches:
        # Verify it actually contains DCM files
        for m in matches:
            if m.is_dir() and list(m.glob("*.dcm")):
                return str(m)
            # Sometimes DCMs are one level deeper
            for sub in m.iterdir():
                if sub.is_dir() and list(sub.glob("*.dcm")):
                    return str(sub)

    # Fallback: look for any folder with DCMs under subject path
    for dcm in subj_path.rglob("*.dcm"):
        parent_name = dcm.parent.name
        # Check if the grandparent or parent name matches image_id
        if dcm.parent.name == image_id or dcm.parent.parent.name == image_id:
            return str(dcm.parent)

    return ""


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 3b: Build binary progression dataset")
    print("=" * 60)

    # ── Load CSVs ──────────────────────────────────────────
    print("\n[1/5] Loading CSVs...")
    prog = pd.read_csv(PROGRESSORS_CSV)
    stab = pd.read_csv(STABLES_CSV)
    dx   = pd.read_csv(DXSUM_CSV)

    print(f"  Progressors: {prog['Subject'].nunique()} unique subjects, {len(prog)} rows")
    print(f"  Stables:     {stab['Subject'].nunique()} unique subjects, {len(stab)} rows")
    print(f"  DXSUM:       {dx['PTID'].nunique()} unique subjects, {len(dx)} rows")

    # ── Pick best baseline MPRAGE ──────────────────────────
    print("\n[2/5] Selecting one baseline MPRAGE scan per subject...")
    prog_best = pick_best_baseline(prog)
    stab_best = pick_best_baseline(stab)

    prog_best['label'] = 1  # progressor
    stab_best['label'] = 0  # stable

    prog_missing = set(prog['Subject'].unique()) - set(prog_best['Subject'].unique())
    stab_missing = set(stab['Subject'].unique()) - set(stab_best['Subject'].unique())

    print(f"  Progressors with baseline MPRAGE: {len(prog_best)}/47")
    print(f"  Stables with baseline MPRAGE:     {len(stab_best)}/48")

    if prog_missing:
        print(f"  Progressors WITHOUT baseline MPRAGE ({len(prog_missing)}): "
              f"{sorted(prog_missing)}")
        print("    → These subjects will be excluded (no usable baseline scan)")
    if stab_missing:
        print(f"  Stables WITHOUT baseline MPRAGE ({len(stab_missing)}): "
              f"{sorted(stab_missing)}")
        print("    → These subjects will be excluded")

    # ── Combine ────────────────────────────────────────────
    print("\n[3/5] Combining and verifying labels with DXSUM...")
    combined = pd.concat([prog_best, stab_best], ignore_index=True)

    # Verify against DXSUM
    verification = verify_labels_with_dxsum(combined, dx)
    combined = combined.merge(verification, on='Subject', how='left')

    n_confirmed   = combined['dxsum_confirmed'].sum()
    n_unconfirmed = (~combined['dxsum_confirmed']).sum()
    print(f"  DXSUM confirmed:   {n_confirmed}/{len(combined)}")
    print(f"  DXSUM mismatches:  {n_unconfirmed}")

    if n_unconfirmed > 0:
        print("\n  ⚠ Subjects with label/DXSUM mismatch:")
        mismatches = combined[~combined['dxsum_confirmed']]
        for _, r in mismatches.iterrows():
            print(f"    {r['Subject']} | label={'Progressor' if r['label']==1 else 'Stable'} "
                  f"| {r['dxsum_note']} | last_dx={r['dxsum_last_dx']}")

    # ── Find DCM folders ───────────────────────────────────
    print("\n[4/5] Locating DCM folders on disk...")
    dcm_folders = []
    not_found   = []

    for _, row in combined.iterrows():
        adni_root = PROG_ADNI_ROOT if row['label'] == 1 else STAB_ADNI_ROOT
        dcm_path  = find_dcm_folder(adni_root, row['Subject'], row['Image Data ID'])

        dcm_folders.append(dcm_path)
        if not dcm_path:
            not_found.append(row['Subject'])

    combined['dcm_folder'] = dcm_folders

    found     = combined['dcm_folder'].ne("").sum()
    not_found_n = combined['dcm_folder'].eq("").sum()
    print(f"  DCM folders found:     {found}/{len(combined)}")
    print(f"  DCM folders NOT found: {not_found_n}")

    if not_found:
        print(f"\n  ⚠ Subjects with no DCM folder found:")
        for s in sorted(set(not_found)):
            row = combined[combined['Subject'] == s].iloc[0]
            print(f"    {s} | Image ID: {row['Image Data ID']} | "
                  f"label={'Progressor' if row['label']==1 else 'Stable'}")

    # ── Save ───────────────────────────────────────────────
    print("\n[5/5] Saving dataset...")

    # Final dataset: only subjects where we found the DCM folder
    final = combined[combined['dcm_folder'].ne("")].copy()

    # Rename columns for consistency with existing pipeline
    final = final.rename(columns={
        'Subject':       'subject_id',
        'Image Data ID': 'image_id',
        'Sex':           'sex',
        'Age':           'age',
        'Visit':         'visit',
        'Description':   'description',
        'Acq Date':      'acq_date',
    })

    # Select and order columns
    cols = ['subject_id', 'image_id', 'label', 'sex', 'age',
            'visit', 'description', 'acq_date', 'dcm_folder',
            'dxsum_confirmed', 'dxsum_note', 'dxsum_last_dx', 'dxsum_n_visits']
    final = final[[c for c in cols if c in final.columns]]

    final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Dataset saved: {OUTPUT_CSV}")
    print(f"Total usable subjects: {len(final)}")
    print(f"  Progressors (label=1): {(final['label']==1).sum()}")
    print(f"  Stables     (label=0): {(final['label']==0).sum()}")
    print()

    # Demographics summary
    print("Demographics:")
    print(f"  Age:  mean={final['age'].mean():.1f}  std={final['age'].std():.1f}  "
          f"range=[{final['age'].min():.0f}, {final['age'].max():.0f}]")
    print(f"  Sex:  {final['sex'].value_counts().to_dict()}")
    print(f"  DXSUM confirmed: {final['dxsum_confirmed'].sum()}/{len(final)}")
    print("=" * 60)
    print("\nNext step: run 03c_preprocess_binary.py")


if __name__ == "__main__":
    main()
