# Alzheimer Progression Prediction via Self-Supervised Learning on 3D MRI

Early prediction of MCI-to-Alzheimer's conversion using SimCLR pretraining on structural brain MRI, with comparison against a supervised baseline.

**Universidad de Sonora — BSc in Computer Science**  
May 2026

**Authors:** Denisse Antunez López · Ana Laura Chenoweth Galaz · Georgina Salcido Valenzuela · Omar Pacheco Velásquez
<img width="1202" height="397" alt="image" src="https://github.com/user-attachments/assets/16c39348-cd2e-4f56-bdad-c35a97870610" />

---

## Overview

Patients with Mild Cognitive Impairment (MCI) don't all progress to Alzheimer's Disease — some stay stable, some even revert. Predicting *which* ones will convert is a critical clinical challenge. This project tackles it using 3D structural MRI from the [ADNI](https://adni.loni.usc.edu/) dataset and three deep learning models:

| Model | Task | Approach |
|---|---|---|
| Model 1 — 4-class classifier | Classify CN / EMCI / LMCI / AD | Supervised ResNet-18 3D |
| Model 2 — SSL + fine-tuning | Predict MCI → AD progression | SimCLR pretraining → binary fine-tuning |
| Model 3 — Supervised baseline | Predict MCI → AD progression | ResNet-18 3D trained from scratch |

Models 2 and 3 are identical in architecture — the only difference is weight initialization. This isolates the benefit of SSL pretraining.

---

## Results

| Metric | SSL (Model 2) | Supervised (Model 3) |
|---|---|---|
| AUC-ROC | **0.612** | 0.550 |
| Balanced Accuracy | **0.773** | 0.600 |
| Sensitivity | 0.709 | **0.855** |
| Specificity | 0.405 | 0.500 |

SSL outperforms the supervised baseline on AUC and balanced accuracy. The higher sensitivity of the supervised model reflects a bias toward predicting "progressor" rather than genuine discriminative ability.

---

## Repository Structure

```
alzheimer-progression/
│
├── scripts/
│   ├── 01_dicom_to_nifti.py         # DICOM → NIfTI conversion via dcm2niix
│   ├── 02_preprocess.py             # Z-score normalization + resize to 96³
│   ├── 03_build_dataset.py          # Build master CSV (4-class, original data)
│   ├── 03b_build_binary_dataset.py  # Build binary CSV (progressors vs stables)
│   ├── 03c_preprocess_binary.py     # Preprocess binary dataset
│   ├── 04_train_baseline.py         # Train Model 1 (4-class supervised)
│   ├── 04b_ssl_pretrain.py          # SimCLR SSL pretraining (Model 2, Phase 1)
│   ├── 04c_finetune_binary.py       # Binary fine-tuning + supervised baseline (Models 2 & 3)
│   ├── 05_gradcam.py                # Grad-CAM for Model 1
│   └── 05b_gradcam_binary.py        # Grad-CAM for Models 2 & 3
│
├── slurm_full_pipeline.sh           # Full SLURM job 
│
│
├── Alzheimer_Progression.pdf        # Project report
└── README.md
```

---

## Data

Data comes from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). Access requires a registered account through the LONI IDA platform.

**Data is not included in this repository.** To reproduce the experiments, download the following from ADNI:

- Multi-phase MRI scans (ADNI 1, GO, 2, 3) for CN, EMCI, LMCI, and AD groups — used for SSL pretraining and the 4-class classifier
- Baseline MPRAGE scans for EMCI subjects with longitudinal follow-up — used to build the binary (progressor vs stable) dataset
- `DXSUM` clinical data table — used to verify progression labels

Expected folder structure on disk:

```
/lustre/cursos/curso06/
├── Alzheimer's disease/new_final/ADNI/   # Original 192-scan dataset
├── Progressors/
│   ├── ADNI/                             # Progressor MRI folders
│   └── Progressors_5_04_2026.csv
└── Stables/
    ├── ADNI/                             # Stable MRI folders
    └── Stables_5_04_2026.csv
```

Paths are configurable at the top of each script.

---

## Pipeline

### Step 1 — DICOM to NIfTI
```bash
python scripts/01_dicom_to_nifti.py
```
Recursively finds `.dcm` folders and converts each scan to a single `.nii.gz` volume using `dcm2niix`. Produces a `conversion_log.csv`.

### Step 2 — Preprocessing
```bash
python scripts/02_preprocess.py
```
For each NIfTI volume:
1. Extract first volume if 4D
2. Discard volumes with any dimension < 32 voxels
3. Z-score normalize intensities
4. Clip to [-3, 3] and rescale to [0, 1]
5. Resize to 96×96×96 via linear spline interpolation

### Step 3 — Build Dataset CSVs
```bash
python scripts/03_build_dataset.py        # 4-class master CSV
python scripts/03b_build_binary_dataset.py  # Binary progressor/stable CSV
python scripts/03c_preprocess_binary.py     # Preprocess binary subjects
```

### Step 4 — Train Models
```bash
# Model 1: 4-class supervised classifier
python scripts/04_train_baseline.py

# Model 2: SSL pretraining (SimCLR, 274 volumes unlabeled)
python scripts/04b_ssl_pretrain.py

# Models 2 & 3: Binary fine-tuning + supervised baseline (5-fold CV)
python scripts/04c_finetune_binary.py
```

### Step 5 — Grad-CAM
```bash
python scripts/05_gradcam.py         # Model 1
python scripts/05b_gradcam_binary.py # Models 2 & 3
```

### Running on SLURM (YUCA)
```bash
# Full binary pipeline (assumes Steps 1-3 already ran)
sbatch slurm_full_pipeline.sh


# Monitor
squeue -u <your_username>
tail -f /lustre/cursos/curso06/alzheimer/logs/pipeline_<JOBID>.out
```

---

## Environment Setup

Tested on YUCA (AMD EPYC 9224, AMD MI210 64GB VRAM, ROCm).

```bash
module load conda
conda create -n alzheimer python=3.10
source activate alzheimer

pip install torch torchvision monai nibabel dcm2niix \
            scikit-learn scipy pandas tqdm matplotlib
```

**ROCm / MIOpen fix** (required on YUCA — add to SLURM scripts):
```bash
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/lustre/cursos/curso06/miopen_cache/$SLURM_JOB_ID"
export MIOPEN_USE_OPENCL=0
export HSA_ENABLE_SDMA=0
export PYTHONUNBUFFERED=1
```

---

## Key Hyperparameters

**SSL Pretraining (SimCLR)**
- Backbone: ResNet-18 3D (MONAI)
- Projection head: 512 → 256 → 128
- Loss: NT-Xent (temperature-scaled InfoNCE)
- Optimizer: AdamW, lr=3×10⁻⁴, weight decay=1×10⁻⁴
- Epochs: 100, batch size: 2 (effective 16 via gradient accumulation ×4)
- Augmentations: random flips, crop+resize (90%), Gaussian noise, intensity jitter, Gaussian blur

**Fine-tuning / Supervised Baseline**
- Classification head: 512 → Dropout(0.5) → 64 → ReLU → Dropout(0.3) → 2
- Loss: Weighted Cross-Entropy
- Evaluation: 5-fold stratified CV
- Phase A (10 epochs): frozen encoder, lr=1×10⁻³
- Phase B (40 epochs): full model, lr=1×10⁻⁴
- Optimizer: AdamW, scheduler: CosineAnnealingLR → 1×10⁻⁶

---

## Compute

All experiments ran on the **YUCA supercomputer** at Universidad de Sonora (ACARUS group):
- CPU: AMD EPYC 9224 (24 cores, 2.5GHz)
- RAM: 1TB
- GPU: AMD MI210 (64GB VRAM)

Approximate runtimes: SSL pretraining ~2h · Fine-tuning (5 folds) ~3h · Grad-CAM ~15min

---

## References

Key papers:

- Chen et al. (2020) — SimCLR: *A Simple Framework for Contrastive Learning of Visual Representations*
- Kaczmarek et al. (2025) — SimCLR adapted for 3D brain MRI across neurological diseases
- Elmannai et al. (2025) — SSL with attention for Alzheimer's detection
- Lim et al. (2022) — CNN-based MCI-to-AD progression prediction on ADNI

---

## License

Data from ADNI is subject to [ADNI's data use agreement](https://adni.loni.usc.edu/data-samples/data-types/). Code in this repository is for academic use.
