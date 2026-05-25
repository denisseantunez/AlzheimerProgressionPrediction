#!/bin/bash
#SBATCH --job-name=alzheimer_pipeline
#SBATCH --output=/lustre/cursos/curso06/alzheimer/logs/pipeline_%j.out
#SBATCH --error=/lustre/cursos/curso06/alzheimer/logs/pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:MI210:1

echo "========================================"
echo "ALZHEIMER FULL PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────
module load conda
source activate alzheimer

# ── ROCm / MIOpen fix ────────────────────────────────────
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="/lustre/cursos/curso06/miopen_cache/$SLURM_JOB_ID"
export MIOPEN_USE_OPENCL=0
export HSA_ENABLE_SDMA=0
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1
export MIOPEN_COMPILE_PARALLEL_LEVEL=1
mkdir -p /lustre/cursos/curso06/miopen_cache/$SLURM_JOB_ID

cd /lustre/cursos/curso06/alzheimer_pipeline

echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ── Output folders ────────────────────────────────────────
mkdir -p /lustre/cursos/curso06/alzheimer/{nifti,processed,checkpoints,runs/baseline,logs,gradcam}
mkdir -p /lustre/cursos/curso06/alzheimer_binary/{logs,nifti,processed,ssl,finetune,gradcam}

# ════════════════════════════════════════════════════════════
# PARTE 1 — PIPELINE ORIGINAL (multiclase CN/EMCI/LMCI/AD)
# Pasos 1-3 ya corrieron exitosamente — COMENTADOS
# ════════════════════════════════════════════════════════════

# ── STEP 1: DICOM → NIfTI (YA CORRIÓ) ────────────────────
# python scripts/01_dicom_to_nifti.py
# if [ $? -ne 0 ]; then echo "STEP 1 FAILED"; exit 1; fi

# ── STEP 2: Preprocessing (YA CORRIÓ) ────────────────────
# python scripts/02_preprocess.py
# if [ $? -ne 0 ]; then echo "STEP 2 FAILED"; exit 1; fi

# ── STEP 3: Build dataset (YA CORRIÓ) ────────────────────
# python scripts/03_build_dataset.py
# if [ $? -ne 0 ]; then echo "STEP 3 FAILED"; exit 1; fi

# ── STEP 4: Train baseline multiclase ────────────────────
#echo "========================================"
#echo "STEP 4: Baseline CNN multiclase (CN/EMCI/LMCI/AD)"
#echo "Started: $(date)"
#echo "========================================"
#python scripts/04_train_baseline.py
#if [ $? -ne 0 ]; then echo "STEP 4 FAILED"; exit 1; fi
#echo "STEP 4 finished: $(date)"

# ── STEP 5: Grad-CAM multiclase ──────────────────────────
#echo ""
#echo "========================================"
#echo "STEP 5: Grad-CAM multiclase"
#echo "Started: $(date)"
#echo "========================================"
#python scripts/05_gradcam.py
#if [ $? -ne 0 ]; then echo "STEP 5 FAILED"; exit 1; fi

#echo "STEP 5 finished: $(date)"

# ════════════════════════════════════════════════════════════
# PARTE 2 — PIPELINE BINARIO (pMCI vs sMCI con SSL)
# ════════════════════════════════════════════════════════════

# ── STEP 3b: Build binary dataset CSV ────────────────────
#echo ""
#echo "========================================"
#echo "STEP 3b: Build binary dataset (pMCI vs sMCI)"
#echo "Started: $(date)"
#echo "========================================"
#python scripts/03b_build_binary_dataset.py
#if [ $? -ne 0 ]; then echo "STEP 3b FAILED"; exit 1; fi
#echo "STEP 3b finished: $(date)"

# ── STEP 3c: DICOM → NIfTI + preprocess binary ───────────
#echo ""
#echo "========================================"
#echo "STEP 3c: Convert + preprocess binary dataset"
#echo "Started: $(date)"
#echo "========================================"
#python scripts/03c_preprocess_binary.py
#if [ $? -ne 0 ]; then echo "STEP 3c FAILED"; exit 1; fi
#echo "STEP 3c finished: $(date)"

# ── STEP 4b: SimCLR SSL pretraining ──────────────────────
#echo ""
#echo "========================================"
#echo "STEP 4b: SimCLR SSL pretraining (sin etiquetas)"
#echo "Started: $(date)"
#echo "========================================"
#python scripts/04b_ssl_pretrain.py
#if [ $? -ne 0 ]; then echo "STEP 4b FAILED"; exit 1; fi
#echo "STEP 4b finished: $(date)"

# ── STEP 4c: Fine-tune binario (5-fold CV) ───────────────
echo ""
echo "========================================"
echo "STEP 4c: Fine-tune binario + baseline supervisado"
echo "Started: $(date)"
echo "========================================"
python scripts/04c_finetune_binary.py
if [ $? -ne 0 ]; then echo "STEP 4c FAILED"; exit 1; fi
echo "STEP 4c finished: $(date)"

# ── STEP 5b: Grad-CAM binario ────────────────────────────
echo ""
echo "========================================"
echo "STEP 5b: Grad-CAM binario (pMCI vs sMCI)"
echo "Started: $(date)"
echo "========================================"
python scripts/05b_gradcam_binary.py
if [ $? -ne 0 ]; then echo "STEP 5b FAILED"; exit 1; fi
echo "STEP 5b finished: $(date)"

# ════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "PIPELINE COMPLETO"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Outputs principales:"
echo "  [Multiclase]"
echo "    Modelo:      /lustre/cursos/curso06/alzheimer/checkpoints/baseline_best.pth"
echo "    Métricas:    /lustre/cursos/curso06/alzheimer/runs/baseline/results.json"
echo "    Grad-CAM:    /lustre/cursos/curso06/alzheimer/gradcam/"
echo ""
echo "  [Binario SSL]"
echo "    Encoder SSL: /lustre/cursos/curso06/alzheimer_binary/ssl/encoder_pretrained.pth"
echo "    Mejor modelo:/lustre/cursos/curso06/alzheimer_binary/finetune/best_model.pth"
echo "    CV resultados:/lustre/cursos/curso06/alzheimer_binary/finetune/ssl_cv_results.csv"
echo "    Comparación: /lustre/cursos/curso06/alzheimer_binary/finetune/comparison.json"
echo "    Grad-CAM:    /lustre/cursos/curso06/alzheimer_binary/gradcam/"