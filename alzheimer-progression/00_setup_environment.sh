#!/bin/bash
# ============================================================
# STEP 0: Setup environment in YUCA
# Run this ONCE. After this, just: source activate alzheimer
# ============================================================

echo ">>> Loading conda module..."
module load conda

echo ">>> Creating alzheimer environment (Python 3.10)..."
conda create -n alzheimer python=3.10 -y

echo ">>> Activating environment..."
source activate alzheimer

echo ">>> Installing PyTorch (CPU first — GPU version after checking CUDA)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ">>> Installing MONAI (medical imaging framework)..."
pip install monai[all]

echo ">>> Installing medical imaging utilities..."
pip install nibabel pydicom

echo ">>> Installing ML utilities..."
pip install scikit-learn pandas numpy matplotlib seaborn tqdm

echo ">>> Installing dcm2niix via conda..."
conda install -c conda-forge dcm2niix -y

echo ">>> Installing tensorboard for training monitoring..."
pip install tensorboard

echo ">>> Verifying installation..."
python -c "
import torch, monai, nibabel, pydicom, sklearn, pandas, numpy
print('torch:', torch.__version__)
print('monai:', monai.__version__)
print('nibabel:', nibabel.__version__)
print('Everything OK!')
"

echo ""
echo "============================================"
echo "DONE. To activate later:"
echo "  module load conda && source activate alzheimer"
echo "============================================"
