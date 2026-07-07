#!/bin/bash
#SBATCH --job-name=v4_deepsets
#SBATCH --output=logs/v4_deepsets_%j.out
#SBATCH --error=logs/v4_deepsets_%j.err
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.vallejos@ing.puc.cl

module load cuda/12.9

source $HOME/.bashrc
conda activate ipre

cd ~/IPRE/PMDKernel
mkdir -p logs

echo "=== ENV ==="
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')"
python -c "import pytorch_lightning as pl; print(f'lightning={pl.__version__}')"
python -c "import comet_ml; print(f'comet_ml={comet_ml.__version__}')"
echo

echo "=== GPU ==="
nvidia-smi

echo "=== TRAIN ==="
python python/train_v4.py \
    --h5 data/datasets/v2_xyz100_step50_n5000.h5 \
    --run-tag v4_deepsets_v2_n5000 \
    --epochs 100 \
    --patience 20 \
    --comet-project pmdkernel \
    --num-workers 4 \
    --pin-memory
