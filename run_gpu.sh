#!/bin/bash
#SBATCH --job-name=actor_gpu
#SBATCH --output=logs/actor_gpu_%j.out
#SBATCH --error=logs/actor_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd /gpfs/bwfor/home/tu/tu_tu/tu_zxope51/ACTORrep

module load devel/python/3.13.1
source venv/bin/activate

mkdir -p logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

python -u scripts/run_all_active_learning.py
