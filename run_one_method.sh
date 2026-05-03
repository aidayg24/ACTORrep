#!/bin/bash
#SBATCH --job-name=actor_method
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

METHOD=$1

cd /gpfs/bwfor/home/tu/tu_tu/tu_zxope51/ACTORrep

export PYTHONPATH=/gpfs/bwfor/home/tu/tu_tu/tu_zxope51/ACTORrep:$PYTHONPATH

module load devel/python/3.13.1
source venv/bin/activate

echo "Running method: $METHOD"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

python -u "$METHOD"
