#!/bin/bash
#SBATCH --job-name=computerVision
#SBATCH --output=output.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pablo.apausa@alumnos.upm.es

SECONDS=0

echo "1 Loading modules"
module --force purge
module load apps/2021
module load Anaconda3/2025.06-1

echo "2 Initializing conda"
source /media/apps/avx512-2021/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh

# Workaround for Intel library conflicts
export MKL_THREADING_LAYER=GNU

echo "3 Activating and testing environment"
conda run -n computerVision python - << 'EOF'
import torch
print("> PyTorch version:", torch.__version__)
if torch.cuda.is_available(): print("> GPU:", torch.cuda.get_device_name(0))
else: print("> No GPU")
EOF

echo "4. Running notebook"
srun conda run -n computerVision python -m jupyter nbconvert --execute --to notebook --inplace /home/x467/x467028/yolov11.ipynb

status=$?

if [ $status -ne 0 ]; then
    echo "[Error] Notebook failed with code $status"
  else
    echo "[Success] Notebook completed"
    printf "Detector runtime: %02d:%02d:%02d (HH:MM:SS)\n" \
      $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60))
fi