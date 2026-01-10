#!/bin/bash
#SBATCH --job-name=computerVision
#SBATCH --output=computerVision.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pablo.apausa@alumnos.upm.es

# === DISABLE VERBOSE TF LOGGING ===
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYDEVD_DISABLE_FILE_VALIDATION=1

# === CARGA DE MÓDULOS ===
module load Jupyter-bundle/20230823-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load rasterio/1.3.9-foss-2023a
module load scikit-learn/1.4.2-gfbf-2023a
module load PyTorch/2.1.2-foss-2023a

srun jupyter nbconvert --execute --to notebook --inplace /home/x467/x467028/yolov11.ipynb
