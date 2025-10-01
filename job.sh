#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=computerVision
#SBATCH --partition=standard-gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
##------------------------ End job description ------------------------

module --force purge && module load apps/2021 CUDA/11.8.0 TensorFlow/2.13.0-foss-2023a

srun jupyter nbconvert --execute --to notebook --inplace imageRecognition/fnn/ffNN_example.ipynb
