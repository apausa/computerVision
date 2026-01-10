# computerVision

Image recognition and object detection challenges

## Run notebook in Cesvima

### Local computer

Login

```
ssh x467028@magerit.cesvima.upm.es
```

Send content

```
scp /Users/pablo/Desktop/repos/computerVision/objectDetection/cesvima/{job.sh,yolov11.ipynb} x467028@magerit.cesvima.upm.es:~/
```

Receive content

```
rsync -avz --progress x467028@magerit.cesvima.upm.es:/home/x467/x467028/{yolov11_test,runs,prediction.json,yolov11.ipynb,output.log} /Users/pablo/desktop/cesvima/
```

### Remote computer

1. Load Anaconda

```
module --force purge
module load apps/2021
module load Anaconda3/2025.06-1
```

2. Initialize Anaconda

```
source /media/apps/avx512-2021/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
```

3. Create and activate environment

```
conda create -n computerVision python=3.10
conda activate computerVision
```

5. Install dependencies (this might take forever)

```
conda install numpy scikit-learn jupyter

conda clean --all -y

pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir ultralytics
```

4. Run job

```
sbatch job.sh
squeue
```

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --override-channels
