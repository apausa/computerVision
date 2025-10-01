# computerVision

Image recognition and object detection challenges

## Run notebook in Cesvima

1. Install miniconda

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p conda
```

2. Create and activate environment

```
conda create -n computerVision python=3.10
export PATH=~/conda/envs/framework/bin:~/conda/bin:$PATH
conda activate computerVision
```

3. Install libraries

```
conda install jupyter nbconvert rasterio scikit-learn matplotlib
```

4. Copy datasets

```
mkdir -p imageRecognition/fnn
ln -s ../PROJECT/xview_recognition imageRecognition/
mkdir objectDetection
ln -s ../PROJECT/xview_detection objectDetection/
```

5. Copy notebook through scp

6. Create job.sh (check file) and run the script.

```
sbatch job.sh
squeue
```

## Run notebook locally

1. Install miniconda

2. Create and activate environment

```
conda create -n computerVision python=3.10
conda activate computerVision
```

3. Install libraries

```
conda install ipykernel rasterio scikit-learn matplotlib
```

4. MacOS only

```
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```
