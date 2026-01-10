# computerVision

Image recognition and object detection challenges

## Run notebook in Cesvima

Local computer

```
scp /Users/pablo/Desktop/repos/computerVision/objectDetection/cesvima/yolov11.ipynb x467028@magerit.cesvima.upm.es:~/
ssh x467028@magerit.cesvima.upm.es
```

Magerit

```
sbatch job.sh
squeue
```

## Run notebook locally

1. Install miniconda

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p conda
```

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
