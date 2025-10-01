# computerVision

Image recognition and object detection challenges

## Loading environment in Miniconda

```
conda env create -f environment/[name].yml
```

## Create environment in Miniconda

```
conda create -n computerVision python=3.10
conda activate computerVision
```

In MacOS

```
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

Verify installation

```
python -c "
    import tensorflow as tf; print('TensorFlow version:', tf.**version**);
    print('GPU devices:', tf.config.list_physical_devices('GPU'))
"
```

Install libraries

```
conda install rasterio scikit-learn matplotlib
```
