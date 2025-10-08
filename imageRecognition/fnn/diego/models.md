# 🧠 Model 6


```python
model = models.Sequential([
    # Input layer
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(1024),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.5),
    
    # Second hidden layer
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.4),
    
    # Third hidden layer
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.3),
    
    # Output layer
    layers.Dense(len(categories), activation='softmax')
])
```
## Results

| Metric             |  Value  |
| :----------------- | :-----: |
| **Mean Accuracy**  | 54.507% |
| **Mean Recall**    | 46.860% |
| **Mean Precision** | 50.245% |

------------------------------------------------------------------------------------------------------------------------------------------

# 🧠 Model 7

Cambio el momentum de 0.9 a 0.5 y aumento a 40 epochs con este modelo.

```python
model = models.Sequential([
    # Input layer
    keras.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    # Second hidden layer
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    # Third hidden layer
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    # Output layer
    layers.Dense(len(categories)),
    layers.Activation('softmax')
])
```

| Metric             |  Value  |
| :----------------- | :-----: |
| **Mean Accuracy**  | 59.253% |
| **Mean Recall**    | 53.466% |
| **Mean Precision** | 54.301% |

# 🧠 Model 8

Cambio toda la red a 512 neuronas, meto otra capa oculta, bajo el dropout en 0.1 y todo eso con la idea de que me va a reconocer los helipuertos

```python
model = models.Sequential([
    # Input layer
    keras.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Second hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    # Fourth hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(len(categories)),
    layers.Activation('softmax')
])
```

__**Reconoce el helipuerto**__

| Metric             |  Value  |
| :----------------- | :-----: |
| **Mean Accuracy**  | 59.893% |
| **Mean Recall**    | 57.888% |
| **Mean Precision** | 57.527% |

# 🧠 Model 9

Idea es mejorar valores de accuracy, recall y precision. Para eso he quitado una hidden layer, he puesto la primera layer como 1024, segunda como 512 y tercera como 256
Sigo con 40 epochs. He bajado el learning rate a 0.001 y momentum sigue en 0.5

```python
model = models.Sequential([
    # Input layer
    keras.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(1024),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Second hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(len(categories)),
    layers.Activation('softmax')
])
```

| Metric             |  Value  |
| :----------------- | :-----: |
| **Mean Accuracy**  | 62.773% |
| **Mean Recall**    | 59.170% |
| **Mean Precision** | 62.693% |

He lanzado otra vez el mismo test para ver si me da los mismos datos!!!

# 🧠 Model 9 (Segunda ejecución)
Para ver si era el mismo modelo y ver como se comporta una segunda vez se lanza con lo mismo:
```python
Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0)
```
* 40 epochs
* batch size de 16

```python
model = models.Sequential([
    # Input layer
    keras.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(1024),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Second hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(len(categories)),
    layers.Activation('softmax')
])
```

| Metric             |  Value  |
| :----------------- | :-----: |
| **Mean Accuracy**  | 62.933% |
| **Mean Recall**    | 61.359% |
| **Mean Precision** | 63.103% |

# 🧠 Model 10


```python
model = models.Sequential([
    # Input layer
    keras.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Second hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(len(categories)),
    layers.Activation('softmax')
])
```

Empeoran las metricas

# 🧠 Model 11

He disminuido el numero de neuronas /2 para ver si haciendolo más sencillo sigue siendo igual de útil.
Es casi igual pero aún así empeora.

```python
model = models.Sequential([
    # Input layer
    keras.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Flatten(),
    
    # First hidden layer
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Second hidden layer
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(len(categories)),
    layers.Activation('softmax')
])
```

| Metric             |  Value  |
| :----------------- | :-----: |
| **Mean Accuracy**  | 62.293% |
| **Mean Recall**    | 59.310% |
| **Mean Precision** | 60.471% |