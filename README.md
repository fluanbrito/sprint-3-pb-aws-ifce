# Avaliação Sprint 3 - Programa de Bolsas Compass.uol / AWS e IFCE

Avaliação da terceira sprint do programa de bolsas Compass.uol para formação em machine learning para AWS.

---

## Equipe
- Dayanne Lucy
- Rafael Pereira
- Jhonatan Teixeira

## Dataset
Utilizamos o dataset [Rock Paper Scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) que contém imagens de gestos de mãos do jogo Pedra, Papel e Tesoura. A ideia é que seja possível identificar qual o tipo de gesto com o máximo de precisão através do upload de imagens.

## Banco de dados
Utilizamos o MongoDB para armazenamento do modelo.

## Código

```
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
Importação dos datasets de treino e de teste
```
!wget --no-check-certificate \
  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
  -O /tmp/rps.zip

!wget --no-check-certificate \
  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
  -O /tmp/rps-test-set.zip
```
Extração dos arquivos zipados
```
import zipfile
import os

local_zip = '/tmp/rps.zip'
zip_extract = zipfile.ZipFile(local_zip, 'r')
zip_extract.extractall('/tmp/')
zip_extract.close()

local_zip = '/tmp/rps-test-set.zip'
zip_extract = zipfile.ZipFile(local_zip, 'r')
zip_extract.extractall('/tmp/')
zip_extract.close()
```

Os dados são aumentados por meio de várias transformações para que o modelo sempre tenha uma imagem diferente, aumentando a eficiência do treinamento.
```
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    horizontal_flip = True,
    shear_range = 0.2,
    fill_mode = 'wrap',
    validation_split = 0.4
)
train_datagen
```
Os caminhos dos diretórios contendo os exemplos de treino e teste são especificados e os dados são preparados para o modelo.
- target_size: as imagens serão redimensionadas para 150x150px;
- class_mode: determina o tipo de label retornado.
```
TRAINING_DIR = '/tmp/rps/'

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (150, 150),
    class_mode = 'categorical',
)

VALIDATION_DIR = '/tmp/rps-test-set'

validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size = (150, 150),
    class_mode = 'categorical',
)
```

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])
```
Especificação da função de perda, otimizador e métricas de avaliação.
```
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.optimizers.Adam(),
    metrics = ['accuracy']               
)
```
Fazendo o fit do modelo para treinamento usando o conjunto de dados de treino que foi preparado no train_generator, a avaliação do modelo é feita usando o validation_generator.
```
history = model.fit(
    train_generator,
    epochs = 30,
    validation_data = validation_generator,
    verbose = 1,
)
```
Agora é feita a previsão do resultado das imagens que são enviadas através do upload.
```
import numpy as np
from google.colab import files
from tensorflow.keras.utils import load_img
#from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

uploaded = files.upload()

for fn in uploaded.keys():
  # predict images
  path = fn
  img_source = tf.keras.utils.load_img(path, target_size = (150, 150))
  imgplot = plt.imshow(img_source)
  x = tf.keras.utils.img_to_array(img_source)
  x = np.expand_dims(x, axis = 0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  print(fn)
  if classes[0, 0] == 1:
    print('Pedra')
  elif classes[0, 1] == 1:
    print('Papel')
  elif classes[0, 2] == 1:
    print('Tesoura')
```