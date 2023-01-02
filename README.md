# Avaliação Sprint 3 - Programa de Bolsas Compass UOL / AWS e IFCE

Avaliação da terceira sprint do programa de bolsas Compass UOL para formação em machine learning para AWS. A avaliação consiste no desenvolvimento de uma rede neural de classificação básica a partir da exploração de dados contidos em um dataset.

---

## Equipe

- Mylena Soares
- Nicolas Ferreira
- Samara Alcântara

## KMNIST

O dataset escolhido foi o KMNIST (Kuzushiji-MNIST) que é uma alternativa ao conjunto de dados MNIST, seus dados são conjuntos de caracteres que representam cada uma das linhas do alfabeto fonético japonês Hiragana.

## Recursos

- [Tutorial Rede Neural de classificação básica](https://www.tensorflow.org/tutorials/keras/classification)
- [Dataset Kmnist](https://www.tensorflow.org/datasets/catalog/kmnist)
- [Download mongoDB](https://www.mongodb.com/try/download/community)
- [Exemplos de Predições](https://www.deeplearningbook.com.br/reconhecimento-de-imagens-com-redes-neurais-convolucionais-em-python-parte-4/)


## Execução

Iniciamos o projeto criando o arquivo principal da aplicação utilizando o jupyter notebook, ao qual nomeamos **kmnist2.ipynb** e para servir como base do nosso desenvolvimento importamos todas as bibliotecas necessárias com o seguinte trecho de código:

```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
```

logo após, o dataset foi carregado e divido em quatros partes fundamentais para treinar a rede neural, separando as imagens de treino e teste e também as labels de treino e teste, conforme o código a seguir:

```python
(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
    name = 'kmnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

O método squeeze foi utilizado para remover entradas unidimensionais dos seguintes arrays
```python
x_train = x_train.squeeze()
x_test = x_test.squeeze()
```

Em seguida, utilizamos o código abaixo para relacionar e exibir as 10 primeiras imagens de treinamento com as suas classes correspondentes

```python
num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(20,20))

for i in range(0, num_classes):
  sample = x_train[y_train == i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Label: {}".format(i), fontsize=16)
```

Transformando os labels abaixo em uma matriz de binários
```
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

## Pré Processamento
Antes de treinar a rede, é necessário realizar o pré-processamento dos dados, o código abaixo exibe a primeira figura e através da biblioteca matplotlib podemos verificar a régua com os valores dos pixels. Para adequa-los à construção do modelo neural é necessário manter esse valores entre 0 e 1, assim dividimos por 250 - maior valor da régua.

```
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

Saída esperada:

![primeira figura](https://user-images.githubusercontent.com/103221427/210280996-b8e7a285-0582-4c14-b42d-6d051f053c7a.png)


Escalonando entre 0 e 1

```
x_train = x_train / 255.0
x_test = x_test / 255.0
```
Para efetuar o pré-processamento agora exibindo as primeiras 25 imagens:

```
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()
```








