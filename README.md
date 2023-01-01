# Avaliação da Sprint 3

> Neste projeto fazemos utilização do DataSet MNIST para reconhecimento de dígitos manuscritos.

[![N|Solid](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/LogoCompasso-positivo.png/440px-LogoCompasso-positivo.png)](https://compass.uol/pt/home/)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/Jeef-Moreira)

Neste arquivo se encontra detalhado o passo a passo de como foi desenvolvido, utilizando a linguagem Python, todo o procedimento de consumo do DataSet MNIST. Será apresentado:
- Apresentação do MNIST
- Ferramentas
- Comandos

## APRESENTAÇÃO DO MNIST

**MNIST**: 
O conjunto de dados MNIST é um acrônimo que significa o conjunto de dados Instituto Nacional Modificado de Padrões e Tecnologia.
É um conjunto de dados de 60.000 pequenas imagens quadradas de 28 × 28 pixels em escala de cinza de dígitos manuscritos entre 0 e 9.
A tarefa consiste em classificar uma dada imagem de um dígito manuscrito em uma das 10 classes que representam valores inteiros de 0 a 9, inclusive.

É um conjunto de dados amplamente usado e profundamente compreendido e, na maioria das vezes, é “resolvido”. Os modelos de melhor desempenho são redes neurais convolucionais de aprendizado profundo que atingem uma precisão de classificação acima de 99%, com uma taxa de erro entre 0,4% e 0,2% no conjunto de dados de teste de retenção.


## FERRAMENTAS

As principais ferramentas utilizadas no projeto foram:

- [DataSet MNIST](https://www.tensorflow.org/datasets/catalog/mnist) - Grande banco de dados de dígitos manuscritos que é amplamente utilizado para treinamento e testes na área de aprendizado de máquina.
- [Google Colab](https://colab.research.google.com/) - Serviço de nuvem gratuito hospedado pelo próprio Google para incentivar a pesquisa de Aprendizado de Máquina e Inteligência Artificial. Onde executamos nossos códigos em Python.
- [Visual Studio Code v.1.73.1](https://code.visualstudio.com/) - Editor de código aberto desenvolvido pela Microsoft. Nesse caso, ele foi usado em prol do desenvolvimento deste README do projeto.

## COMANDOS

Busque um DataSet de sua preferência, no nosso caso escolhemos o DataSet MNIST.

```python
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
```

```python
from keras.datasets import mnist
objects = mnist
(train_img, train_lab), (test_img, test_lab) = objects.load_data()
```

Observe que foi iniciado o download das imagens.

```python
plt.imshow(train_img[5])
print(train_lab[5])
```

Veja que com a utilização do comando print, foi gerada a imagem pertecente ao train.

```python
for i in range(20):
  plt.subplot(4, 5, i+1)
  plt.imshow(train_img[i], cmap='gray_r')
  plt.title("Dígito: {}".format(train_lab[i]))
  plt.subplots_adjust(hspace=0.5)
  plt.axis('off')
```

Caso você tenha exercido cada linha de comando correta, foi gerado uma tabela com diferentes imagens de dígitos manuscritos cada qual correspondente à um digito diferente de 0 - 9.

```python
print('Shape imagens de treino: ',train_img.shape)
print('Shape imagens de teste: ',test_img.shape)
```

Comando usado para gerar as informações de tamanho e formato de ambas as classes de imagens. Observe o resultado:

```python
train_img = train_img / 255.0
test_img = test_img / 255.0
```

Normalização das imagens.

```python
from keras.models import Sequential
from keras.layers import Flatten,Dense
model = Sequential()

input_layer = Flatten(input_shape=(28,28))
model.add(input_layer)
hidden_layer1 = Dense(512, activation='relu')
model.add(hidden_layer1)
hidden_layer2 = Dense(512, activation='relu')
model.add(hidden_layer2)
output_layer = Dense(10, activation='softmax')
model.add(output_layer)

model.summary()
```

Observe que foi gerado as informações do modelo sequencial, tais como: tipo, sua saída junto do seu formato e os parâmetros.

```python
# Compilando o modelo sequencial
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
model.fit(train_img, train_lab, epochs=10)
```

```python
# Acesso ao Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Salvar modelo no Google Drive
model.save('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5')
```

```python
# Use quando treinar o modelo
model.load_weights('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5')

# Compilando o modelo sequencial
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
loss_and_acc = model.evaluate(test_img, test_lab, verbose=2)
print("Teste de perda", loss_and_acc[0])
print("Teste de precisão", loss_and_acc[1])
```

Observe os resultados apurados:

```python
plt.imshow(test_img[0], cmap='gray_r')
plt.title('Valor atual: {}'.format(test_lab[0]))
prediction = model.predict(test_img)
plt.axis('off')
print('Valor predito: ', np.argmax(prediction[0]))

if(test_lab[0] == (np.argmax(prediction[0]))):
  print('Previsão bem-sucedida')
else:
  print('Previsão sem sucesso')
```

```python
from IPython.display import Image
Image(test_img[2], width=250, height=250)
#plt.imshow(test_img[2], cmap='gray_r')
```

```python
# Faça uma predição para uma nova imagem
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model

# Carregar e preparar a imagem
def load_image(filename):

  # Carregar a imagem
  img = load_img(filename, grayscale=True, target_size=(28,28))

  # Converter para array
  img = img_to_array(img)

  # Remodelar em uma única amostra com 1 canal
  img = img.reshape(1, 28, 28)

  # Preparar dados de pixel
  img = img.astype('float32')
  img = img / 255.0

  return img
```

```python
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
  x = uploaded[filename]

img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
```

Sinaliza se foi realizado o upload do arquivo.

```python
img = load_image(filename)
label = int(input('Número atual = '))
predict = model.predict(img)
classify = np.argmax(predict)
print('Valor predito: ', classify)

if(label == (np.argmax(predict))):
  print('Previsão bem-sucedida')
else:
  print('Previsão sem sucesso')

show=cv2.imread(filename)
plt.imshow(show)
```

## CONCLUSÃO

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dev]: <https://github.com/Jeef-Moreira>
   [dev]: <https://github.com/EdivalcoAraujo>
   [learning]: <https://compassuol.udemy.com>
   [boss]: <https://compass.uol/pt/home>
   [Google Colab]: <https://colab.research.google.com/>
   [Google Drive]: <https://drive.google.com/drive/my-drive>
   [Visual Studio Code]: <https://code.visualstudio.com>