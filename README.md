# Avaliação da Sprint 3

> Neste projeto fazemos utilização do DataSet MNIST para reconhecimento de dígitos manuscritos.

[![N|Solid](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/LogoCompasso-positivo.png/440px-LogoCompasso-positivo.png)](https://compass.uol/pt/home/)

Neste arquivo se encontra detalhado o passo a passo de como foi desenvolvido, utilizando a linguagem Python, todo o procedimento de consumo do DataSet MNIST. Será apresentado:
- Apresentação do MNIST
- Ferramentas
- Comandos

## Apresentação do MNIST

**MNIST**: 
O conjunto de dados MNIST é um acrônimo que significa o conjunto de dados Instituto Nacional Modificado de Padrões e Tecnologia.
É um conjunto de dados de 60.000 pequenas imagens quadradas de 28 × 28 pixels em escala de cinza de dígitos manuscritos entre 0 e 9.
A tarefa consiste em classificar uma dada imagem de um dígito manuscrito em uma das 10 classes que representam valores inteiros de 0 a 9, inclusive.

É um conjunto de dados amplamente usado e profundamente compreendido e, na maioria das vezes, é “resolvido”. Os modelos de melhor desempenho são redes neurais convolucionais de aprendizado profundo que atingem uma precisão de classificação acima de 99%, com uma taxa de erro entre 0,4% e 0,2% no conjunto de dados de teste de retenção.


## Ferramentas

As principais ferramentas utilizadas no projeto foram:

- [DataSet MNIST](https://www.tensorflow.org/datasets/catalog/mnist) - Grande banco de dados de dígitos manuscritos que é amplamente utilizado para treinamento e testes na área de aprendizado de máquina.
- [Google Colab](https://colab.research.google.com/) - Serviço de nuvem gratuito hospedado pelo próprio Google para incentivar a pesquisa de Aprendizado de Máquina e Inteligência Artificial. Onde executamos nossos códigos em Python.
- [Visual Studio Code v.1.73.1](https://code.visualstudio.com/) - Editor de código aberto desenvolvido pela Microsoft. Nesse caso, ele foi usado em prol do desenvolvimento deste README do projeto.
- [Firebase]() - Firebase é um conjunto de serviços de hospedagem para qualquer tipo de aplicativo (Android, iOS, Javascript, Node.js, Java, Unity, PHP, C++...). Oferece NoSQL e hospedagem em tempo real de bancos de dados, conteúdo, autenticação social (Google, Facebook, Twitter e Github) e notificações, ou serviços, como um servidor de comunicação em tempo real.

## Comandos

Busque um DataSet de sua preferência, no nosso caso escolhemos o DataSet MNIST.

```python
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pyrebase
import urllib.request
```

Chamamos as bibliotecas que serão utilizadas na aplicação.

```python
firebaseConfig = {
  'apiKey': "AIzaSyB__HOXZ8U-0kVjOsb0brpayU3RTXdd8ZI",
  'authDomain': "fir-test-c0374.firebaseapp.com",
  'projectId': "fir-test-c0374",
  'storageBucket': "fir-test-c0374.appspot.com",
  'messagingSenderId': "1039503808359",
  'appId': "1:1039503808359:web:539714356b998cd38fc1c2",
  'measurementId': "G-2GNBXS00CK",
  'databaseURL': ""
}
```

Setamos as configurações do serviço de armazenamento do nosso Firebase (App Web).

```python
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
```

Inicializamos nosso App Firebase. E instanciamos o serviço de armazenamento.

```python
from keras.datasets import mnist
objects = mnist
(train_img, train_lab), (test_img, test_lab) = objects.load_data()
```

Importamos o DataSet MNIST do Keras, instanciando-o. Carregamos os dados e dividimos-os em imagens/dígitos de treino e teste. Observe que foi iniciado o download das imagens presentes no DataSet.

```python
plt.imshow(train_img[5])
print(train_lab[5])
```

Com auxílio do comando print e método **imshow()**, foi gerada uma imagem/dígito presente no DataSet para demonstração.

```python
for i in range(20):
  plt.subplot(4, 5, i+1)
  plt.imshow(train_img[i], cmap='gray_r')
  plt.title("Dígito: {}".format(train_lab[i]))
  plt.subplots_adjust(hspace=0.5)
  plt.axis('off')
```

Geramos uma tabela com diferentes imagens de dígitos manuscritos, cada qual correspondente à um digito diferente de 0 - 9.

```python
print('Shape imagens de treino: ', train_img.shape)
print('Shape imagens de teste: ', test_img.shape)
```

Usamos o método **shape** para gerar as informações de tamanho e formato de ambas as classes de imagens.

```python
train_img = train_img / 255.0
test_img = test_img / 255.0
```

Normalização das imagens.

```python
from keras.models import Sequential
from keras.layers import Flatten, Dense
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

Importamos e instanciamos o modelo sequencial do Keras, e os módulos _Flatten_ e _Dense_ da biblioteca de camadas. Adicionamos ao modelo quatro camadas: a de entrada do tipo Flatten, duas camadas ocultas com função de ativação _Relu_ e uma camada de saída com função de ativação _Softmax_. Estas três últimas do tipo Dense.

![mnist1](https://uploaddeimagens.com.br/images/004/281/387/full/mnist1.png?1672667073)

Observe que foi gerado as informações do modelo sequencial, tais como: tipo, sua saída junto do seu formato e os parâmetros.

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Compilamos o modelo sequencial.

```python
model.fit(train_img, train_lab, epochs=3)
```

Fitamos o modelo utilizando nossos conjuntos de dados de treino.

![mnist2](https://uploaddeimagens.com.br/images/004/281/392/full/mnist2.png?1672667253)

Ao fitarmos nosso modelo obtemos dados referentes à perdas e precisões de cada _epoch_, a quantidade de _epochs_ define o número de vezes que nosso modelo será treinado, podendo assim aumentar sua precisão. (O baixo número deve-se à quantidade de tempo de processamento)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Criamos um acesso ao Google Drive e montamos um diretório.

```python
model.save('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5')
```

Salvamos nosso modelo no Google Drive.

```python
file = '/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5'
cloudfilename = 'model.h5'
storage.child(cloudfilename).put(file)
```

Pegamos o arquivo armazenado no Drive e armazenamos em nosso BD Firebase.

```python
storage.child('model.h5').download('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Firebase/model.h5')
```

Baixamos o nosso model.h5 armazenado no Firebase e o armazenamos novamente no Drive.

```python
model.load_weights('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Quando formos treinar o modelo acessamos ele no diretório no Drive, e carregamos os **pesos**. E novamente compilamos o modelo sequencial.

```python
loss_and_acc = model.evaluate(test_img, test_lab, verbose=2)
print("Teste de perda", loss_and_acc[0])
print("Teste de precisão", loss_and_acc[1])
```

![mnist3](https://uploaddeimagens.com.br/images/004/281/399/full/mnist3.png?1672667692)

Avaliamos o modelo, e exibimos suas métricas de perda e precisão.

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

Realizamos uma predição de um dígito manuscrito presente no DataSet.

![mnist4](https://uploaddeimagens.com.br/images/004/281/402/full/mnist4.png?1672667770)

Nosso modelo conseguiu realizar com sucesso e rapidamente a predição de um dos dígitos manuscritos presentes no DataSet.

```python
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model

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

Criamos uma função que realizará o tratamento de uma imagem que será passada pelo usuário. Com auxílio dos módulos de processamento de imagem, presentes no Keras.

```python
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
  x = uploaded[filename]

img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
```

Usamos a biblioteca do Colab para realizar o upload da imagem.

```python
img = load_image(filename)
label = int(input('Número atual = '))
predict = model.predict(img)
classify = np.argmax(predict)
print('Valor predito:', classify)

if(label == (np.argmax(predict))):
  print('Previsão bem-sucedida')
else:
  print('Previsão sem sucesso')

show = cv2.imread(filename)
plt.imshow(show)
```

Por fim realizamos a predição da imagem enviada pelo usuário.

![mnist5](https://uploaddeimagens.com.br/images/004/281/406/full/mnist5.png?1672667948)

O valor enviado pelo usuário foi predito com sucesso.

## Autores

* [@EdivalcoAraujo](https://github.com/EdivalcoAraujo)
* [@Jeef-Moreira](https://github.com/Jeef-Moreira)