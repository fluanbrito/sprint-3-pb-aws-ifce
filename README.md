# Utilização do DataSet MNIST como forma de gerar imagens escritas a mão
## Este é o projeto onde iremos desenvolver o DataSet MNIST que é um DataSet que trabalha em gerar imagens de caligráfias escritas a mão.

[![N|Solid](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/LogoCompasso-positivo.png/440px-LogoCompasso-positivo.png)](https://compass.uol/pt/home/)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/Jeef-Moreira)

Neste arquivo se encontra detalhado todo o experimento de como foi desenvolvido ena linguagem Python todo o procedimento de consumo de um determinado DataSet, neste caso, o MNITS.. Será apresentado:
- Apresentação do MNIST
- Ferramentas
- ✨Comandos✨

## APRESENTAÇÃO DO MNIST
MNIST: 
O conjunto de dados MNIST é um acrônimo que significa o conjunto de dados Instituto Nacional Modificado de Padrões e Tecnologia.
É um conjunto de dados de 60.000 pequenas imagens quadradas de 28 × 28 pixels em escala de cinza de dígitos manuscritos entre 0 e 9.
A tarefa consiste em classificar uma dada imagem de um dígito manuscrito em uma das 10 classes que representam valores inteiros de 0 a 9, inclusive.

É um conjunto de dados amplamente usado e profundamente compreendido e, na maioria das vezes, é “resolvido”. Os modelos de melhor desempenho são redes neurais convolucionais de aprendizado profundo que atingem uma precisão de classificação acima de 99%, com uma taxa de erro entre 0,4% e 0,2% no conjunto de dados de teste de retenção.


## FERRAMENTAS

As principais ferramentas utilizadas no projeto foram:

- [DataSet MNIST](https://www.tensorflow.org/datasets/catalog/mnist) - É uma API pública capaz de gerar informações como: cidade, estado, CEP e etc, isso sendo necessário só adicionar um CEP valido.
- [Google Colab](https://colab.research.google.com/) - É uma plataforma de software para aplicativos escaláveis do lado do servidor e de rede. Os aplicativos Node.js são escritos em JavaScript e podem ser executados no tempo de execução Node.js no Mac OS X, Windows e Linux sem alterações.
- [Visual Studio Code v.1.73.1](https://code.visualstudio.com/) - O Visual Studio Code (VS Code) é um editor de código de código aberto desenvolvido pela Microsoft. Ele está disponível para Windows, Mac e Linux. Nesse caso, ele foi usado em prol do desenvolvimento deste README do projeto.


## Comandos

Busque um DataSet de sua preferência, no meu caso escolhemos o DataSet MNIST.

```
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
```

Obs.: lembre de substituir o parametro de =="test"== por =="start"== + nodemon <nome_da_aplicação.js>. No meu caso: =="start": "nodemon app.js"==. Isso é necessário para a execução do nodemon no nó da aplicação para sempre fazer o reset quando for feito uma nova alteração.

Feito a etapa anterior, abra o ==app.js== é visualize as rotas que foram criadas para acessar a API. No caso vou mostrar logo abaixo é detalhar um pouco sobre elas.
==app.js==
```
from keras.datasets import mnist
objects=mnist
(train_img,train_lab), (test_img,test_lab)=objects.load_data()
```

Observe que foi iniciado o Download das Imagens
* Agora, crie um arquivo chamado ==app.js== preenchendo todas as informações, ele deve ficar parecido como algo deste tipo:

```
plt.imshow(train_img[5])
print(train_lab[5])
```
Veja que com a utilização do comando Print, foi gerada a imagem pertecente ao Train
```
for i in range(20):
  plt.subplot(4,5,i+1)
  plt.imshow(train_img[i], cmap='gray_r')
  plt.title("Digit : {}".format(train_lab[i]))
  plt.subplots_adjust(hspace=0.5)
  plt.axis('off')
```
Caso você tenha exercido cada linha de comando correta, foi gerado uma tabela com diferentes imagens de caligrafia escritas a mão cada qual correspondente á um digito diferente.
* Agora, crie um arquivo chamado ==index.html== da seguinte maneira.

```
print('Training images shape : ',train_img.shape)
print('Testing images shape : ',test_img.shape)
```
Comando usado para gerar as informações de tamanho e formato de ambas as imagens. Observe o resultado:

```
train_img=train_img/255.0
test_img=test_img/255.0
```



```
from keras.models import Sequential
from keras.layers import Flatten,Dense
model=Sequential()
input_layer= Flatten(input_shape=(28,28))
model.add(input_layer)
hidden_layer1=Dense(512,activation='relu')
model.add(hidden_layer1)
hidden_layer2=Dense(512,activation='relu')
model.add(hidden_layer2)
output_layer=Dense(10,activation='softmax')
model.add(output_layer)
model.summary()
```
Observe que foi gerado as informações do model sequencial, tais como: tipo, sua saída junto do teu formato e os parâmetros.
```
#compiling the sequential model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
```

* Confirme se a imagem foi criada atráves do comando:
```
model.fit(train_img,train_lab,epochs=10)
```
Este comando irá lista todas as imagens baixadas no teu diretório.
* Para executar o docker container com esta imagem, use o seguinte comando.
```
# Mengakses Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

* Abra o navegador de tua preferência e entre com o endereço localhost:5001 , e nosso aplicativo expresso retornará a resposta executada.

```
# Gunakan untuk menyimpan di Google Drive
model.save('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5')
```

```
# Gunakan hanya joka lagi men-training model
model.load_weights('/content/drive/MyDrive/Colab Notebooks/Remote Sensing/Save/model.h5')
#compiling the sequential model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
```

```
loss_and_acc=model.evaluate(test_img,test_lab,verbose=2)
print("Test Loss", loss_and_acc[0])
print("Test Acurracy", loss_and_acc[1])
```
Observe os resultados apurados:

```
plt.imshow(test_img[0],cmap='gray_r')
plt.title('Actual Value: {}'.format(test_lab[0]))
prediction=model.predict(test_img)
plt.axis('off')
print('Predicted Value: ',np.argmax(prediction[0]))
if(test_lab[0]==(np.argmax(prediction[0]))):
  print('Sucessful prediction')
else:
  print('Unsuccessful prediction')
```
Após o carregamento da imagem, será sinalizado o status da operação junto da imagem.

```
from IPython.display import Image
Image(test_img[2],width=250,height=250)
#plt.imshow(test_img[2],cmap='gray_r)
```

```
# make a prediction for a new image
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
  # load the image
  img = load_img(filename, grayscale=True, target_size=(28,28))
  # convert to array
  img = img_to_array(img)
  # reshape into a single sample with 1 channel
  img = img.reshape(1, 28, 28)
  # prepare pixel data
  img = img.astype('float32')
  img = img/255.0
  return img
```

```
from google.colab import files
uploaded = files.upload()
for filename in uploaded.keys():
  x=uploaded[filename]
img=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
```
Sinaliza se foi realizado o Upload do arquivo 
```
from IPython.display import Image
Image(test_img[2],width=250,height=250)
#plt.imshow(test_img[2],cmap='gray_r)
```

```
img = load_image(filename)
label=int(input('Actual Number= '))
predict=model.predict(img)
classify=np.argmax(predict)
print('Predicted Value : ',classify)
if(label==(np.argmax(predict))):
  print('Sucessful prediction')
else:
  print('Unsuccessful prediction')
show=cv2.imread(filename)
plt.imshow(show)
```




CONCLUSÃO:

 

## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dev]: <https://github.com/Jeef-Moreira>
   [dev]: <https://github.com/EdivalcoAraujo>
   [learning]: <https://compassuol.udemy.com>
   [boss]: <https://compass.uol/pt/home>
   [Google Colab]: <https://colab.research.google.com/>
   [Google Drive]: <https://drive.google.com/drive/my-drive>
   [Visual Studio Code]: <https://code.visualstudio.com>
  
