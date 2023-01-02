# _DataSet "horses_or_humans"_ (Avaliação Sprint 3 - Programa de Bolsas Compass.uol / AWS e IFCE)_

### Integrantes Grupo-2:
- Francisco Luan
- Guilherme de Oliveira
- Rosemelry Mendes

## 📚 Sobre o DataSet
O DataSet escolhido foi o _horses_or_humans_ no qual ele reconhece e classifica imagens de cavalos ou humanos. O conjunto contém 500 imagens renderizadas de várias espécies de cavalos em várias poses em vários locais. Ele também inclui 527 imagens renderizadas de humanos em diferentes poses e planos de fundo. Foi enfatizado a diversidade dos humanos, então há homens, mulheres, asiáticos, negros, sul-asiáticos e caucasianos presentes no conjunto de treinamento.

## Recursos Necessários
- [DataSet Horses or Humans](https://www.tensorflow.org/datasets/catalog/horses_or_humans) - DataSet de Imagens Cavalos vs Humanos.
- [Visual Studio Code v.1.73.1](https://code.visualstudio.com/) - VS Code.
- [Google Colab](https://colab.research.google.com/) - Ferramenta que permite a misture de código fonte (geralmente em python) e texto rico (geralmente em markdown) com imagens e o resultado desse código. Técnica conhecida como: notebook.
- [MongoDB](https://www.mongodb.com/home) - Banco de dados orientado a documentos livre, de código aberto e multiplataforma. Classificado como um programa de banco de dados NoSQL, usa documentos semelhantes a JSON com esquemas.

## Impedimentos
Tivemos algumas dificuldades, uma delas foi a demora de processamento na sessão de treinamento, mais precisamente nos _Epoch_. Fazendo alguns testes, o tempo de processamento diminuia consideravelmente, porém obteve vários erros de _Predicted_, então optamos por deixar da maneira inicial (processamento lento).

## Observação 1:
Foi colocado comentários com a Tag (#) em todo o código (aqui anexados), sinalizando o passo-a-passo de cada etapa, com isso será citado com exemplos aqui no Readme somente alguns pontos mais relevantes, assim otimizando a quantidade de informações descritas aqui.

## Etapas de desenvolvimento
- Seção de Importação:

Nessa seção foram baixados os arquivos de exemplo do DataSet (https://laurencemoroney.com/datasets.html#horses-or-humans-dataset), como o dataset de treino e de teste, como também foi extraído os arquivos zip e os arquivos para validação. Segue os comando para baixar o dataset (treino/teste).
```
#baixando o dataset de treino
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip  -O /tmp/horse-or-human.zip
```
```
#baixando o dataset de teste
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip  -O /tmp/validation-horse
```
- Seção de Vizualização de Dados:

Nessa seção teremos os comando que será _printado_ os nomes dos arquivos e o total de imagens (horse/human). Como também a criação do gráfico que nos mostra algumas imagens (horse/human).
![imagens](https://user-images.githubusercontent.com/106123150/210187576-93052d0c-cc18-43b6-a67c-2d37d3760f70.png)

- Seção de Treinamento (Criando Modelo):

Nessa seção teremos a criação do modelo neural com base nos dados do dataset, onde tivemos cinco _convoluções_ (é extraída características da imagem de entrada). Onde foi usado apenas um neurônio como saída, sendo 0 para cavalos e 1 para humanos. Como mostra a seguir o resumo desse modelo, onde tivemos 1.704,097 parâmetros treináveis e 0 (zero) parâmetros não treináveis.

![Captura de Tela (113)](https://user-images.githubusercontent.com/106123150/210187921-a3b77f4d-da39-4be3-98d9-bef60e78e66e.png)

## Observação 2:
O modelo já foi treinado e pode ser baixado na próxima seção. Caso prefira treinar o modelo, pode executar a seção anterior e ignorar a próxima.

- Seção de Importação do modelo(MongoDB)

Nessa seção teremos a importação do modelo para MongoDB (banco de dados escolhido), no qual foi usado o seguinte comando para a instalação do wrapper do mongo para python.
```
! python -m pip install pymongo==3.7.2
```
Em seguida foram feitas as importações das dependências para o banco de dados e para o tensorflow.

**OBS:** Foi feito um login público, para que todos possam acessar o modelo, mas apenas administradores podem alterar o modelo.
```
uri = 'mongodb+srv://public:public@cluster0.aqfdkqq.mongodb.net/test'
```
- uri: é o que define os parâmetros da conexão.

O próximo comando inicia uma conexão a partir do _uri_ e cria um cliente:
```
client = MongoClient( uri )
```
Em seguida foram usados os comandos em que é selecionado o banco, é pego a referência do modelo de acordo com o nome, depois é baixado do banco o modelo e por fim é feito o carregamento desse modelo, como mostra a imagem a seguir:
![Captura de Tela (117)](https://user-images.githubusercontent.com/106123150/210188601-9fc4d7c0-04c3-456d-baf7-422b85a457e0.png)

- Seção de Teste e Validação:
Nessa seção teremos a criação de uma função para testar o "dataset de testes" e em seguida foi feito o teste com cavalos percorrendo uma lista e testando, como também foi feito o mesmo teste com humanos, percorrendo uma lista e testando, como mostra as imagens a seguir:
![Captura de Tela (125)](https://user-images.githubusercontent.com/106123150/210189142-354634e2-5d1d-4b88-8d81-5d532574afbd.png)
![Captura de Tela (123)](https://user-images.githubusercontent.com/106123150/210189205-c2d38795-4cc7-44e6-9685-1a3393f43728.png)

- Seção de Gráfico e Métricas de Desempenho:

Nessa seção teremos um gráfico para a visualização das métricas de desempenho onde a faixa laranja corresponde a _perda_(loss) e a faixa azul corresponde a _precisão_(accuracy).
![img grafico](https://user-images.githubusercontent.com/106123150/210189347-ca5fc239-f0cc-4ed0-bdfe-32739358ab7b.png)

- Seção de Visualização das Camadas/Layers:

Nessa seção teremos a demonstração das camadas, no qual foi feita a importação do pacote _numpy_ para se trabalhar com computação numérica e em seguida é mostrado as imagens dessas camadas, como mostra a seguir:
![Captura de Tela (112)](https://user-images.githubusercontent.com/106123150/210189519-a3701229-e940-43df-8d8b-73b5ac064f28.png)

- Seção de Reconhecer um novo exemplo:

Nessa seção teremos o reconhecimento de um novo exemplo, onde faremos as devidas importações, o Upload de arquivo para teste, e por fim uma condicional para _printar_ de acordo com o resultado. Será feito o download das imagens(cavalo e humano) que tenha a mesma resolução das imagens treinadas(300x300) e ao fazer o _predicted_ será nos informado se aquela imagem é "human or horse".
```
#teste passando um arquivo 
import keras
import tensorflow as tf
import numpy as np
from google.colab import files
from keras.preprocessing import image
 
uploaded = files.upload()
 
for fn in uploaded.keys():
 
  path = '/content/' + fn
  img = load_img(path, target_size=(300,300))
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
```
Resultado Final:

- Ao passar uma imagem de um cavalo(teste 1) ele reconheceu e nos informou que aquela imagem era de um cavalo(horse).
- Ao passar uma imagem de um humano(teste 2) ele reconheceu e nos informou que aquela imagem era de um humano(human).

![Captura de Tela (118)](https://user-images.githubusercontent.com/106123150/210189805-155efb2a-b2ad-49bc-b11e-a18a4ee9f468.png)
![Captura de Tela (119)](https://user-images.githubusercontent.com/106123150/210189815-1dcd60ea-96eb-48a0-90e2-e64b84864a32.png)

