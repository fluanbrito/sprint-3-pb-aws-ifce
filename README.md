# Avaliação da Sprint 3

Documentando o desenvolvimento da avaliação do projeto de bolsas da Compass UOL

### 📚 Para realização da avaliação, foram utilizados as seguintes bibliotecas e ferramentas estudadas na Sprint 3:

#### Tensor Flow
Biblioteca de código-fonte aberto para Computação Numérica em Python, facilitando e agilizando o aprendizado de máquinas

### Keras 
Biblioteca aberta de Deep Learning implementada utilizando o TensorFlow para diversas linguagens/plataformas, entre elas o Python, com foco na facilidade para a utilização.

### Numpy
Biblioteca para cálculos fáceis e rápidos. Utiliza-se seus Arrays para armazenar dados de treinamento assim como parâmetros de modelos

### Matplotlib
Biblioteca Python para visualização gráfica  e plotagem 2d de dados, auxiliando na análise deles.

### %matplotlib inline
Comando do matplotlib utilizado para posicionar o resultado do comando abaixo da cedula onde ele é executado.

### Pandas 
Também é uma biblioteca para Python, fornece ferramentas de análise de dados e estruturas de alta performance fáceis de usar;

### PIL
Biblioteca Python que permite a manipulação de imagens.



### 📋 Objetivo

Construir uma rede neural capaz de classificar imagens entre gatos e cachorros.



### 🖥️ Desenvolvimento

O dataset escolhido foi o **oxford_iiit_pet**, ele é um conjunto de dados de animais de estimação com 37 categorias com aproximadamente 200 imagens para cada classe. As imagens têm grandes variações de escala, pose e iluminação. Todas as imagens têm uma anotação de fundo de raça associada.

Foi criado um arquivo csv com base no txt com as classes e feito o tratamento das imagens usando funções do pandas e PIL pra deixar no formato esperado pela rede neural. 
Após isso foi pré-processado os dados e construído o modelo com o uso das bibliotecas matplotlib e tensorflow.A mesma foi construída usando:
- Data Augmentation: técnica usada para permitir que as imagens sejam rotacionadas e aplicado zoom de forma aleatória, com o objetivo de melhorar o desempenho da rede
- 4 camadas de Convolução: responsável por extrair informações da imagem
- 4 camadas de Pooling: responsável pela diminuição da dimensão dos dados 
- 1 camada de Flatten:  realiza um "achatamento" nos dados, deixando-os num formato de array
- 1 camada de Rede totalmente conectada
- 1 camada de Dropout: técnica usada para evitar overfitting baseada no desligamento aleatório de alguns neurônios 
- 1 camada de Saída: é a que contém o resultado final, usa a função _sigmoid_

Também vale ressaltar que o otimizador utilizado foi o Adam com _learning_rate_ = 0.001 e a função de custo foi a _binary_crossentropy_.

 Depois, foi realizado o treinamento e a análise do desempenho:
![Logo do R](https://user-images.githubusercontent.com/80013300/210249077-094cbade-8246-46e6-8417-87a3a425057e.png)



    


### Conclusão
 A avalição foi concluída com êxito e seu desenvolvimento foi essencial para a equipe em conjunto fixar os conceitos estudados na sprint 3, além de expandir e aprimorar práticas sobre os conteúdos estudados.

---
---