from insert_bd import get_database
from tensorflow import keras
from bson.json_util import dumps
from PIL import Image
import numpy as np
from tkinter import filedialog

if(__name__ == "__main__"):
   # Conecta ao banco de dados:
   bd_rede_neural = get_database()
   
   # Acessa a coleção:
   collection = bd_rede_neural["modelo-treinado"]
   
   # Realiza a consulta e converte para json
   item_details = collection.find()
   model = dumps(item_details[0])

   # Lê o modelo e compila:
   loaded_model = keras.models.model_from_json(model)
   loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   # Lê os pesos:
   loaded_model.load_weights("model-weights.h5")

   print("\nModelo lido com sucesso!\n")

   classes = ["gato", "cachorro"]

   # abre o gerenciador de arquivos para escolha da imagem:
   imagem_escolhida = filedialog.askopenfilename()
   test_image = Image.open(imagem_escolhida).convert('RGB').resize((64,64))

   # Faz a predição
   pred = loaded_model.predict(np.expand_dims(test_image, axis = 0))
   print("A imagem é: %s \t valor predito: (%.2f)" % (classes[int(pred>0.5)], pred[0][0]))    