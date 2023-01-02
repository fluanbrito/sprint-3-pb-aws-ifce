""" Pra usar o pymongo rodar: 

1. python -m venv env source env/bin/activate
2. python -m pip install "pymongo[srv]" 

"""

from pymongo import MongoClient
import json

def get_database():
    CONNECTION_STRING = "mongodb://localhost:27017"
    try:
        client = MongoClient(CONNECTION_STRING)
        "Conectou"
        return client['rede-neural']
    except:
        raise ConnectionError("Não conectou")
  
if __name__ == "__main__":   
  
    bd_rede_neural = get_database()
    
    data = bd_rede_neural["modelo-treinado"]

    jsonfile = open('model.json', 'r')
    loaded_model_json = json.load(jsonfile)
    jsonfile.close()

    data.insert_one(loaded_model_json)