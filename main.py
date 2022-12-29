from io import BytesIO

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

from readModel import modelTF

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)

def criartabela():
    db.create_all()
    pass

def addmodel(nome,patch):
    def convert_into_binary(file_path):
        with open(file_path, 'rb') as file:
            binary = file.read()
        return binary
    data=convert_into_binary(patch)
    upload = Upload(filename=nome, data=data)
    db.session.add(upload)
    db.session.commit()
    pass

def criarh5(binary_data, file_name):
  with open(file_name, 'wb') as file:
    file.write(binary_data)

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/", methods=["POST"])
def post_file():
    arquivo=request.files.get("minhaImage")
    modelo=request.files.get("minhaimagem")
    patch="images/"+arquivo.filename
    arquivo.save(patch)
    a=modelTF.readModel(patch,"modelo_treinado\modelo.h5")
    return a[0]



if __name__ == '__main__':
    app.run(port=3000)