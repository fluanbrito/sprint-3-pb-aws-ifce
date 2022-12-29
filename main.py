from flask import Flask, render_template, request

from readModel import modelTF

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    #a=modelTF.readModel("images\passaro.jpg","modelo.h5")
    return render_template('index.html')

@app.route("/", methods=["POST"])
def post_file():
    arquivo=request.files.get("minhaImage")
    path="images/"+arquivo.filename
    arquivo.save(path)
    a=modelTF.readModel(path,"modelo.h5")
    return a[0]



if __name__ == '__main__':
    app.run(port=3000)