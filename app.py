from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow import keras
from keras.applications.resnet import ResNet50
import keras.utils as image
from keras.applications.resnet import preprocess_input, decode_predictions


app = Flask(__name__)

model = ResNet50(weights='imagenet')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file.save(filename)
            img = image.load_img(filename, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            results = decode_predictions(preds, top=3)[0]
            return render_template("results.html", results=results)
    return "Error"

if __name__ == "__main__":
    app.run(debug=True)
