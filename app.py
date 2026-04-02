import tensorflow as tf
import numpy as np
from PIL import Image

from flask import Flask, render_template, request
import os

app = Flask(__name__)

# folder upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model (sekali saja)
model = tf.keras.models.load_model("model_buah.h5")

# fungsi prediksi
def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150,150))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Busuk 💀"
    else:
        return "Segar 🍎"

# route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    result = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_path = filepath

            # prediksi AI
            result = predict_image(filepath)

    return render_template('index.html', image_path=image_path, result=result)

# jalankan app
if __name__ == "__main__":
import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)