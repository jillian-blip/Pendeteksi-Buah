import os
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

# folder upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔥 DOWNLOAD MODEL DULU
if not os.path.exists("model_buah.h5"):
    url = "https://drive.google.com/uc?id=1YcbGrFOxVFrSiNBa0YdJO1MVSPWW-KzG"
    gdown.download(url, "model_buah.h5", quiet=False)

# 🔥 BARU LOAD MODEL
model = tf.keras.models.load_model("model_buah.h5")

# fungsi prediksi
def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150,150))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    return "Busuk 💀" if prediction > 0.5 else "Segar 🍎"

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

            result = predict_image(filepath)

    return render_template('index.html', image_path=image_path, result=result)

# 🔥 PALING BAWAH
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)