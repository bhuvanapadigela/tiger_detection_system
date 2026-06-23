from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("tiger_model.h5")

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    image_path = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        image_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            file.filename
        )

        file.save(image_path)

        img = image.load_img(
            image_path,
            target_size=(150, 150)
        )

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction >= 0.5:
            result = "Tiger"
            confidence = round(prediction * 100, 2)
        else:
            result = "Not Tiger"
            confidence = round((1 - prediction) * 100, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)