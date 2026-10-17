import random
import base64
import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)


# █░░ █▀█ ▄▀█ █▀▄   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░
# █▄▄ █▄█ █▀█ █▄▀   █░▀░█ █▄█ █▄▀ ██▄ █▄▄
model = tf.keras.models.load_model("modelRAF.h5")

# 6 emotii 
# !ordinea trebuie să fie exact ca la antrenare - bug reparat
emotions = ['neutral', 'sad', 'disgust', 'surprise', 'happy', 'angry']



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


CONFIDENCE_THRESHOLD = 0.25


# █▀▀ █▀█ █▀█ █▄░█ ▀█▀ █▀▀ █▄░█ █▀▄
# █▀░ █▀▄ █▄█ █░▀█ ░█░ ██▄ █░▀█ █▄▀
PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>FER Camera</title>
</head>
<body>
    <h2>Real-Time Emotion Detection</h2>

    <video id="video" autoplay playsinline width="600"></video>
    <canvas id="canvas" style="display:none;"></canvas>

    <h3 id="result">Waiting...</h3>

    <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        const result = document.getElementById("result");

        navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        });

        setInterval(() => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            ctx.drawImage(video, 0, 0);

            const image = canvas.toDataURL("image/jpeg");

            fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ image })
            })
            .then(r => r.json())
            .then(data => {
                const probPercent = data.probability !== undefined
                    ? " (" + (data.probability * 100).toFixed(1) + "%)"
                    : "";
                result.innerText = "Emotion: " + data.emotion + probPercent;
            });

        }, 1500);
    </script>
</body>
</html>
"""



# █▀█ █▀█ █░█ ▀█▀ █▀▀ █▀
# █▀▄ █▄█ █▄█ ░█░ ██▄ ▄█
@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    # decode base64
    img_str = data.split(",")[1]
    img_bytes = base64.b64decode(img_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"emotion": "no face"})

    x, y, w, h = faces[0]
    face = gray[y:y + h, x:x + w]

    # preprocesare (48x48 grayscale, [0,1])
    face = cv2.resize(face, (48, 48))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)

    # modelRAF.h5 are deja softmax la final
    prediction = model.predict(face)[0]

    max_prob = float(np.max(prediction))
    max_idx = int(np.argmax(prediction))

    if max_prob < CONFIDENCE_THRESHOLD:
        return jsonify({
            "emotion": "no emotion",
            "probability": max_prob
        })

    emotion = emotions[max_idx]
    return jsonify({
        "emotion": emotion,
        "probability": max_prob
    })


# █▀█ █░█ █▄░█   █▀ █▀▀ █▀█ █░█ █▀▀ █▀█
# █▀▄ █▄█ █░▀█   ▄█ ██▄ █▀▄ ▀▄▀ ██▄ █▀▄
if __name__ == "__main__":
    print("Starting RAF server at http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
