import json

import cv2
import flask
import numpy as np
from keras.models import model_from_json

app = flask.Flask(__name__)
model = None
WIDTH = HEIGHT = 48
x = y = None
labels = ['Sad', 'Disgust', 'Happy', 'Surprised', 'Neutral', 'Disappointed', 'Angry']
clf_paths = ['../haarcascade_frontalface_default.xml', '../haarcascade_frontalface_alt.xml',
             '../haarcascade_frontalface_alt2.xml']


def load_model():
    global model
    json_file = open('models/fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models/fer.h5")
    print("Loaded model from disk")


def find_faces(image, width=WIDTH, height=HEIGHT):
    gray = cv2.imdecode(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    faces = []
    for clf_path in clf_paths:
        print(f'using detector: {clf_path}, faces finded: {len(faces)}')
        face = cv2.CascadeClassifier(clf_path)
        faces = face.detectMultiScale(gray, 1.1, 3)
        if len(faces):
            print('faces finded, stopping detector')
            return gray, faces

    if not len(faces):
        print('no faces finded:(')
    return gray, faces


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = np.fromstring(image, np.uint8)
            full_size_image = image.copy()
            image, faces = find_faces(image)

            for (x, y, w, h) in faces:
                roi_gray = image[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                ypred = model.predict(cropped_img)
                print(type(ypred), ypred.shape)
                emotions_info = dict(zip(labels, ypred[0]))
                cv2.putText(full_size_image, labels[int(np.argmax(ypred))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, cv2.LINE_AA)
                print("Emotion: " + labels[int(np.argmax(ypred))])
                data['predictions'] = {}
                for i_emotion in labels:
                    data['predictions'][i_emotion] = float(emotions_info[i_emotion])

            data['ypred'] = ypred.astype(float).tolist()
            data['faces'] = faces.astype(float).tolist()
            print(type(data['ypred']), type(data['faces']))
            data["success"] = True
            print(data)
            data_json = json.dumps(data)

    return data_json


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
