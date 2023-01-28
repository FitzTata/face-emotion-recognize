import time

import cv2
import numpy as np
import requests

from run_server import labels

IMAGE_PATH = "../test_pics/test1.jpg"

KERAS_REST_API_URL = "http://localhost:5000/predict"
start_time = time.time()
cur_time = None
image = open(IMAGE_PATH, "rb").read()
full_size_image = cv2.imread(IMAGE_PATH)
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()
print(type(r), r)
if r['success']:
    r['faces'] = np.asarray(r['faces']).astype(int)
    r['ypred'] = np.asarray(r['ypred'])
    r['predictions'] = {k: v for k, v in sorted(r['predictions'].items(), key=lambda x: x[1], reverse=True)}
    print(r)
    cur_time = time.time()
    for (i, result) in enumerate(r['predictions']):
        print(f'{i + 1}. {result} --> {r["predictions"][result]:.4f}')

    print(f'time: {cur_time - start_time}')
    # c_image = np.fromstring(image, np.uint8)
    for (x, y, w, h) in r['faces']:
        print('drawing')
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(full_size_image, labels[int(np.argmax(r['ypred']))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: " + labels[int(np.argmax(r['ypred']))])

    cv2.imshow('Emotion', full_size_image)
    cv2.waitKey()


else:
    print("Request failed")
