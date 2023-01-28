# face-emotion-recognize #FER #keras #multi-faces #rest-api
Face emotions recognise

Solving Face emotions recognise problen with keras-built nn.
Dataset : FER 2013 (https://www.kaggle.com/datasets/msambare/fer2013)
Referencies: Haar cascade classifiers (https://github.com/opencv/opencv/tree/master/data/haarcascades)
Stack: Keras, TF, numpy, REST-API keras, flask

How to use:
1. Copy repository from github using

git clone https://github.com/FitzTata/face-emotion-recognize

2. Run file run_server.py
3. Put photos in ./test_pics
4. Edit client.py path to photo like that

ln9: IMAGE_PATH = "../test_pics/test1.jpg" to "../test_pics/your_pic.jpg"

5. Run file client.py
6. You and your photo are awesome! You will see result in about 3 seconds (depends on your pc resources).

Example:
![alt text](img/sample.png)
