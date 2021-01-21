from flask import Flask, request
import os
from cv2 import cv2
import numpy as np
import urllib.request
from flask_cors import CORS, cross_origin

from Binary_model import *
from tensorflow.keras.applications import efficientnet


app = Flask(__name__)



cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def index():
  return "<h1>Welcome to CodingX</h1>"

@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # return the image
    return image

#URL = 'https://res.cloudinary.com/dasqhv6aq/image/upload/v1611130059/kmpa8rzlr5fovtg8tha3.png'


@app.route('/bin_pred', methods=['POST'])
def bin_pred():

    data = request.get_json(force=True)
    img_url = data[0]['url']
    image = url_to_image(img_url)
    result = binary_model.predict(efficientnet.preprocess_input(image).reshape(1, 224, 224, 3))
    print(result)
    prediction = result > 0.5
    print(prediction)
    return f"{prediction[0]}"



#@app.route('/multi_pred', methods=['POST'])
#def multi_pred():
#
#    data = request.get_json(force=True)
#    img_url = data[0]['url']
#    image = url_to_image(img_url)
#    result = binary_model.predict(efficientnet.preprocess_input(image).reshape(1, 224, 224, 3))
#    prediction = result > 0.5
#
#    return f"{prediction[0]}"






if __name__ == '__main__':
      app.run(threaded=True, port=5000)
  
  
  
    # port = os.environ.get('PORT')
    # if port:
    #     app.run(host='0.0.0.0', port=int(port))
    # else:
    #     app.run(port=9000, debug=True)