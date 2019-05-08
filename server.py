from PIL import Image
import imutils
import cv2
import numpy as np
from flask import render_template, Flask, request, make_response
import passport
import json
import os

app = Flask(__name__)
DEBUG = True


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        image = request.files['file']
        image = np.array(Image.open(image))

        responce = handle_passport(image)

        return json.dumps(responce, ensure_ascii=False)

def handle_passport(image):
    """
    Function for full handling of passport image
    :param image: np array
    :return: dict
    """

    # Preprocessing
    image = imutils.resize(image, width=1000)
    image = passport.rotate_passport(image)
    image = passport.cut_passport(image)
    image = passport.skew_text_correction(image)

    # Cutting passport into parts and reading these
    (h, w, _) = image.shape
    box = image[int(h / 2):h, int(w / 3):w]
    bottom = passport.locate_text(box, type_='bottom')

    box = image[0:int(h / 3), 0:w]
    top = passport.locate_text(box, type_='top')

    image_ = imutils.rotate_bound(image, -90)
    (h, w, _) = image_.shape
    number = passport.read_text_from_box(image_, 0, 0, w, int(h / 10))

    # Processing the text
    responce = passport.procces_passport(top, bottom, number)
    return responce


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
