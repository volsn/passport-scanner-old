from PIL import Image
import imutils
import cv2
import numpy as np
from flask import render_template, Flask, request, make_response
import passport
import json
import os
from pytesseract import image_to_string

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
    image = passport.rotate_passport(image)
    image = passport.cut_passport(image)

    # Cutting passport into parts and reading these
    (h, w, _) = image.shape
    top = image[0:int(h/2), 0:w]
    name = image[int(h/2 + h/20):int(h-h/4),int(w/3):w]
    bottom = image[int(h/2): h, 0:w]

    number = passport.read_passport_number(image)
    full_name = passport.find_person_name(name)
    bottom = image_to_string(bottom, lang='rus')
    top = image_to_string(top, lang='rus')

    # Processing the text
    responce = passport.procces_passport(full_name, top, bottom, number)
    #responce['mrz'] = passport.read_passport_mrz(image)

    return responce


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
