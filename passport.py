from PIL import Image
import pdf2image
import imutils
from pytesseract import image_to_string
import cv2
import os
import numpy as np
import re


def rotate_passport(image):
    """
    rotating an image so passport could be readed
    :param image: np array
    :return: np array
    """

    # Initializing cascade
    cascade = cv2.CascadeClassifier('cascade.xml')
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    rotates = 0
    # Looking for a face
    for _ in range(4):

        face = cascade.detectMultiScale(gray, 1.3, 5)

        if face is not ():
            return imutils.rotate_bound(image, 90 * rotates)

        gray = imutils.rotate_bound(gray, 90)
        rotates += 1

    # Return false if the given picture is not a passport
    return False


def cut_passport(image):
    """
    Cutting an image so only passport was left
    :param image: np array
    :return: np array
    """

    # Initializing cascade
    cascade = cv2.CascadeClassifier('cascade.xml')
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Finding a face
    face = cascade.detectMultiScale(gray, 1.3, 5)

    # Cutting the image so only passport was left
    (x, y, w, h) = face[0]
    image = image[y - 6 * h: y + 3 * h, x - w:x + 6 * w]

    return image


def read_text_from_box(image):
    """
    Reading text from bounding box
    :param image: np array
    :return: string
    """

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)

    blurred = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 225,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = image_to_string(thresh, lang='rus').replace('\n', ' ')

    return text


def procces_passport(top, bottom, number):
    """
    Classifing data for given passport photo
    :param top: string
    :param bottom: string
    :param number: string
    :return: dict
    """

    passport = {
        'ocr_result': {
            'doc_type': 'passport',
            'issue_authority': '',
            'issue_code': '',
            'issue_date': '',
            'surname': '',
            'name': '',
            'patronymic_name': '',
            'birth_date': '',
            'gender': '',
            'birth_place': '',
            'series': '',
            'number': '',
        },
        'text': top + bottom + number,
    }

    # Looking for issue code
    code = re.search(r'\d{3}-\d{3}', top)
    if code is not None:
        passport['ocr_result']['issue_code'] = code[0]

    # Looking for issue authority

    AUTHORITIES = ['отделом', 'УФМС', 'МФЦ']

    for auth in AUTHORITIES:
        if re.search(auth, top, flags=re.I) is not None:

            issued = re.search(r'{}.*'.format(auth), top, flags=re.I)[0].split(' ')
            authority = ''
            for i in issued:
                if all(c.isupper() or c == '-' or c == '.' for c in i):
                    authority += i + ' '

            passport['ocr_result']['issue_authority'] = authority

    # Looking for issue date
    date = re.search(r'\d{2}\.\d{2}\.\d{4}', top)
    if date is not None:
        passport['ocr_result']['issue_date'] = date[0]

    # Looking for name
    full_name = re.search(r'(.*(ВИЧ|ВНА))', bottom, flags=re.I)

    name = []
    if full_name is not None:
        full_name = full_name[0].split(' ')

        for n in full_name:
            if all(c.isupper() for c in n) and len(n) > 3:
                name.append(n)

        if len(name) >= 3:
            passport['ocr_result']['patronymic_name'] = name[-1]
            passport['ocr_result']['name'] = name[-2]
            passport['ocr_result']['surname'] = name[-3]

    # Looking for birth date
    date = re.search(r'\d{2}\.\d{2}\.\d{4}', bottom)
    if date is not None:
        passport['ocr_result']['birth_date'] = date[0]

    # Looking for gender

    if passport['ocr_result']['patronymic_name'].endswith('ВИЧ') \
            or re.search(r'(МУЖ|МУЖ.) ', bottom) is not None:
        passport['ocr_result']['gender'] = 'male'
    elif passport['ocr_result']['patronymic_name'].endswith('ВНА') \
            or re.search(r'(ЖЕН|ЖЕН.) ', bottom) is not None:
        passport['ocr_result']['gender'] = 'female'

    # Looking for place of birth

    genders = ['МУЖ', 'МУЖ.', 'ЖЕН', 'ЖЕН.']
    birth_place = ''
    for word in bottom.split():
        if all(c.isupper() or c == '.' for c in word) and word not in name \
                and word not in genders and len(word) > 2:
            birth_place += word + ' '
    passport['ocr_result']['birth_place'] = birth_place

    # Looking for passport series

    series = re.search(r'(\d{2} \d{2})', number)
    if series is not None:
        passport['ocr_result']['series'] = series[0]

    # Looking for passport number

    num = re.search(r'(\d{6})', number)
    if num is not None:
        passport['ocr_result']['number'] = num[0]

    return passport
