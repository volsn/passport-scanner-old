from PIL import Image
import pdf2image
import imutils
from imutils import contours
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

    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h),
                  (0, 0, 255), 2)

    """
    cv2.imshow('Output', output)
    cv2.waitKey(0)
    """

    (H, W, _) = image.shape

    if y - int(6 * h) < 0:
        startY = 0
    else:
        startY = y - int(6 * h)

    if y + 3 * h > H:
        endY = H
    else:
        endY = y + 3 * h

    if x - w < 0:
        startX = 0
    else:
        startX = x - w

    if x + 6 * w > W:
        endX = W
    else:
        endX = x + 6 * w

    image = image[startY:endY, startX:endX]

    """
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    """

    return image


def skew_text_correction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    """
    cv2.imshow('Thresh', thresh)
    cv2.waitKey(0)
    """

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    """
    cv2.imshow("Input", image)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    """

    return rotated


def locate_text(image, type_):
    if type_ == 'top':
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    elif type_ == 'bottom':
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    # image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    """
    cv2.imshow('Tophat', tophat)
    cv2.waitKey(0)
    """

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                      ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    """
    cv2.imshow('Gradient', gradX)
    cv2.waitKey(0)
    """

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    """
    cv2.imshow('Thresh', thresh)
    cv2.waitKey(0)
    """

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts,
                    method="top-to-bottom")[0]

    locs = []

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if w > 10 and h > 10 and ar > 2.5:
            locs.append((x, y, w, h))

    output = []
    text = ''

    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        groupOutput = []

        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


        """
        cv2.rectangle(image, (gX - 5, gY - 5),
                        (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        """

        text += read_text_from_box(image, gX - 5, gY - 5,
                                   gX + gW + 5, gY + gH + 5) + ' '

        """
        cv2.imshow('ROI', image)
        cv2.waitKey(0)
        """

    return text


def read_text_from_box(image, startX, startY, endX, endY):
    """
    Reading text from bounding box
    :param image: np array
    :return: string
    """

    box = image[startY:endY, startX: endX]

    """
    cv2.imshow('ROI', box)
    cv2.waitKey(0)
    """

    box = cv2.resize(box, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(box.copy(), cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 1)
    thresh = cv2.threshold(blurred.copy(), 0, 225,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return image_to_string(thresh, lang='rus').replace('\n', ' ')


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
