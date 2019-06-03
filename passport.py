from PIL import Image
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
    image = imutils.resize(image.copy(), width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rotates = 0
    # Looking for a face
    for _ in range(4):

        face = cascade.detectMultiScale(gray, 1.3, 5)

        if face is not ():
            return imutils.rotate_bound(image, 90 * rotates)

        gray = imutils.rotate_bound(gray, 90)
        rotates += 1

    # Return false if the given picture is not a passport
    return imutils.rotate_bound(image, 90)


def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


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


def cut_passport(image):
    """
    Cutting an image so only passport was left
    :param image: np array
    :return: np array
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(gray, 75, 200)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, rectKernel)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    hullImage = np.zeros(gray.shape[:2], dtype="uint8")

    max_area = 0
    for (i, c) in enumerate(cnts):

        area = cv2.contourArea(c)

        if area > max_area:

            max_area = area

            (x, y, w, h) = cv2.boundingRect(c)

            aspectRatio = w / float(h)

            extent = area / float(w * h)

            hull = cv2.convexHull(c)
            hullArea = cv2.contourArea(hull)

            solidity = area / float(hullArea)


            output = image.copy()
            cv2.drawContours(output, [c], -1, (240, 0, 159), 3)


    cv2.drawContours(hullImage, [hull], -1, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=hullImage)
    croped = get_segment_crop(image, mask=hullImage)
    croped = skew_text_correction(croped)

    return croped


def parse_name(full_name):
    return None


def read_text_from_box(image, startX, startY, endX, endY):

    box = image[startY:endY, startX: endX]

    box = cv2.resize(box, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(box.copy(), cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 1)
    thresh = cv2.threshold(blurred.copy(), 0, 225,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return image_to_string(thresh, lang='rus').replace('\n', ' ')


def find_person_name(ROI):

    bottom = ROI.copy()

    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, rectKernel)

    edged = cv2.erode(edged, (25, 25), iterations=2)
    edged = cv2.dilate(edged, (25, 25), iterations=2)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts,
                        method="top-to-bottom")[0]
    hullImage = np.zeros(gray.shape[:2], dtype="uint8")

    full_name = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / h

        if w > 100 and h > 100 and ar > 2.5:

            output = bottom.copy()

            full_name.append(image_to_string(output[y:y+h,x:x+w], lang='rus').replace('\n', ' '))

    result = []

    if len(full_name) >= 3:
        result.append([re.sub(r'[^а-яА-Я]+', '', f) for f in full_name
                        if f != '' and f is not None][:3])

    else:
        full_name = image_to_string(bottom, lang='rus')
        result.append(full_name)

    return full_name


def read_passport_number(image, show_steps=False):
    """
    Function for reading passport numbers by finding red zones
    :image: np.array
    :show_steps: bool
    :rtype: string
    """

    # Rotating the image
    image = imutils.rotate_bound(image, -90)
    number = ''

    if show_steps:
        cv2.imshow('Bottom', image)
        cv2.waitKey(0)


    # Creaing and aplying red mask on the image
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    masked = cv2.bitwise_and(image, image, mask=mask)

    if show_steps:
        cv2.imshow('Masked', masked)
        cv2.waitKey(0)


    # Deleting tresh
    kernel = np.ones((5, 5), np.uint8)
    masked = cv2.dilate(masked, kernel, iterations=3)

    if show_steps:
        cv2.imshow('Dilate', masked)
        cv2.waitKey(0)


    # Closing gaps betweeen letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)

    if show_steps:
        cv2.imshow('Closed', closed)
        cv2.waitKey(0)


    # Finding image contours
    closed = cv2.cvtColor(closed, cv2.COLOR_HSV2BGR)
    closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

    """
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts,
                    method="top-to-bottom")[0]

    # Reading text from bounding boxes
    for c in cnts:
        (gX, gY, gW, gH) = cv2.boundingRect(c)
        cv2.rectangle(image, (gX - 5, gY - 5),
                    (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)

        if show_steps:
            cv2.imshow('ROI', image)
            cv2.waitKey(0)

        number += read_text_from_box(image, gX, gY, gX + gW, gY + gH)

    """

    number += image_to_string(image, lang='rus')
    print(number)

    return number

def read_passport_mrz(image):

    image = imutils.resize(image, width=600)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                    ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    c = cnts[0]

    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    crWidth = w / float(gray.shape[1])

    """
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.03)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))
    """

    mrz = image[y:y + h, x:x + w].copy()

    gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
    return image_to_string(gray, lang='eng')


def procces_passport(full_name, top, bottom, number):
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
            'not_parsed_name': '',
            'birth_date': '',
            'gender': '',
            'birth_place': '',
            'series': '',
            'number': '',
        },
        'text': top + bottom + number,
    }

    # Looking for issue code
    code = re.search(r' \d{3}\.\d{3} ', top)
    if code is not None:
        passport['ocr_result']['issue_code'] = code[0]

    # Looking for issue authority
    passport['ocr_result']['issue_authority'] = re.search(r'\d{3}-\d{3}', top)[0]

    # Looking for issue date
    date = re.search(r'\d{2}\.\d{2}\.\d{4}', top)
    if date is not None:
        passport['ocr_result']['issue_date'] = date[0]

    # Looking for name

    if len(full_name) == 3:
        passport['ocr_result']['patronymic_name'] = full_name[0][-1]
        passport['ocr_result']['name'] = full_name[0][-2]
        passport['ocr_result']['surname'] = full_name[0][-3]

    passport['ocr_result']['not_parsed_name'] = full_name[1]

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
        if all(c.isalpha() or c == '.' for c in word) and word not in full_name[0] \
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
