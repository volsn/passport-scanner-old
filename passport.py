import cv2
import imutils
from imutils import rotate_bound
from imutils.object_detection import non_max_suppression
from imutils.contours import sort_contours
import os
import re
import math
import random
import numpy as np
from pytesseract import image_to_string


def passport_border(image, mode='reading'):

    # Initializing cascade
    #image = cv2.imread(filename)
    #image = imutils.resize(image, width=1000)
    cascade = cv2.CascadeClassifier('cascade.xml')
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Finding a face
    face = cascade.detectMultiScale(gray, 1.3, 5)

    if face is not ():
        # Cutting the image so only passport was left
        (x, y, w, h) = face[0]

        (H, W, _) = image.shape

        if mode == 'reading':
            # Mode designed to find passport border
            # as acurate as possible in order to read it

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

            mask = np.zeros((H, W), dtype=np.uint8)
            mask[startY:endY, startX:endX] = 255

            masked = cv2.bitwise_and(image, image, mask=mask)
            masked = get_segment_crop(image, mask=mask)

            """
                cv2.imwrite(os.path.join('output', output + '.png'), masked)

            else:
                cv2.imwrite(os.path.join('output', output + '.png'), image)
            """

            return masked

    else:
        return image


def rotate_passport(passport):
    """
    rotating an image so passport could be readed
    :param image: np array
    :return: np array
    """

    # Initializing cascade
    cascade = cv2.CascadeClassifier('cascade.xml')
    image = imutils.resize(passport.copy(), width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rotates = 0
    # Looking for a face
    for _ in range(4):

        face = cascade.detectMultiScale(gray, 1.3, 5)

        if face is not ():
            return imutils.rotate_bound(passport, 90 * rotates)

        gray = imutils.rotate_bound(gray, 90)
        rotates += 1

    (h, w, _) = passport.shape
    if w > h:
        passport = rotate_bound(passport,angle=-90)

    print('Falsed')
    # Return false if the given picture is not a passport
    return passport


def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def cut_passport(image):

    image = image.copy()
    # image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    blended = cv2.addWeighted(src1=sobelX, alpha=0.5, src2=sobelY, beta=0.5, gamma=0)

    kernel = np.ones((20, 20), dtype=np.uint8)
    opening = cv2.morphologyEx(blended, cv2.MORPH_OPEN, kernel)

    min_ = np.min(opening)
    opening = opening - min_
    max_ = np.max(opening)
    div = max_/255
    opening = np.uint8(opening / div)

    blurred = cv2.GaussianBlur(opening, (1, 1), 0)
    thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    (h, w) = thresh.shape
    edgeH = int(h * 0.01)
    edgeW = int(w * 0.01)
    thresh[0:edgeH,0:w] = 255
    thresh[h-edgeH:h,0:w] = 255
    thresh[0:h,0:edgeW] = 255
    thresh[0:h, w-edgeW:w] = 255

    kernel = np.ones((20, 20), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    inverse = cv2.bitwise_not(thresh)

    """
    coords = np.column_stack(np.where(inverse > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    image = rotate_bound(image,angle=angle)
    inverse = rotate_bound(inverse,angle=angle)"""

    masked = get_segment_crop(image, mask=inverse)
    """
    if not os.path.exists('output1'):
        os.mkdir('output1')

    cv2.imwrite(os.path.join('output1', output + '.png'), masked)
    """
    return masked


def locate_text(image):

    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.01:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    mask = np.zeros(orig.shape[:2],dtype=np.uint8)
    ROIs = []
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    boxes = sorted(boxes,key=lambda x:x[1])

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        w = endX - startX
        h = endY - startY

        if w > 70 and h > 30:

            endX += 100
            startX -= 100
            startY -= 5
            endY += 5

            #cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

            mask[startY:endY,startX:endX] = 255
            ROIs.append(orig[startY:endY,startX:endX].copy())

    img_cnt, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    text = ''
    if len(cnts) > 0:

        (cnts, boundingBoxes) = sort_contours(cnts, method='top-to-bottom')

        ROIs = []
        for cnt, box in zip(cnts, boundingBoxes):

            temp_mask = np.zeros(mask.shape[:2],dtype=np.uint8)
            #temp_mask = cv2.drawContours(temp_mask, [cnt], -1, (255,255,255), -1)

            (x, y, w, h) = box
            y -= 15
            h += 30

            temp_mask[y:y+h,x:x+w] = 255


            roi = get_segment_crop(orig, orig, mask=temp_mask)
            ROIs.append(roi)

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            roi_text = image_to_string(thresh, lang='rus')

            text += roi_text + '\n'

        masked = cv2.bitwise_and(orig,orig,mask=mask)
    return text


def read_passport(image):

    image = image.copy()
    (h,w) = image.shape[:2]

    authority = image[0:h//2, 0:w]
    name = image[h//2:h//4*3, w//3:w]
    birth_place = image[h//4*3:h, 0:w]

    gray_authority= cv2.cvtColor(authority, cv2.COLOR_BGR2GRAY)
    gray_name = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
    gray_birth_place = cv2.cvtColor(birth_place, cv2.COLOR_BGR2GRAY)

    ret, thresh_authority = cv2.threshold(gray_authority, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, thresh_name = cv2.threshold(gray_name, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, thresh_birth_place = cv2.threshold(gray_birth_place, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    authority = image_to_string(authority, lang='rus').replace('\n', ' ')
    name = image_to_string(name, lang='rus').replace('\n', ' ')
    birth_place = image_to_string(birth_place, lang='rus').replace('\n', ' ')
    raw = image_to_string(image, lang='rus').replace('\n', ' ')

    side = imutils.rotate_bound(image, angle=-90)
    (h, w) = side.shape[:2]
    side = side[0:h//10, 0:w]
    gray_side = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
    ret, thresh_side = cv2.threshold(gray_side, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    number = image_to_string(thresh_side, lang='rus')

    #(authority, name, birth_place) = preprocess_text(authority, name, birth_place)

    return (authority, name, birth_place, raw, number)


def preprocess_text(authority, name, birth_place, name_backend):

    authority = re.sub(r'[^А-Я- \.\d]+', '', authority)
    name_backend = re.sub(r'[^А-Я ]+', '', name_backend)
    name = re.sub(r'[^А-Я ]+', '', name)
    birth_place = re.sub(r'[^а-яА-Я- \.]+', '', birth_place)


    ALLOLEW_SMALL_STRINGS_AUTHORITIES = ['и', 'в']
    authority = authority.split()
    for i, word in enumerate(authority):
        if len(word) <= 2 and word.lower() not in ALLOLEW_SMALL_STRINGS_AUTHORITIES:
            del(authority[i])

    name = name.split()
    for i, word in enumerate(name):
        if len(word) <= 2:
            del(name[i])

    ALLOLEW_SMALL_STRINGS_BIRTH_PLACE = []
    birth_place = birth_place.split()
    for i, word in enumerate(birth_place):
        if len(word) <= 2 and word.lower() not in ALLOLEW_SMALL_STRINGS_BIRTH_PLACE:
            del(birth_place[i])

    authority = ''.join('{} '.format(word) for word in authority)
    birth_place = ''.join('{} '.format(word) for word in birth_place)
    name = ''.join('{} '.format(word) for word in name)

    return (authority, name, birth_place, name_backend)


def read_text(image):

    image = image.copy()
    (h, w) = image.shape[:2]

    bottom_image = image[h//2:h,w//3:w]
    bottom_processed = locate_text(bottom_image)
    top_image = image[0:h//2,0:w]
    top_processed = locate_text(top_image)

    (authority, name, birth_place, raw, number) = read_passport(image)

    return (bottom_image, top_processed, authority, name, birth_place, raw, number)


def parse_name(name, backend, raw):

    #parse namev
    result1 = {"surname":'',"name":'',"patronymic_name":''}
    full_name = re.search(r'(.* (.*(ВИЧ|ВНА|ВНЯ)))', name, flags=re.I)
    if full_name is not None:

        result1['patronymic_name'] = full_name[2]
        full_name = full_name[0].split()
    else:
        full_name = name.split()

    for word in full_name:
        if re.match(r'[А-Я]{2,}(ОВ|ОВА|ЕВ|ЕВА|ЁВ|ЁВА|ИХ|ЫХ|ИЙ|ЫЙ|АЯ|КО|АЙ|ИК|УК|ЮК|ЕЦ|ЛО|ИН|ИНА|УН)$', word):
            result1['surname'] = word

    for word in full_name:
        if word != result1['surname'] and word != result1['patronymic_name']:
            result1['name'] = word


    # parse backend
    result2 = {"surname":'',"name":'',"patronymic_name":''}
    full_name = re.search(r'(.* (.*(ВИЧ|ВНА|ВНЯ)))', backend, flags=re.I)
    if full_name is not None:

        result2['patronymic_name'] = full_name[2]
        full_name = full_name[0].split()
    else:
        full_name = backend.split()

    for word in full_name:
        if re.match(r'[А-Я]{2,}(ОВ|ОВА|ЕВ|ЕВА|ЁВ|ЁВА|ИХ|ЫХ|ИЙ|ЫЙ|АЯ|КО|АЙ|ИК|УК|ЮК|ЕЦ|ЛО|ИН|ИНА|УН)$', word):
            result2['surname'] = word

    for word in full_name:
        if word != result1['surname'] and word != result1['patronymic_name']:
            result2['name'] = word

    #parse raw
    result3 = {"surname":'',"name":'',"patronymic_name":''}
    full_name = re.search(r'(.* (.*(ВИЧ|ВНА|ВНЯ)))', raw, flags=re.I)
    if full_name is not None:

        result3['patronymic_name'] = full_name[2]
        full_name = full_name[0].split()
    else:
        full_name = name.split()

    for word in full_name:
        if re.match(r'[А-Я]{2,}(ОВ|ОВА|ЕВ|ЕВА|ЁВ|ЁВА|ИХ|ЫХ|ИЙ|ЫЙ|АЯ|КО|АЙ|ИК|УК|ЮК|ЕЦ|ЛО|ИН|ИНА|УН)$', word):
            result3['surname'] = word

    for word in full_name:
        if word != result1['surname'] and word != result1['patronymic_name']:
            result3['name'] = word

    print(raw)

    if result1['surname'] is None and result2['surname'] is not None:
        result1['surname'] = result2['surname']

    if result1['name'] is None and result2['name'] is not None:
        result1['name'] = result2['name']


    if result1['surname'] is None and result3['surname'] is not None:
        result1['surname'] = result3['surname']

    if result1['name'] is None and result3['name'] is not None:
        result1['name'] = result3['name']

    return result1


def parse_passport(authority, name, birth_place, raw, number, name_backend):

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
        'text': '',
        'FIO': '',
    }

    # Looking for dates of issue and birth
    date_zone1 = authority.replace(' ', '')
    date1 = re.findall(r'(\d{2}[^\d]{1,3}\d{2}[^\d]{1,3}\d{4})', date_zone1)
    if date1 != []:
        passport['ocr_result']['issue_date'] = date1[0]

    date_zone2 = raw.replace(' ', '')
    date2 = re.findall(r'(\d{2}[^\d]{1,3}\d{2}[^\d]{1,3}\d{4})', date_zone2)
    if date2 != []:
        passport['ocr_result']['birth_date'] = date2[0]

    # Looking for issue code
    code = re.search(r'\d{3}-\d{3}', authority)
    if code is not None:
        passport['ocr_result']['issue_code'] = code[0]

    #authority = re.sub(r'[^А-Я- \.]+', '', authority)

    AUTHORITIES = ['отделом', 'УФМС', 'МФЦ', 'ГОМ', 'УВД']
    issued = None
    for auth in AUTHORITIES:
        if re.search(auth, authority, flags=re.I) is not None:
            issued = re.findall(r'({}.*)'.format(auth), authority, flags=re.I)[0]
            break

    if issued is None:
        issued = authority
    passport['ocr_result']['issue_authority'] = issued


    LOCALITIES = ['пос', 'гор', r'с\.']
    born = None
    for local in LOCALITIES:
        if re.search(local, birth_place, flags=re.I) is not None:
            born = re.findall(r'({}.*)'.format(local), birth_place, flags=re.I)[0]
            break

    if born is None:
        born = birth_place
    passport['ocr_result']['birth_place'] = born


    FIO = parse_name(name, name_backend, raw)
    passport['ocr_result']['patronymic_name'] = FIO['patronymic_name']
    passport['ocr_result']['name'] = FIO['name']
    passport['ocr_result']['surname'] = FIO['surname']


    # Looking for gender
    if passport['ocr_result']['patronymic_name'].endswith('ВИЧ') \
                        or re.search(r'(МУЖ|МУЖ.) ', raw) is not None:
        passport['ocr_result']['gender'] = 'male'
    elif passport['ocr_result']['patronymic_name'].endswith('ВНА') \
                        or re.search(r'(ЖЕН|ЖЕН.) ', raw) is not None:
        passport['ocr_result']['gender'] = 'female'

    # Looking for passport series
    series = re.search(r'(\d{2} \d{2})', number)
    if series is not None:
        passport['ocr_result']['series'] = series[0]

    # Looking for passport number
    num = re.search(r'(\d{6})', number)
    if num is not None:
        passport['ocr_result']['number'] = num[0]

    passport['text'] = authority + name + birth_place + raw + number

    return passport


def analyze_passport(image):

    orig = image.copy()
    orig = cut_passport(orig)
    orig = rotate_passport(orig)
    passport = passport_border(orig)

    (authority, name_backend, birth_place, raw, number) = read_passport(passport)
    #instance1 = parse_passport(authority, name, birth_place, raw, number)

    #result['1'] = instance1

    (h,w) = passport.shape[:2]

    name = passport[h//2:h,w//3:w]
    text_name = locate_text(name).replace('\n', ' ')

    """top = passport[0:h//2,0:w]
    text_top = locate_text(top).replace('\n', ' ')

    birth_place = passport[h//4*3:h,0:w]
    text_birth_place = locate_text(birth_place).replace('\n', ' ')"""

    """raw = passport[h//2:h,0:w]
    text_raw = locate_text(raw).replace('\n', ' ')"""

    (authority, text_name, birth_place, name_backend) = preprocess_text(authority, text_name, birth_place, name_backend)
    print(name_backend)
    print(text_name)

    result = parse_passport(authority, text_name, birth_place, raw, number, name_backend)

    return result
