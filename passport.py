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

def read_name(mask, bottom):

    (H,W) = bottom.shape[:2]

    img_cnt, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    orig = bottom.copy()

    text = []
    masked = np.zeros(orig.shape[:2],dtype=np.uint8)
    if len(cnts) >= 3:

        (cnts, boundingBoxes) = sort_contours(cnts, method='top-to-bottom')

        ROIs = []
        for j, (cnt, box)in enumerate(zip(cnts, boundingBoxes)):

            temp_mask = np.zeros(mask.shape[:2],dtype=np.uint8)
            temp_mask = cv2.drawContours(temp_mask, [cnt], -1, (255,255,255), -1)

            (x, y, w, h) = box

            if x >= 0.1 * W:

                """x -= int(0.1 * w)
                w += int(0.1 * w) * 2"""
                x -= 100
                w += 200
                y -= int(0.5 * h)
                h += int(0.5 * h) * 2

                temp_mask[y:y+h,x:x+w] = 255

                #cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)

                roi = get_segment_crop(orig, orig, mask=temp_mask)
                ROIs.append(roi)

                """cv2.imwrite('Mask/{}.png'.format(name), orig)
                cv2.imwrite('ROIs/{}_{}.png'.format(name,j),roi)
                """

                roi_text = read_text_from_blob(roi)

                text.append(roi_text + '\n')

    #print(text)
    result = []
    for word in text[:3]:
        word = word.replace('\n', ' ')
        word = re.sub(r'[^а-яА-Я ]+', '', word)
        potentials = word.split()
        if potentials != []:
            result.append(sorted(potentials, key=len)[-1])

    return result

def locate_text(image):

    orig = image.copy()
    (H, W) = image.shape[:2]

    gray = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    raw_text = image_to_string(thresh, lang='rus')

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    net = cv2.dnn.readNet('EAST.pb')

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

        #if w > 50 and h > 15:

        endX += 25
        startX -= 25
        """startY -= 5
        endY += 5"""

        mask[startY:endY,startX:endX] = 255
        ROIs.append(orig[startY:endY,startX:endX].copy())

    located = cv2.bitwise_and(orig, orig, mask=mask)

    return mask, located

def read_text_from_blob(image):

    roi = image.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray,3)
    ret, thresh = cv2.threshold(blurred,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    roi_text = image_to_string(thresh, lang='rus')

    if roi_text == '':
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(5,5),0)
        ret, thresh = cv2.threshold(blurred,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        roi_text = image_to_string(thresh, lang='rus')

    if roi_text == '':
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(1,1),0)
        dilated = cv2.dilate(blurred, (3,3), iterations=1)
        ret, thresh = cv2.threshold(dilated,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        roi_text = image_to_string(thresh, lang='rus')

    return roi_text

def analyze_passport(image):

    orig = image.copy()
    orig = cut_passport(orig)
    orig = rotate_passport(orig)
    passport = passport_border(orig)

    (H,W) = passport.shape[:2]

    bottom = passport[H//2:H,W//3:W]
    mask,located = locate_text(bottom)

    pasport = {'name': '', 'surname': '', 'patronymic_name': '', 'authority': ''}

    full_name = read_name(mask, bottom)
    """print(full_name)
    if len(full_name) == 3:
        passport['surname'] = full_name[0]
        passport['name'] = full_name[1]
        passport['patronymic_name'] = full_name[2]
    else:
        passport['FIO'] = full_name"""

    return full_name

    """
    # TODO Read Authority
    """

    """
    # TODO Parse Numerical Data
    """
