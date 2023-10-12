import numpy as np
import cv2
import imutils
import easyocr
import sys
import pytesseract
import pandas as pd
import time
import cv2imshow
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

harcascade = "model/haarcascade_russian_plate_number.xml"


def convertText(img):
    image = cv2.imread(img)
    print("Detected")
    # image = imutils.resize(image, width=500)
    # cv2.imshow("Original Image", image)

    # convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Conversion", gray)

    # blur to reduce noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # cv2.imshow("Bilateral Filter", gray)

    # perform edge detection
    # (og)edged = cv2.Canny(gray, 170, 200)
    edged = cv2.Canny(gray, 30, 200)
    # cv2.imshow("Canny Edges", edged)

    # (og) cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # (og) find contours in the edged image
    # (cnts, _) = cv2.findContours(edged.copy(),
    #                              cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    keypoints = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    print(location)

    # NumberPlateCnt = None
    # count = 0
    # loop over contours
    # for c in cnts:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     # if the approximated contour has four points, then assume that screen is found
    #     if len(approx) == 4:
    #         NumberPlateCnt = approx
    #         break
    # cv2.imshow("Original Image", image)

    # mask the part other than the number plate
    # (og)mask = np.zeros(gray.shape, np.uint8)
    # new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    # new_image = cv2.bitwise_and(image, image, mask=mask)
    # cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Final Image", new_image)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print(result[0][-2])
    # configuration for tesseract
    # config = ('-l eng --oem 1 --psm 3')

    # # run tesseract OCR on image
    # text = pytesseract.image_to_string(new_image, config=config)

    # data is stored in CSV file
    raw_data = {'date': [time.asctime(time.localtime(time.time()))], '': [
        result[0][-2]]}
    df = pd.DataFrame(raw_data)
    df.to_csv('data.csv', mode='a')

    # print recognized text
    # print(text)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


cap = cv2.VideoCapture(0)

cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
count = 0

while True:
    success, img = cap.read()
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # cropping number plate
            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # save the image
        cv2.imwrite("plates/img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        convertText("plates/img_" + str(count) + ".jpg")
        count += 1


# press s to save the image
# press ctrl+c to exit
