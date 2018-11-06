import os
import cv2
import numpy as np

def blob(image):

    blobs = []
    maxInertia = 0

    while(len(blobs) <= 0):
        maxInertia += 0.01
        print("iner: " + str(maxInertia))

        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        # Remove noise
        kernel = np.ones((3, 3),np.uint8)
        # Erosion of the image.
        opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        Bparams = cv2.SimpleBlobDetector_Params()

        Bparams.minDistBetweenBlobs = 0

        Bparams.filterByArea = False
        Bparams.filterByCircularity = False
        Bparams.filterByConvexity = False
        Bparams.filterByInertia = True
        Bparams.maxInertiaRatio = maxInertia


        Btector = cv2.SimpleBlobDetector_create(Bparams)
        blobs = Btector.detect(opening)

    return maxInertia


basedir = os.getcwd()
files = os.listdir(basedir + "/testimg")

for x in files:

    image = cv2.imread("testimg/" + x, 0)

    val = blob(image)

    file = open(basedir + "/testimg/" + x + ".txt", "w")

    file.write("maxInertiaRatio: " + str(val))
    file.close()
