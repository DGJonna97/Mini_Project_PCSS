import cv2
import os
import math
import numpy as np
import cvb

from BLOB import BLOB
from BLOB import getBlobs

path = cvb.install_path() + "Drivers/GenICam.vin"
device = cvb.DeviceFactory.open(path)
stream = device.stream
stream.start()


#Function for segmenting the image. Following simlifyed method by Jonatan in watershed
def segment(image):

    image = cv2.medianBlur(image, 11)

    #Exponential mapping
    k = 1.05

    c = 255 / (math.pow(k, np.max(image)) - 1)

    thresh = c * (np.power(k, image) - 1)

    _, thresh = cv2.threshold(thresh.astype(np.uint8), 1, 256, cv2.THRESH_BINARY)

    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3),np.uint8), iterations=2)

def withinRange(var, low, high):
    if(var < high and var > low):
        return True
    else:
        return False

def getDistance(vec1, vec2):
    if(len(vec1) != len(vec2)):
        print("Vector sizes do not match")
        return -1

    sum = 0

    for x in range(len(vec1)):
        sum += math.pow(vec2[x] - vec1[x], 2)

    return math.sqrt(sum)

def labelBlobs(image):
    seg = segment(image)

    #Getting connectedComponents from image
    _, components = cv2.connectedComponents(seg, connectivity=4)

    #Getting blobs from connectedComponents
    BLOBS = getBlobs(components)

    #Converting image to color since I want to draw colored boxed around blobs
    final = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #For every blob I evaluate if it looks like a human (Currently only Area)
    #And draws a rectangle around it if it is.
    for x in range(len(BLOBS)):
        blob = BLOBS[x]

        # The average of compactness is 0.4785
        # The average of circularity is 2.6173
        # The average of centerOfMass is (0.5073, 0.4675)

        distance = getDistance([blob.getCompactness(), blob.getCircularity(), blob.getCenterOfMass()[1]], [0.4785, 2.6173, 0.4675])

        if(distance < 1.3):

            x, y, w, h = blob.getRect()
            xCom, yCom = blob.getCenterOfMass()

            xCom = int((xCom * w) + x)
            yCom = int((yCom * h) + y)

            #Drawing bounding box
            cv2.rectangle(final, (x, y), (w+x, h+y), (0,255,0), 1)
            #Drawing Center Of Mass
            cv2.rectangle(final, (xCom-1, yCom-1), (xCom+1, yCom+1), (255,0,0), 1)

    return final, seg


#Getting the base directory for the project
basedir = os.getcwd()
#specifying which folder the image files are in and getting an array of files
files = os.listdir(basedir + "/cc_mat/testset")

#for evey file in the folder /cc_mat/trainingset try to load the image and find blobs
while True:
    img, status = stream.wait()
    inputImage = cvb.as_array(img, copy=False)

    image, seg = labelBlobs(inputImage)

    cv2.imshow("Image", image)
    cv2.waitKey(1)
